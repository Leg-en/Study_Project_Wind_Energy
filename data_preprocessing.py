import os
import arcpy
import geopandas as gpd
import numpy as np
import shapefile
from arcpy import analysis
from arcpy import conversion
from arcpy import management
import json
from tqdm import tqdm
from glob import glob

# Hier entsprechend eigene Pfade (/Variablen) Setzen
cell_size = 20
WKA_data_path = r"C:\workspace\Study_Project_Wind_Energy\base_information_enercon_reformatted.json"
processed_data = r"C:\workspace\Study_Project_Wind_Energy\data\processed_data_" + str(cell_size) + "cell_size"

with open(WKA_data_path, "r") as f:
    WKA_data = json.load(f)

acc_data = None

windklasse = WKA_data["area_information"]["wind_class"]

arcpy.env.workspace = r"C:\workspace\MasterSemester1\WindEnergy\Project\data\study area"
arcpy.env.parallelProcessingFactor = "100%"
arcpy.env.outputCoordinateSystem = 'PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Mercator_Auxiliary_Sphere"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",0.0],PARAMETER["Standard_Parallel_1",0.0],PARAMETER["Auxiliary_Sphere_Type",0.0],UNIT["Meter",1.0]]'

numpy_array = processed_data + r"\numpy_array"
os.mkdir(processed_data)
os.mkdir(numpy_array)

# flurstuecke = r"flurstuecke.shp"
flurstuecke = r"C:\workspace\MasterSemester1\WindEnergy\Project\ArcGIS Project\WindEnergy.gdb\Flurstuecke_Area_Intersect"
potential_areas = r"potential_areas_lpa_400m.shp"
hausumringe = r"hausumringe.shp"
wege = r"strassen_und_wege.shp"

for WKA in tqdm(WKA_data["turbines"]):
    if windklasse not in WKA["height"]["wind_classes"]:
        print("WKA Verfügt nicht über die Windklasse")
        continue

    s_path = processed_data + r"\type_" + WKA["type"].replace(" ", "_")
    os.mkdir(s_path)
    s_path_gdb = processed_data + r"\type_" + WKA["type"].replace(" ", "_") + ".gdb"
    arcpy.CreateFileGDB_management(processed_data, r"\type_" + WKA["type"].replace(" ", "_") + ".gdb")

    # Berechnet den Buffer für die Häuser. Es wird immer die n fache WKA Höhe genommen. in diesem falle ist n=2
    buffer_size = WKA["height"]["hub_height_in_meter"][windklasse][0]
    haus_buffer_path = s_path_gdb + r"\haus_buffer"
    analysis.PairwiseBuffer(hausumringe,
                            haus_buffer_path,
                            str(buffer_size * 2) + " Meters", "NONE", None, "PLANAR", "0 Meters")

    # Berechnet Buffer zu wegen. Der buffer ist immer 30 Meter
    wege_buffer_path = s_path_gdb + r"\wege_buffer"
    analysis.PairwiseBuffer(wege,
                            wege_buffer_path,
                            "30 Meters", "NONE", None, "PLANAR", "0 Meters")

    # Berechnet einen buffer so das der Rotor immer innerhalb der planfläche ist
    diameter = WKA["rotor_diameter_in_meter"]
    radius = diameter / 2
    rotor_in_area_buffer_path = s_path_gdb + r"\rotor_in_area_buffer"
    analysis.PairwiseBuffer(potential_areas,
                            rotor_in_area_buffer_path,
                            str(-radius) + " Meters", "NONE", None, "PLANAR", "0 Meters")

    # Die folgenden beiden stellen sicher das 50meter + die fundament größe abstand zu flurstücksgrenzen gehalten werden.
    # Erst flurstücke buffern
    flurstuecke_buffered_path = s_path_gdb + r"\flurstuecke_buffered"
    buffer_size = 50 + (WKA["fundament_diameter_in_meter"]/2)  # Annahme das die Fundament size den Radius wiedergibt
    analysis.PairwiseBuffer(flurstuecke,
                            flurstuecke_buffered_path,
                            str(buffer_size) + " Meters", "NONE", None, "PLANAR", "0 Meters")

    # dann mit sich selbst intersecten
    # Anscheinend kommt dieses, aber auch nur dieses, tool nicht mit leerzeichen im dateinamen klar...
    flurstuecke_buffered_self_intersected_path = s_path_gdb + r"\flurstuecke_buffered_self_intersected"
    analysis.PairwiseIntersect(in_features=flurstuecke_buffered_path,
                               out_feature_class=flurstuecke_buffered_self_intersected_path,
                               join_attributes="ALL", cluster_tolerance=None, output_type="INPUT")

    flurstuecke_erased_path = s_path_gdb + r"\flurstuecke_erased"
    analysis.PairwiseErase(rotor_in_area_buffer_path, flurstuecke_buffered_self_intersected_path,
                           flurstuecke_erased_path,
                           None)

    flurstuecke_wege_erased_path = s_path_gdb + r"\flurstuecke_wege_erased"
    analysis.PairwiseErase(flurstuecke_erased_path, wege_buffer_path,
                           flurstuecke_wege_erased_path,
                           None)
    flurstuecke_wege_hauser_erased_path = s_path_gdb + r"\flurstuecke_wege_hauser_erased"
    arcpy.analysis.PairwiseErase(flurstuecke_wege_erased_path, haus_buffer_path,
                                 flurstuecke_wege_hauser_erased_path,
                                 None)

    flurstuecke_wege_hauser_erased_path_shapefile = s_path + r"\flurstuecke_wege_hauser_erased"
    # Exportieren in shapefile um die koordinaten auszulesen
    conversion.ExportFeatures(flurstuecke_wege_erased_path, flurstuecke_wege_hauser_erased_path_shapefile, '',
                              "NOT_USE_ALIAS", None, None)

    sf = shapefile.Reader(flurstuecke_wege_hauser_erased_path_shapefile)
    originCoordinate = "" + str(sf.bbox[0]) + " " + str(sf.bbox[1])
    yAxisCoordinate = "" + str(sf.bbox[0]) + " " + str(sf.bbox[1] + 10)
    oppositeCoorner = "" + str(sf.bbox[2]) + " " + str(sf.bbox[3])
    labels = 'LABELS'
    # Extent is set by origin and opposite corner - no need to use a template fc
    templateExtent = '#'
    geometryType = 'POLYLINE'
    sf.close()

    files = glob(s_path + r"\*")
    for file in files:
        os.remove(file)

    # Fishnet erstellen
    fishnet_path = s_path_gdb + r"\fishnet"
    management.CreateFishnet(fishnet_path, originCoordinate, yAxisCoordinate, cell_size, cell_size, None, None,
                             oppositeCoorner,
                             labels, templateExtent, geometryType)

    intersection_points = s_path_gdb + r"\inter_points"
    analysis.PairwiseClip(fishnet_path + "_label", flurstuecke_wege_erased_path,
                          intersection_points,
                          None)

    # Erstellt ein neues Feld und setzt dort den WKA Typ
    management.AddField(intersection_points, "WKA_TYPE", "TEXT")
    management.CalculateField(
        intersection_points,
        "WKA_TYPE", "'" + WKA["type"].replace(" ", "_") + "'")

    points_shapefile_path = s_path + r"\points_shape"
    conversion.ExportFeatures(intersection_points, points_shapefile_path, '', "NOT_USE_ALIAS", None, None)

    df = gpd.read_file(points_shapefile_path + ".shp")
    df_np = df.to_numpy()
    if acc_data is not None:
        acc_data = np.concatenate((acc_data, df_np))
    else:
        acc_data = df_np

with open(numpy_array + "\points_" + str(cell_size) + ".npy",
          "wb") as f:
    np.save(f, acc_data)
