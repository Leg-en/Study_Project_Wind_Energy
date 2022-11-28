import os
import arcpy
import geopandas as gdf
import numpy as np
import shapefile
from arcpy import analysis
from arcpy import conversion
from arcpy import management

cell_size = 5

arcpy.env.workspace = r"C:\workspace\MasterSemester1\WindEnergy\Project\data\study area"
arcpy.env.parallelProcessingFactor = "100%"
arcpy.env.outputCoordinateSystem = 'PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Mercator_Auxiliary_Sphere"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",0.0],PARAMETER["Standard_Parallel_1",0.0],PARAMETER["Auxiliary_Sphere_Type",0.0],UNIT["Meter",1.0]]'

processed_data = r"C:\workspace\MasterSemester1\WindEnergy\Project\data\processed_data_" + str(cell_size) + "cell_size"
os.mkdir(processed_data)

flurstuecke = r"flurstuecke.shp"
potential_areas = r"potential_areas_lpa_400m.shp"
hausumringe = r"hausumringe.shp"

Clipped = processed_data + r"\clipped_data"
analysis.PairwiseClip(flurstuecke, potential_areas, Clipped)

# Erfüllt constraint mit 15m abstand zur flurstücksgrenze
Area_Flurstuecke_bufferd = processed_data + r"\bestehend"
analysis.PairwiseBuffer(Clipped, Area_Flurstuecke_bufferd, "-15 Meters", "NONE", None, "PLANAR", "0 Meters")

# Bufferd_hausumringe = processed_data + r"\bufferd_data"
# analysis.PairwiseBuffer(hausumringe, Bufferd_hausumringe, "50 Meters", "NONE", None, "PLANAR", "0 Meters")
#
# House_Erased_data = processed_data + r"\house_erased"
# analysis.PairwiseErase(Area_Flurstuecke_bufferd, Bufferd_hausumringe, House_Erased_data)


# Ab hier ist die erfüllung des Constraints von mindestens 50m abstand zu einander
bestehend_buffered = processed_data + r"\bestehend_buffered"
analysis.PairwiseBuffer(potential_areas, bestehend_buffered, "50 Meters", "NONE", None, "PLANAR", "0 Meters")

buffered_intersects = processed_data + r"\buffered_intersects"
analysis.PairwiseIntersect(bestehend_buffered, buffered_intersects, "ALL", None, "INPUT")

Intersects_erased = processed_data + r"\Intersects_erased"
analysis.PairwiseErase(Area_Flurstuecke_bufferd, buffered_intersects, Intersects_erased, None)

# Points über das gesamte ausmaß ansetzen und dann entsprechend Clippend. Deutlich Simpler, stärker Limitiert. Daher grobere zellgröße

sf = shapefile.Reader(Intersects_erased)
originCoordinate = "" + str(sf.bbox[0]) + " " + str(sf.bbox[1])
yAxisCoordinate = "" + str(sf.bbox[0]) + " " + str(sf.bbox[1] + 10)
oppositeCoorner = "" + str(sf.bbox[2]) + " " + str(sf.bbox[3])
labels = 'LABELS'
# Extent is set by origin and opposite corner - no need to use a template fc
templateExtent = '#'
geometryType = 'POLYLINE'

gdb_path = processed_data + r"\result.gdb"
arcpy.CreateFileGDB_management(processed_data, "result.gdb")

points = gdb_path + r"\points"
management.CreateFishnet(points, originCoordinate, yAxisCoordinate, cell_size, cell_size, None, None, oppositeCoorner,
                         labels, templateExtent, geometryType)

intersection_points = gdb_path + r"\inter_points"
analysis.PairwiseClip(points + "_label", Intersects_erased,
                      intersection_points,
                      None)

points_final = processed_data + r"\final_points"
conversion.ExportFeatures(intersection_points, points_final, '', "NOT_USE_ALIAS", None, None)

df = gdf.read_file(points_final + ".shp")
dfnp = df["geometry"].to_numpy()
with open(r"C:\workspace\MasterSemester1\WindEnergy\Project\data\numpy_arr\cell_size" + str(cell_size) + ".npy",
          "wb") as f:
    np.save(f, dfnp)

# Todo: Fundamente mit einbeziehen, Abstände zu gebäuden mit einbeziehen
