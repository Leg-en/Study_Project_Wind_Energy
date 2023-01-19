import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
#from pymoo.operators.crossover.pntx import TwoPointCrossover
#from pymoo.operators.mutation.bitflip import BitflipMutation
#from pymoo.operators.sampling.rnd import BinaryRandomSampling
#from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

reduced = True  # Das sind im Worst Case immer noch 40765935 Mögliche Kombinationen mit dem verkleinerten gebiet..
RUN_LOCAL = False
POOL_SIZE = 10
SMART_REPAIR = True
PROFIT_FIVE_YEARS = 7.79 # Angabe in ct pro kW/h für die ersten 5 Jahre nach Installation
PROFIT_LATER_YEARS = 4.25 # Angabe in ct pro kW/h nach den ersten 5 Jahren nach Installation


cell_size = 100
timeString = "03:50:00"

# Pfade müssen angepasst werden
USER = 'Emily'

if USER == 'Emily':
    if RUN_LOCAL:
        if reduced:
            points_path = fr"C:\workspace\Study_Project_Wind_Energy\data\processed_data_{cell_size}cell_size_reduced\numpy_array\points_{cell_size}.npy"
        else:
            points_path = fr"C:\workspace\Study_Project_Wind_Energy\data\processed_data_{cell_size}cell_size\numpy_array\points_{cell_size}.npy"
        WKA_data_path = r"C:\workspace\Study_Project_Wind_Energy\base_information_enercon_reformatted.json"
    else:
        if reduced:
            points_path = fr"/scratch/tmp/m_ster15/points_{cell_size}_reduced.npy"
        else:
            points_path = fr"/scratch/tmp/m_ster15/points_{cell_size}.npy"
        WKA_data_path = r"/home/m/m_ster15/WindEnergy/base_information_enercon_reformatted.json"
elif USER == 'Josefina':
    if RUN_LOCAL:
        points_path = fr"/Users/josefinabalzer/Desktop/WS22_23/Study_Project/Study_Project_Wind_Energy/data/points_{cell_size}.npy"
        WKA_data_path = r"/Users/josefinabalzer/Desktop/WS22_23/Study_Project/Study_Project_Wind_Energy/base_information_enercon_reformatted.json"
    else:
        points_path = fr"/scratch/tmp/jbalzer/Study_Project/data/points_{cell_size}.npy"
        WKA_data_path = r"/home/j/jbalzer/Study_Project_Wind_Energy/base_information_enercon_reformatted.json"

