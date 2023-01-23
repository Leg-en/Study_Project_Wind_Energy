import json
import pickle
from itertools import combinations
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
import multiprocessing
from pymoo.core.problem import StarmapParallelization
from pymoo.visualization.scatter import Scatter
import logging
import sys
from pymoo.core.repair import Repair
from pymoo.termination import get_termination

repair_mode = True
reduced = True  # Das sind im Worst Case immer noch 40765935 Mögliche Kombinationen mit dem verkleinerten gebiet..
RUN_LOCAL = False

cell_size = 100
timeString = "23:50:00"

# Pfade müssen angepasst werden
USER = 'Emily'

if USER == 'Emily':
    if RUN_LOCAL:
        if reduced:
            points_path = fr"C:\workspace\Study_Project_Wind_Energy\data\processed_data_{cell_size}cell_size_reduced\numpy_array\points_{cell_size}.npy"
        else:
            points_path = fr"C:\workspace\Study_Project_Wind_Energy\data\processed_data_{cell_size}cell_size\numpy_array\points_{cell_size}.npy"
        WKA_data_path = r"../base_information_enercon_reformatted.json"
    else:
        if reduced:
            points_path = fr"/scratch/tmp/m_ster15/points_{cell_size}_reduced.npy"
        else:
            points_path = fr"/scratch/tmp/m_ster15/points_{cell_size}.npy"
        WKA_data_path = r"/home/m/m_ster15/WindEnergy/Algorithms/base_information_enercon_reformatted.json"
elif USER == 'Josefina':
    if RUN_LOCAL:
        points_path = fr"/Users/josefinabalzer/Desktop/WS22_23/Study_Project/Study_Project_Wind_Energy/data/points_{cell_size}.npy"
        WKA_data_path = r"/Users/josefinabalzer/Desktop/WS22_23/Study_Project/Study_Project_Wind_Energy/base_information_enercon_reformatted.json"
    else:
        points_path = fr"/scratch/tmp/jbalzer/Study_Project/data/points_{cell_size}.npy"
        WKA_data_path = r"/home/j/jbalzer/Study_Project_Wind_Energy/base_information_enercon_reformatted.json"

with open(points_path, "rb") as f:
    points = np.load(f, allow_pickle=True)
with open(WKA_data_path, "r") as f:
    WKA_data = json.load(f)

WKAs = {}
for wka in WKA_data["turbines"]:
    WKAs[wka["type"].replace(" ", "_")] = wka


class WindEnergySiteSelectionProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        # super().__init__(n_var=gdf_optimization.shape[0], n_obj=2, n_ieq_constr=0, xl=0.0, xu=1.0)
        if repair_mode:
            super().__init__(n_var=points.shape[0], n_obj=2, n_ieq_constr=0, xl=0.0,
                             xu=1.0, **kwargs)  # Bearbeitet weil v_var nicht mehr gepasst hat
        else:
            super().__init__(n_var=points.shape[0], n_obj=2, n_ieq_constr=1, xl=0.0,
                             xu=1.0, **kwargs)  # Bearbeitet weil v_var nicht mehr gepasst hat

    def _evaluate(self, x, out, *args, **kwargs):
        indices = np.where(x)[0]
        combs = combinations(indices, 2)
        constraints_np = -1

        def repair(x1, x2):
            x[x1] = False

        for combination in combs:
            if combination[0] and combination[1]:
                WKA1 = points[combination[0]]
                WKA2 = points[combination[1]]
                WKA1_type = WKAs[WKA1[0]]
                WKA2_type = WKAs[WKA2[0]]
                d = WKA1[1].distance(WKA2[1])
                if 3 * WKA1_type["rotor_diameter_in_meter"] < d and 3 * WKA2_type["rotor_diameter_in_meter"] < d:
                    if repair_mode:
                        repair(combination[0], combination[1])
                    else:
                        constraints_np = 1
                        break

        vals = np.where(x, points[:, 0], "")

        uniques, count = np.unique(vals, return_counts=True)

        type_prices = {}
        for idx, item in enumerate(uniques):
            if item == "":
                continue
            building_price = WKAs[item]["price"]["price_building"]
            price_per_year = WKAs[item]["price"]["price_per_year"]
            sum_price_one = building_price + (price_per_year * WKAs[item]["life_expectancy_in_years"])
            type_prices[item] = sum_price_one

        for key, value in type_prices.items():
            vals[vals == key] = value
        vals[vals == ""] = 0
        vals_sum = np.sum(vals)

        # Grundlage für Energieberechnung https://www.energie-lexikon.info/megawattstunde.html
        vals_ = np.where(x, points[:, 0], "")
        uniques, count = np.unique(vals_, return_counts=True)

        type_energy = {}
        for idx, item in enumerate(uniques):
            if item == "":
                continue
            nominal_power = WKAs[item]["nominal_power_in_kW"]
            lifetime_hours = WKAs[item][
                                 "life_expectancy_in_years"] * 8760  # Laut google ist 1 Jahr 8760 stunden # trifft nicht auf schaltjahre zu. Dort sind es 8784
            kwh = nominal_power * lifetime_hours
            type_energy[item] = kwh

        for key, value in type_energy.items():
            vals_[vals_ == key] = value
        vals_[vals_ == ""] = 0
        vals__sum = np.sum(vals_)

        out["F"] = np.column_stack([vals_sum, vals__sum])
        if not repair_mode:
            out["G"] = np.asarray([constraints_np])


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.off = {}

    def notify(self, algorithm):
        logging.info("Callbackk Call")
        self.off[algorithm.n_gen] = algorithm


def main():
    if USER == "Emily":
        if RUN_LOCAL:
            logging.basicConfig(filename="../Results/old/Reduced Area Repair Excluded/WindEnergy.log",
                                level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(filename="/home/m/m_ster15/WindEnergy/WindEnergy.log",
                                level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    if USER == "Josefina":
        # Todo: Pfade anpassen
        if RUN_LOCAL:
            logging.basicConfig(filename="../Results/old/Reduced Area Repair Excluded/WindEnergy.log",
                                level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(filename="/home/m/m_ster15/WindEnergy/WindEnergy.log",
                                level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    sys.stderr.write = logging.error
    sys.stdout.write = logging.info

    logging.info("Daten geladen und bereit")
    logging.info(f"{points.shape[0]} Punkte werden Prozessiert")
    # Todo: Population Size und Iterationsanzahl passend wählen
    algorithm = NSGA2(pop_size=100,
                      sampling=BinaryRandomSampling(),
                      crossover=TwoPointCrossover(),
                      mutation=BitflipMutation(),
                      eliminate_duplicates=True)

    n_proccess = 9
    pool = multiprocessing.Pool(n_proccess)
    runner = StarmapParallelization(pool.starmap)

    problem = WindEnergySiteSelectionProblem(elementwise_runner=runner)
    # problem = WindEnergySiteSelectionProblem()
    callback = MyCallback()
    logging.info("Starte Minimierung")
    termination = get_termination("time", timeString)
    res = minimize(problem,
                   algorithm,
                   callback=callback,
                   termination=termination,
                   seed=1,
                   verbose=True)

    logging.info("Minimierung Abgeschlossen")

    if USER == 'Emily':
        if RUN_LOCAL:
            with open("../Results/ga_reduced/result.pkl", "wb") as out:
                pickle.dump(res, out, pickle.HIGHEST_PROTOCOL)

            with open("callback.pkl", "wb") as out:
                pickle.dump(callback, out, pickle.HIGHEST_PROTOCOL)
        else:
            with open("/scratch/tmp/m_ster15/result.pkl", "wb") as out:
                pickle.dump(res, out, pickle.HIGHEST_PROTOCOL)

            with open("/scratch/tmp/m_ster15/callback.pkl", "wb") as out:
                pickle.dump(callback, out, pickle.HIGHEST_PROTOCOL)
    elif USER == 'Josefina':
        # Todo: Pfade anpassen
        if RUN_LOCAL:
            with open("/data/result.pkl", "wb") as out:
                pickle.dump(res, out, pickle.HIGHEST_PROTOCOL)

            with open("/data/callback.pkl", "wb") as out:
                pickle.dump(callback, out, pickle.HIGHEST_PROTOCOL)
        else:
            with open("/scratch/tmp/jbalzer/result.pkl", "wb") as out:
                pickle.dump(res, out, pickle.HIGHEST_PROTOCOL)

            with open("/scratch/tmp/jbalzer/callback.pkl", "wb") as out:
                pickle.dump(callback, out, pickle.HIGHEST_PROTOCOL)
    logging.info("Speichern Abgeschlossen")

    # Pymoo scatter
    if RUN_LOCAL:
        Scatter().add(res.F).show()
    logging.info("Programm Terminiert..")


if __name__ == "__main__":
    main()
