import json
import pickle
from itertools import combinations
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.repair import Repair
from multiprocessing import Pool
import logging
import sys
from pymoo.core.problem import Problem
#from pymoo.termination import get_termination





reduced = False  # Das sind im Worst Case immer noch 40765935 Mögliche Kombinationen mit dem verkleinerten gebiet..
RUN_LOCAL = False
POOL_SIZE = 10


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


with open(points_path, "rb") as f:
    points = np.load(f, allow_pickle=True)
with open(WKA_data_path, "r") as f:
    WKA_data = json.load(f)

WKAs = {}
for wka in WKA_data["turbines"]:
    WKAs[wka["type"].replace(" ", "_")] = wka


# print("Daten geladen und bereit")


# https://pymoo.org/constraints/repair.html
class CustomRepair(Repair):

    def repair_mp(self, item):
        row = item[0]
        idx = item[1]
        indices = np.where(row)[0]
        combs = combinations(indices, 2)
        for combination in combs:
            if combination[0] and combination[1]:
                WKA1 = points[combination[0]]
                WKA2 = points[combination[1]]
                WKA1_type = WKAs[WKA1[0]]
                WKA2_type = WKAs[WKA2[0]]
                d = WKA1[1].distance(WKA2[1])
                if 3 * WKA1_type["rotor_diameter_in_meter"] < d and 3 * WKA2_type["rotor_diameter_in_meter"] < d:
                    row[combination[0]] = False #Todo: Sinnvollen ersatz Finden, einfach durch random choice ersetzen verschlechtert das ergebnis einfach
        return (idx, row)

    def _do(self, problem, X, **kwargs):
        row_gen = (x for x in X)
        indices = np.arange(X.shape[0])

        res = pool.map(self.repair_mp, zip(row_gen, indices))
        res.sort(key=lambda elem: elem[0])
        for idx, item in res:
            X[idx, :] = item
        return X



class WindEnergySiteSelectionProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=points.shape[0], n_obj=2, n_ieq_constr=1, xl=0.0,
                         xu=1.0, **kwargs)  # Bearbeitet weil v_var nicht mehr gepasst hat


    def const_check(self, item):
        row = item[0]
        idx = item[1]
        indices = np.where(row)[0]
        combs = combinations(indices, 2)
        constraints_np = -1
        for combination in combs:
            if combination[0] and combination[1]:
                WKA1 = points[combination[0]]
                WKA2 = points[combination[1]]
                WKA1_type = WKAs[WKA1[0]]
                WKA2_type = WKAs[WKA2[0]]
                d = WKA1[1].distance(WKA2[1])
                if 3 * WKA1_type["rotor_diameter_in_meter"] < d and 3 * WKA2_type["rotor_diameter_in_meter"] < d:
                    constraints_np = 1
                    break
        return (idx, constraints_np)

    def _evaluate(self, X, out, *args, **kwargs):

        constraints_np = np.zeros((X.shape[0]))
        row_gen = (x for x in X)
        indices = np.arange(X.shape[0])

        res = pool.map(self.const_check, zip(row_gen, indices))
        res.sort(key=lambda elem: elem[0])
        for idx, item in res:
            constraints_np[idx] = item




        vals = np.where(X, points[:, 0], "")

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
        vals_sum = np.sum(vals, axis=1)


        # Grundlage für Energieberechnung https://www.energie-lexikon.info/megawattstunde.html
        vals_ = np.where(X, points[:, 0], "")
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
        vals__sum = np.sum(vals_, axis=1)
        vals__sum = -vals__sum

        out["F"] = np.column_stack([vals_sum, vals__sum])
        out["G"] = np.asarray([constraints_np])



def main():
    global pool

    if USER == "Emily":
        if RUN_LOCAL:
            logging.basicConfig(filename="WindEnergy.log",
                                level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(filename="/home/m/m_ster15/WindEnergy/WindEnergy.log",
                                level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    if USER == "Josefina":
        # Todo: Pfade anpassen
        if RUN_LOCAL:
            logging.basicConfig(filename="WindEnergy.log",
                                level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(filename="/home/m/m_ster15/WindEnergy/WindEnergy.log",
                                level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    #sys.stderr.write = logging.error
    sys.stdout.write = logging.info

    logging.info("Daten geladen und bereit")
    logging.info(f"{points.shape[0]} Punkte werden Prozessiert")

    # Todo: Population Size und Iterationsanzahl passend wählen
    try:
        pool = Pool(POOL_SIZE)
        algorithm = NSGA2(pop_size=100,
                          sampling=BinaryRandomSampling(),
                          crossover=TwoPointCrossover(),
                          mutation=BitflipMutation(),
                          eliminate_duplicates=True,
                          repair=CustomRepair())


        problem = WindEnergySiteSelectionProblem()
        logging.info("Starte Minimierung")
        #termination = get_termination("time", timeString)
        #termination = get_termination("n_gen", 100)
        res = minimize(problem,
                       algorithm,
                       termination=('n_gen', 100),
                       seed=1,
                       verbose=True,
                       save_history=True)

        logging.info("Minimierung Abgeschlossen")

        if USER == 'Emily':
            if RUN_LOCAL:
                with open("result2.pkl", "wb") as out:
                    pickle.dump(res, out, pickle.HIGHEST_PROTOCOL)
            else:
                with open("/home/m/m_ster15/WindEnergy/result.pkl", "wb") as out:
                    pickle.dump(res, out, pickle.HIGHEST_PROTOCOL)

        elif USER == 'Josefina':
            # Todo: Pfade anpassen
            if RUN_LOCAL:
                with open("result.pkl", "wb") as out:
                    pickle.dump(res, out, pickle.HIGHEST_PROTOCOL)

            else:
                with open("/result.pkl", "wb") as out:
                    pickle.dump(res, out, pickle.HIGHEST_PROTOCOL)


        logging.info("Speichern Abgeschlossen")

        # Pymoo scatter
        if RUN_LOCAL:
            Scatter().add(res.F).show()
        pool.close()
        logging.info("Programm Terminiert..")
    except Exception as exc:
        pool.close()
        logging.info("Unbekannte Exception")
        logging.error(exc.with_traceback())
        raise exc




if __name__ == "__main__":
    main()
