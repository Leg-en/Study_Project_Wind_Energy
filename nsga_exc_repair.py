import json
import logging
import pickle
import sys
from itertools import combinations
from multiprocessing import Pool

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# from pymoo.termination import get_termination


reduced = True  # Das sind im Worst Case immer noch 40765935 Mögliche Kombinationen mit dem verkleinerten gebiet..
RUN_LOCAL = False
POOL_SIZE = 10
SMART_REPAIR = True
PROFIT_FIVE_YEARS = 7.79 # Angabe in ct pro kW/h für die ersten 5 Jahre nach Installation
PROFIT_LATER_YEARS = 7.79 # Angabe in ct pro kW/h nach den ersten 5 Jahren nach Installation


cell_size = 100 # maybe we could change this to 50
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

    # Checkt welche elemente die meisten kollisionen verursachen und deaktiviert diese
    def smart_repair(self, item):
        row = item[0]
        idx = item[1]
        indices = np.where(row)[0]
        combs = combinations(indices, 2)
        collisions = {}
        for combination in combs:
            if combination[0] and combination[1]:
                WKA1 = points[combination[0]]
                WKA2 = points[combination[1]]
                WKA1_type = WKAs[WKA1[0]]
                WKA2_type = WKAs[WKA2[0]]
                d = WKA1[1].distance(WKA2[1])
                if 3 * WKA1_type["rotor_diameter_in_meter"] < d and 3 * WKA2_type["rotor_diameter_in_meter"] < d:
                    if combination[0] in collisions:
                        collisions[combination[0]].append(combination[1])
                    else:
                        collisions[combination[0]] = [combination[1]]
                    if combination[1] in collisions:
                        collisions[combination[1]].append(combination[0])
                    else:
                        collisions[combination[1]] = [combination[0]]
        colls_sorted = dict(sorted(collisions.items(), key=lambda elem: len(elem)))
        collisions = None
        del collisions
        for key in colls_sorted.copy().keys():
            if key in colls_sorted:
                row[key] = False
                colls_sorted.pop(key, None)
                for subkey in colls_sorted.copy().keys():
                    if subkey in colls_sorted:
                        if key in colls_sorted[subkey]:
                            colls_sorted[subkey].remove(key)
                            if len(colls_sorted[subkey]) == 0:
                                colls_sorted.pop(subkey, None)
        return (idx, row)

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
                    row[combination[
                        0]] = False  # Todo: Sinnvollen ersatz Finden, einfach durch random choice ersetzen verschlechtert das ergebnis einfach
        return (idx, row)

    def _do(self, problem, X, **kwargs):
        row_gen = (x for x in X)
        indices = np.arange(X.shape[0])

        if SMART_REPAIR:
            res = pool.map(self.smart_repair, zip(row_gen, indices))
        else:
            res = pool.map(self.repair_mp, zip(row_gen, indices))
        # Evtl überflüssig
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


        #Ab hier werden die kosten berechnet
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

        #Ab hier wird die energie produktion berechnet
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
            logging.basicConfig(filename="/home/m/m_ster15/WindEnergy/WindEnergy2.log",
                                level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    if USER == "Josefina":
        # Todo: Pfade anpassen
        if RUN_LOCAL:
            logging.basicConfig(filename="WindEnergy.log",
                                level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(filename="/home/m/m_ster15/WindEnergy/WindEnergy.log",
                                level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    # sys.stderr.write = logging.error
    sys.stdout.write = logging.info

    logging.info("Daten geladen und bereit")
    logging.info(f"{points.shape[0]} Punkte werden Prozessiert")

    # Todo: Population Size und Iterationsanzahl passend wählen
    try:
        pool = Pool(POOL_SIZE)
        algorithm = NSGA2(pop_size=100,
                          sampling=BinaryRandomSampling(),
                          crossover=TwoPointCrossover(), #Evtl uniformcrossover probieren from pymoo.operators.crossover.ux import UniformCrossover
                          mutation=BitflipMutation(),
                          eliminate_duplicates=True,
                          repair=CustomRepair())

        problem = WindEnergySiteSelectionProblem()
        logging.info("Starte Minimierung")
        # termination = get_termination("time", timeString)
        # termination = get_termination("n_gen", 100)
        res = minimize(problem,
                       algorithm,
                       ('n_gen', 100),
                       seed=1,
                       verbose=True,
                       save_history=True)

        logging.info("Minimierung Abgeschlossen")

        if USER == 'Emily':
            if RUN_LOCAL:
                with open("result2.pkl", "wb") as out:
                    pickle.dump(res, out, pickle.HIGHEST_PROTOCOL)
            else:
                with open("/home/m/m_ster15/WindEnergy/result2.pkl", "wb") as out:
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
        logging.error("Unbekannte Exception")
        logging.error(exc)
        raise exc


if __name__ == "__main__":
    main()
