import json
import logging
import os
import pickle
import sys
from itertools import combinations
from multiprocessing import Pool
from pymoo.algorithms.soo.nonconvex.ga import GA
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import random
import dill
from pymoo.termination import get_termination
from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.default import DefaultSingleObjectiveTermination
import argparse
from pymoo.core.termination import Termination
from custom_termination import customSingleObjectiveTermination

parser = argparse.ArgumentParser()

parser.add_argument("config_path", help="Pfad zur Config Datei", type=str)
args = parser.parse_args()

device = "pc"

RUN_NAME = None
REDUCED = None
RUN_LOCAL = None
POOL_SIZE = None
REPAIR = None
STROMPREIS = None
CELL_SIZE = None
USER = None
termination = None
base_data_path = None
base_save_path = None
points_path = None
points = None
WKA_data = None
WKAs = None


# print("Daten geladen und bereit")


# https://pymoo.org/constraints/repair.html
class CustomRepair(Repair):

    def __init__(self, points, WKAs, REPAIR):
        super().__init__()
        self.points = points
        self.WKAs = WKAs
        self.REPAIR = REPAIR

    # Checkt welche elemente die meisten kollisionen verursachen und deaktiviert diese
    def smart_repair(self, item):
        row = item[0]
        idx = item[1]
        indices = np.where(row)[0]
        combs = combinations(indices, 2)
        collisions = {}
        for combination in combs:
            WKA1 = self.points[combination[0]]
            WKA2 = self.points[combination[1]]
            WKA1_type = self.WKAs[WKA1[0]]
            WKA2_type = self.WKAs[WKA2[0]]
            d = WKA1[1].distance(WKA2[1])
            if 3 * WKA1_type["rotor_diameter_in_meter"] > d:
                if combination[0] in collisions:
                    collisions[combination[0]].append(combination[1])
                else:
                    collisions[combination[0]] = [combination[1]]
            if 3 * WKA2_type["rotor_diameter_in_meter"] > d:
                if combination[1] in collisions:
                    collisions[combination[1]].append(combination[0])
                else:
                    collisions[combination[1]] = [combination[0]]
        colls_sorted = sorted(collisions.items(), key=lambda elem: len(elem[1]), reverse=True)
        colls_sorted_as_np = np.asarray([val[0] for val in colls_sorted])
        if not colls_sorted_as_np.shape == (0,):
            for key in colls_sorted_as_np:
                if key in collisions:
                    row[key] = False
                    collisions.pop(key, None)
                    subarray = sorted(collisions.items(), key=lambda elem: len(elem[1]), reverse=True)
                    subarray_np = np.asarray([val[0] for val in subarray])
                    if not subarray_np.shape == (0,):
                        for subkey in subarray_np:
                            if subkey in collisions:
                                if key in collisions[subkey]:
                                    collisions[subkey].remove(key)
                                    if len(collisions[subkey]) == 0:
                                        collisions.pop(subkey, None)
        return (idx, row)

    def random_repair(self, item):
        '''
        Identical to repair_mp besides the randomized choice of the the element that should be removed.
        :param item:
        :return:
        '''
        row = item[0]
        idx = item[1]
        indices = np.where(row)[0]
        combs = combinations(indices, 2)
        for combination in combs:
            WKA1 = self.points[combination[0]]
            WKA2 = self.points[combination[1]]
            WKA1_type = self.WKAs[WKA1[0]]
            WKA2_type = self.WKAs[WKA2[0]]
            d = WKA1[1].distance(WKA2[1])
            if 3 * WKA1_type["rotor_diameter_in_meter"] > d or 3 * WKA2_type["rotor_diameter_in_meter"] > d:
                row[combination[random.choice([0, 1])]] = False
        return (idx, row)

    def repair_mp(self, item):
        row = item[0]
        idx = item[1]
        indices = np.where(row)[0]
        combs = combinations(indices, 2)
        for combination in combs:
            WKA1 = self.points[combination[0]]
            WKA2 = self.points[combination[1]]
            WKA1_type = self.WKAs[WKA1[0]]
            WKA2_type = self.WKAs[WKA2[0]]
            d = WKA1[1].distance(WKA2[1])
            # Alt und Falsch: if 3 * WKA1_type["rotor_diameter_in_meter"] < d and 3 * WKA2_type["rotor_diameter_in_meter"] < d:
            if 3 * WKA1_type["rotor_diameter_in_meter"] > d or 3 * WKA2_type["rotor_diameter_in_meter"] > d:
                row[combination[
                    0]] = False
        return (idx, row)

    def _do(self, problem, X, **kwargs):
        row_gen = (x for x in X)
        indices = np.arange(X.shape[0])

        if self.REPAIR == "smart_repair":
            res = pool.map(self.smart_repair, zip(row_gen, indices))
        elif self.REPAIR == "simple":
            res = pool.map(self.repair_mp, zip(row_gen, indices))
        elif self.REPAIR == "random":
            res = pool.map(self.random_repair, zip(row_gen, indices))

        # Evtl überflüssig
        res.sort(key=lambda elem: elem[0])
        for idx, item in res:
            X[idx, :] = item
        return X


class WindEnergySiteSelectionProblem(Problem):

    def __init__(self, points, WKAs, REPAIR, **kwargs):
        super().__init__(n_var=points.shape[0], n_obj=1, n_ieq_constr=1, xl=0.0,
                         xu=1.0, **kwargs)  # Bearbeitet weil v_var nicht mehr gepasst hat
        self.points = points
        self.WKAs = WKAs
        self.REPAIR = REPAIR

    def const_check(self, item):
        row = item[0]
        idx = item[1]
        indices = np.where(row)[0]
        combs = combinations(indices, 2)
        constraints_np = -1
        for combination in combs:
            WKA1 = self.points[combination[0]]
            WKA2 = self.points[combination[1]]
            WKA1_type = self.WKAs[WKA1[0]]
            WKA2_type = self.WKAs[WKA2[0]]
            d = WKA1[1].distance(WKA2[1])
            if 3 * WKA1_type["rotor_diameter_in_meter"] > d or 3 * WKA2_type["rotor_diameter_in_meter"] > d:
                constraints_np = 1
                break
        return (idx, constraints_np)

    def _evaluate(self, X, out, *args, **kwargs):

        constraints_np = np.empty((X.shape[0]))
        row_gen = (x for x in X)
        indices = np.arange(X.shape[0])

        res = pool.map(self.const_check, zip(row_gen, indices))
        res.sort(key=lambda elem: elem[0])
        for idx, item in res:
            constraints_np[idx] = item

        # Ab hier werden die kosten berechnet
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

        # Ab hier wird die energie produktion berechnet
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
        vals__sum = vals__sum * STROMPREIS
        result = vals__sum - vals_sum
        result = -result
        out["F"] = result
        out["G"] = np.asarray([constraints_np])


def main(points, WKAs, REPAIR):
    global pool
    save_path = os.path.join(base_save_path, f"saves_{RUN_NAME}")
    os.mkdir(save_path)
    logging.basicConfig(filename=os.path.join(save_path, RUN_NAME + ".log"),
                        level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Arguments given: {args}")

    # sys.stderr.write = logging.error
    sys.stdout.write = logging.info

    logging.info("Daten geladen und bereit")
    logging.info(f"{points.shape[0]} Punkte werden Prozessiert")

    try:
        pool = Pool(POOL_SIZE)
        if REPAIR.lower() != "none":
            algorithm = GA(pop_size=100,
                           sampling=BinaryRandomSampling(),
                           crossover=TwoPointCrossover(),
                           # Evtl uniformcrossover probieren from pymoo.operators.crossover.ux import UniformCrossover
                           mutation=BitflipMutation(),
                           eliminate_duplicates=True,
                           repair=CustomRepair(points, WKAs, REPAIR))
        else:
            algorithm = GA(pop_size=100,
                           sampling=BinaryRandomSampling(),
                           crossover=TwoPointCrossover(),
                           # Evtl uniformcrossover probieren from pymoo.operators.crossover.ux import UniformCrossover
                           mutation=BitflipMutation(),
                           eliminate_duplicates=True)

        problem = WindEnergySiteSelectionProblem(points, WKAs, REPAIR)
        logging.info("Starte Minimierung")
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       verbose=True,
                       save_history=True)

        logging.info("Minimierung Abgeschlossen")

        with open(os.path.join(save_path, "res.dill"), "wb") as file:
            dill.dump(res, file)

        logging.info("Speichern Abgeschlossen")

        pool.close()
        logging.info("Programm Terminiert..")
    except Exception as exc:
        pool.close()
        logging.error("Unbekannte Exception")
        logging.error(exc)
        raise exc
    logging.shutdown()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


def meta_main():
    global RUN_NAME, REDUCED, RUN_LOCAL, POOL_SIZE, REPAIR, STROMPREIS, CELL_SIZE, USER, termination, base_data_path, base_save_path, points_path, points, WKA_data, WKAs
    with open(args.config_path, "r") as file:
        config = json.load(file)
    for run in config["runs"]:
        RUN_NAME = run["RUN_NAME"]
        REDUCED = run["REDUCED"]
        RUN_LOCAL = config["global_flags"]["RUN_LOCAL"]
        POOL_SIZE = config["global_flags"]["pool_size"]
        REPAIR = run["REPAIR"]
        STROMPREIS = run["STROMPREIS"]
        CELL_SIZE = run["CELL_SIZE"]
        USER = config["global_flags"]["USER"]
        if run["termination"]["time"] and run["termination"]["robust_crit"]:
            termination = customSingleObjectiveTermination(max_time=run["termination"]["time"],
                                                           ftol=run["termination"]["robust_crit"], period=100)
        elif run["termination"]["time"]:
            termination = get_termination("time", run["termination"]["time"])
        elif run["termination"]["robust_crit"]:
            termination = RobustTermination(
                MultiObjectiveSpaceTermination(tol=run["termination"]["robust_crit"]), period=100)
        elif run["termination"]["max_iter"]:
            termination = get_termination("n_gen", run["termination"]["max_iter"])
        else:
            raise ValueError("No Termination Criterion given")

        if USER == "Emily":
            if RUN_LOCAL:
                if device == "pc":
                    base_data_path = r"C:\Users\Emily\OneDrive - Universität Münster\Uni\WiSe22 23\StudyProject\processed_data_collection"
                    base_save_path = r"C:\Users\Emily\OneDrive - Universität Münster\Uni\SoSe23\Paper\results"
                elif device == "mac":
                    base_data_path = r"/Users/emily/Library/CloudStorage/OneDrive-UniversitätMünster/Uni/WiSe22 23/StudyProject/processed_data_collection"
                    base_save_path = r"/Users/emily/Desktop/Workspace/Study_Project_Wind_Energy/results"
            if not RUN_LOCAL:
                base_data_path = r"/home/m/m_ster15/WindEnergy/source_data"
                base_save_path = r"/home/m/m_ster15/WindEnergy/saves"
        if USER == "Josefina":
            if RUN_LOCAL:
                base_data_path = r""
                base_save_path = r""
            if not RUN_LOCAL:
                base_data_path = r"/home/j/jbalzer/Study_Project_Wind_Energy/Algorithms/source_data"
                base_save_path = r"/home/j/jbalzer/Study_Project_Wind_Energy/saves"
        WKA_data_path = os.path.join(base_data_path, "base_information_enercon_reformatted.json")

        if REDUCED == "reduced":
            points_path = os.path.join(base_data_path, f"points_{CELL_SIZE}_reduced.npy")
        elif REDUCED == "full":
            points_path = os.path.join(base_data_path, f"points_{CELL_SIZE}.npy")
        elif REDUCED == "single":
            points_path = os.path.join(base_data_path, f"points_{CELL_SIZE}_single.npy")

        with open(points_path, "rb") as f:
            points = np.load(f, allow_pickle=True)
        with open(WKA_data_path, "r") as f:
            WKA_data = json.load(f)

        WKAs = {}
        for wka in WKA_data["turbines"]:
            WKAs[wka["type"].replace(" ", "_")] = wka
        main(points, WKAs, REPAIR)


if __name__ == "__main__":
    meta_main()
