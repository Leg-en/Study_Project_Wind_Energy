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
import matplotlib.pyplot as plt
import random
import dill
from pymoo.termination.max_gen import MaximumGenerationTermination

# from pymoo.termination import get_termination

RUN_NAME = "Test"
reduced = True  # Das sind im Worst Case immer noch 40765935 Mögliche Kombinationen mit dem verkleinerten gebiet..
RUN_LOCAL = True
POOL_SIZE = 8
SMART_REPAIR = True
max_base_generations = 100
max_add_generations = 2 #Ist jeweils 100 generationen zusätzlich
strompreis = 0.10
cell_size = 100


# Pfade müssen angepasst werden
USER = 'Emily'

if USER == "Emily":
    if RUN_LOCAL:
        base_data_path = r"C:\workspace\Study_Project_Wind_Energy\Algorithms\source_data"
        base_save_path = r"C:\workspace\Study_Project_Wind_Energy\Results"
    if not RUN_LOCAL:
        base_data_path = r"/home/m/m_ster15/WindEnergy/source_data"
        base_save_path = r"/home/m/m_ster15/WindEnergy/saves"
if USER == "Josefina":
    if RUN_LOCAL:
        base_data_path = r""
        base_save_path = r""
    if not RUN_LOCAL:
        base_data_path = r""
        base_save_path = r""



WKA_data_path = os.path.join(base_data_path, "base_information_enercon_reformatted.json")
if reduced:
    points_path = os.path.join(base_data_path, f"points_{cell_size}_reduced.npy")
else:
    points_path = os.path.join(base_data_path, f"points_{cell_size}.npy")

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
            WKA1 = points[combination[0]]
            WKA2 = points[combination[1]]
            WKA1_type = WKAs[WKA1[0]]
            WKA2_type = WKAs[WKA2[0]]
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
            WKA1 = points[combination[0]]
            WKA2 = points[combination[1]]
            WKA1_type = WKAs[WKA1[0]]
            WKA2_type = WKAs[WKA2[0]]
            d = WKA1[1].distance(WKA2[1])
            if 3 * WKA1_type["rotor_diameter_in_meter"] > d or 3 * WKA2_type["rotor_diameter_in_meter"] > d:
                row[combination[random.choice([0,1])]] = False
        return (idx, row)

    def repair_mp(self, item):
        row = item[0]
        idx = item[1]
        indices = np.where(row)[0]
        combs = combinations(indices, 2)
        for combination in combs:
            WKA1 = points[combination[0]]
            WKA2 = points[combination[1]]
            WKA1_type = WKAs[WKA1[0]]
            WKA2_type = WKAs[WKA2[0]]
            d = WKA1[1].distance(WKA2[1])
            # Alt und Falsch: if 3 * WKA1_type["rotor_diameter_in_meter"] < d and 3 * WKA2_type["rotor_diameter_in_meter"] < d:
            if 3 * WKA1_type["rotor_diameter_in_meter"] > d or 3 * WKA2_type["rotor_diameter_in_meter"] > d:
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
        super().__init__(n_var=points.shape[0], n_obj=1, n_ieq_constr=1, xl=0.0,
                         xu=1.0, **kwargs)  # Bearbeitet weil v_var nicht mehr gepasst hat

    def const_check(self, item):
        row = item[0]
        idx = item[1]
        indices = np.where(row)[0]
        combs = combinations(indices, 2)
        constraints_np = -1
        for combination in combs:
            WKA1 = points[combination[0]]
            WKA2 = points[combination[1]]
            WKA1_type = WKAs[WKA1[0]]
            WKA2_type = WKAs[WKA2[0]]
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
        vals__sum = vals__sum * strompreis
        result = vals__sum - vals_sum
        result = -result
        out["F"] = result
        out["G"] = np.asarray([constraints_np])


def main():
    global pool
    save_path = os.path.join(base_save_path, f"saves_{RUN_NAME}")
    os.mkdir(save_path)
    logging.basicConfig(filename=os.path.join(save_path, RUN_NAME+".log"),
                        level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # sys.stderr.write = logging.error
    sys.stdout.write = logging.info

    logging.info("Daten geladen und bereit")
    logging.info(f"{points.shape[0]} Punkte werden Prozessiert")

    # Todo: Population Size und Iterationsanzahl passend wählen
    try:
        pool = Pool(POOL_SIZE)
        algorithm = GA(pop_size=100,
                       sampling=BinaryRandomSampling(),
                       crossover=TwoPointCrossover(),
                       # Evtl uniformcrossover probieren from pymoo.operators.crossover.ux import UniformCrossover
                       mutation=BitflipMutation(),
                       eliminate_duplicates=True,
                       repair=CustomRepair())

        problem = WindEnergySiteSelectionProblem()
        logging.info("Starte Minimierung")
        # termination = get_termination("time", timeString)
        # termination = get_termination("n_gen", 100)
        max_base_generations = 100
        res = minimize(problem,
                       algorithm,
                       ('n_gen', max_base_generations),
                       seed=1,
                       verbose=True,
                       save_history=True)
        logging.info("First Minimization done")
        save_state = 0
        for gen in range(max_add_generations):
            logging.info(f"{gen} Iteration of minimization done")
            max_base_generations += 100
            algorithm.termination = MaximumGenerationTermination(max_base_generations)
            res = minimize(problem,
                           algorithm,
                           seed=1,
                           verbose=True,
                           save_history=True)
            with open(os.path.join(save_path, f"res{save_state%2}.dill"), "wb") as file:
                dill.dump(algorithm, file)
            save_state += 1


        logging.info("Minimierung Abgeschlossen")

        with open(os.path.join(save_path, "res.dill"), "wb") as file:
            dill.dump(res, file)

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


def plot(res):
    fitness_vals = []

    for iteration in res.history:
        x = []
        for item in iteration.pop:
            x.append(item.F[0])
        fitness_vals.append(x)
    np_fitness = np.asarray(fitness_vals)

    # Manual Scotter
    plot_val = [1, 10, 30, 50]
    for i in plot_val:
        plt.scatter(np_fitness[i], np.zeros(len(np_fitness[i])))
    # plt.scatter(res.F[:, 0], res.F[:, 1])
    plt.show()
