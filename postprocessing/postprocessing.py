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
        for key in colls_sorted.keys():
            if key in colls_sorted:
                row[key] = False
                colls_sorted.pop(key, None)
                for subkey in colls_sorted.keys():
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
    with open(r"PATH", "rb") as file:
        res = pickle.load(file)
    res.X
