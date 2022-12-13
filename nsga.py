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
from pymoo.visualization.scatter import Scatter
import multiprocessing
from pymoo.core.problem import StarmapParallelization

Wind_deg = 270

points_path = r"C:\workspace\Study_Project_Wind_Energy\data\processed_data_50cell_size\numpy_array\points_50.npy"
WKA_data_path = r"C:\workspace\Study_Project_Wind_Energy\base_information_enercon_reformatted.json"

with open(points_path, "rb") as f:
    points = np.load(f, allow_pickle=True)
with open(WKA_data_path, "r") as f:
    WKA_data = json.load(f)

WKAs = {}
for wka in WKA_data["turbines"]:
    WKAs[wka["type"].replace(" ", "_")] = wka

print("Daten geladen und bereit")


class WindEnergySiteSelectionProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        # super().__init__(n_var=gdf_optimization.shape[0], n_obj=2, n_ieq_constr=0, xl=0.0, xu=1.0)
        super().__init__(n_var=points.shape[0], n_obj=1, n_ieq_constr=1, xl=0.0,
                         xu=1.0, **kwargs)  # Bearbeitet weil v_var nicht mehr gepasst hat

    def _evaluate(self, x, out, *args, **kwargs):
        indices = np.where(x)[0]
        combs = combinations(indices, 2)
        constraints_np = -1
        for combination in combs:
            WKA1 = points[combination[0]]
            WKA2 = points[combination[1]]
            WKA1_type = WKAs[WKA1[0]]
            WKA2_type = WKAs[WKA2[0]]
            d = WKA1[1].distance(WKA2[1])
            if 3 * WKA1_type["rotor_diameter_in_meter"] < d and 3 * WKA2_type["rotor_diameter_in_meter"] < d:
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

        # Grundlage für Energieberechnung https://www.energie-lexikon.info/megawattstunde.html
        vals_ = np.where(x, points[:, 0], "")
        uniques, count = np.unique(vals_, return_counts=True)

        type_energy = {}
        for idx, item in enumerate(uniques):
            if item == "":
                continue
            nominal_power = WKAs[item]["nominal_power_in_kW"]
            lifetime_hours = WKAs[item]["life_expectancy_in_years"] * 8760  # Laut google ist 1 Jahr 8760 stundn
            kwh = nominal_power * lifetime_hours
            type_prices[item] = kwh

        for key, value in type_energy.items():
            vals_[vals_ == key] = value
        vals_[vals_ == ""] = 0

        out["F"] = np.column_stack([np.sum(vals), np.sum(vals_)])
        out["G"] = np.asarray([constraints_np])


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.off = {}

    def notify(self, algorithm):
        self.off[algorithm.n_gen] = algorithm


def main():
    #Todo: Population Size und Iterationsanzahl passend wählen
    algorithm = NSGA2(pop_size=100,
                      sampling=BinaryRandomSampling(),
                      crossover=TwoPointCrossover(),
                      mutation=BitflipMutation(),
                      eliminate_duplicates=True)

    n_proccess = 10
    # pool = multiprocessing.Pool(n_proccess)
    # runner = StarmapParallelization(pool.starmap)

    # problem = WindEnergySiteSelectionProblem(elementwise_runner=runner)
    problem = WindEnergySiteSelectionProblem()
    callback = MyCallback()
    res = minimize(problem,
                   algorithm,
                   callback=callback,
                   termination=('n_gen', 100),
                   seed=1,
                   verbose=True)

    with open("result.pkl", "wb") as out:
        pickle.dump(res, out, pickle.HIGHEST_PROTOCOL)

    with open("callback.pkl", "wb") as out:
        pickle.dump(callback, out, pickle.HIGHEST_PROTOCOL)

    # Pymoo scatter
    Scatter().add(res.F).show()


if __name__ == "__main__":
    main()
