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

Wind_deg = 270

points_path = r"C:\workspace\MasterSemester1\WindEnergy\Project\data\numpy_arr\15cell_np.npy"
WKA_data_path = r"C:\workspace\MasterSemester1\WindEnergy\Project\input_WKAs.json"

with open(points_path, "rb") as f:
    points = np.load(f, allow_pickle=True)
with open(WKA_data_path, "r") as f:
    WKA_data = json.load(f)

dir = r'input'
rfile = 'potentialareas_400m_forest.shp'


# Die distanzmatrix enthält jetzt alle relevanten distanz informationen

class WindEnergySiteSelectionProblem(ElementwiseProblem):
    import numpy as np

    def angle_between(self, p1, p2):
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))

    def __init__(self, **kwargs):
        # super().__init__(n_var=gdf_optimization.shape[0], n_obj=2, n_ieq_constr=0, xl=0.0, xu=1.0)
        super().__init__(n_var=(points.shape[0] * len(WKA_data["turbines"])), n_obj=2, n_ieq_constr=1, xl=0.0,
                         xu=1.0, **kwargs)  # Bearbeitet weil v_var nicht mehr gepasst hat

    def _evaluate(self, x, out, *args, **kwargs):
        indices = np.where(x)[0]
        combs = combinations(indices, 2)
        for combination in combs:
            WKA_Type1 = int(combination[0] / len(points))
            WKA_Type2 = combination[0] / len(points)
            d = points[combination[0] % len(points)].distance(points[combination[1] % len(points)])

            # Todo: Überprüfen ob die Winkelberechnung auch nur etwas sinn macht.
            coor_1 = (points[combination[0] % len(points)].x, points[combination[0] % len(points)].y)
            coor_2 = (points[combination[1] % len(points)].x, points[combination[1] % len(points)].y)
            angle = self.angle_between(coor_2, coor_2)
            angle_corr = (angle + Wind_deg) % 360
            print()


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.off = {}

    def notify(self, algorithm):
        self.off[algorithm.n_gen] = algorithm.off


def main():
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
