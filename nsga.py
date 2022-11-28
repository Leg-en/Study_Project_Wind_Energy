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

Wind_Vektor = (0, 0)

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

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        # Ich hab die methode einfach kopiert und hab keine ahnung ob die so richtig ist
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

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
            coor_1_wind = (coor_1[0] + Wind_Vektor[0], coor_1[1] + Wind_Vektor[1])
            coor_2_wind = (coor_2[0] + Wind_Vektor[0], coor_2[1] + Wind_Vektor[1])
            angle = self.angle_between(coor_1_wind, coor_2_wind)
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
