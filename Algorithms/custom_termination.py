from pymoo.core.termination import Termination
from pymoo.termination.ftol import SingleObjectiveSpaceTermination
from pymoo.termination.max_time import TimeBasedTermination
from pymoo.termination.robust import RobustTermination


class DefaultTermination(Termination):

    def __init__(self, f, t) -> None:
        super().__init__()
        self.f = f
        self.t = t
        self.criteria = [self.f, self.t]

    def _update(self, algorithm):
        p = [criterion.update(algorithm) for criterion in self.criteria]
        print(p)
        return max(p)


class customSingleObjectiveTermination(DefaultTermination):

    def __init__(self, max_time, ftol=1e-6, period=30, **kwargs) -> None:
        print(type(max_time))
        f = RobustTermination(SingleObjectiveSpaceTermination(ftol, only_feas=True), period=period)
        t = TimeBasedTermination(max_time)
        print(t)
        super().__init__(f, t)
