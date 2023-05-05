from pymoo.core.termination import Termination
from pymoo.termination.cv import ConstraintViolationTermination
from pymoo.termination.ftol import SingleObjectiveSpaceTermination, MultiObjectiveSpaceTermination
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.robust import RobustTermination
from pymoo.termination.xtol import DesignSpaceTermination
from pymoo.termination import get_termination

class DefaultTermination(Termination):

    def __init__(self, f, t) -> None:
        super().__init__()
        self.f = f
        self.t = t
        self.criteria = [self.f, self.t]

    def _update(self, algorithm):
        p = [criterion.update(algorithm) for criterion in self.criteria]
        return max(p)
class customSingleObjectiveTermination(DefaultTermination):

    def __init__(self,max_time:int, ftol=1e-6, period=30, **kwargs) -> None:
        f = RobustTermination(SingleObjectiveSpaceTermination(ftol, only_feas=True), period=period)
        t = get_termination("time", max_time)
        super().__init__(f,t)
