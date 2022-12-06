## Parameter input Files line 150

import numpy as np
import math
import os
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.petal import Petal
import matplotlib.pyplot as plt
from pymoo.util import plotting
from abc import abstractmethod
from pymoo.factory import get_algorithm
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import shutil
import utilities

PROJECT_DIR = os.getcwd()


def out_to_ndarray(out):
    for key, val in out.items():
        if val is not None:
            if not isinstance(val, np.ndarray):
                out[key] = np.array([val])


def check(problem, X, out):
    elementwise = X.ndim == 1
    n_evals = X.shape[0]
    F, df, G, dG = out.get("F"), out.get("dF"), out.get("G"), out.get("dG")


def elementwise_eval(problem, x, out, args, kwargs):
    problem._evaluate(x, out, *args, **kwargs)
    out_to_ndarray(out)
    check(problem, x, out)
    return out


def looped_eval(func_elementwise_eval, problem, X, out, *args, **kwargs):
    return [func_elementwise_eval(problem, x, dict(out), args, kwargs) for x in X]


def at_least_2d_array(x, extend_as="row", return_if_reshaped=False):
    if x is None:
        return x
    elif not isinstance(x, np.ndarray):
        x = np.array([x])

    has_been_reshaped = False

    if x.ndim == 1:
        if extend_as == "row":
            x = x[None, :]
        elif extend_as == "column":
            x = x[:, None]

        has_been_reshaped = True
    if return_if_reshaped:
        return x, has_been_reshaped
    else:
        return x


class ElementwiseProblem(Problem):
    def __init__(self,
                 func_elementwise_eval=elementwise_eval,
                 func_eval=looped_eval,
                 exclude_from_serialization=None,
                 runner=None,
                 **kwargs):
        super().__init__(exclude_from_serialization=exclude_from_serialization, **kwargs)
        self.func_elementwise_eval = func_elementwise_eval
        self.func_eval = func_eval
        self.runner = runner
        self.exclude_from_serialization = self.exclude_from_serialization + ["runner"]

    def do(self, X, out, *args, **kwargs):
        ret = self.func_eval(self.func_elementwise_eval, self, X, out, *args, **kwargs)
        keys = list(ret[0].keys())
        for key in keys:
            assert all([key in _out for _out in ret]), f"For some elements the {key} value has not been set."
            vals = []
            for elem in ret:
                val = elem[key]
                if val is not None:
                    if isinstance(val, list) or isinstance(val, tuple):
                        val = np.array(val)
                    elif not isinstance(val, np.ndarray):
                        val = np.full(1, val)
                    val = at_least_2d_array(val, extend_as="row")
                vals.append(val)
            if all([val is None for val in vals]):
                out[key] = None
            else:
                out[key] = np.row_stack(vals)
        return out

    @abstractmethod
    def _evaluate(self, x, out, *args, **kwargs):
        pass


def func_return_none(*args, **kwargs):
    return None


class FunctionalProblem(ElementwiseProblem):
    def __init__(self, n_var, objs, constr_ieq=[], constr_eq=[], constr_eq_eps=1e-5, func_pf=func_return_none,
                 func_ps=func_return_none, **kwargs):
        if callable(objs):
            objs = [objs]
        self.objs = objs
        self.constr_ieq = constr_ieq
        self.constr_eq = constr_eq
        self.constr_eq_eps = constr_eq_eps
        self.func_pf = func_pf
        self.func_ps = func_ps
        n_constr = len(constr_ieq) + len(constr_eq)

        super().__init__(n_var=n_var, n_obj=len(self.objs), n_constr=n_constr, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        ieq = np.array([constr(x) for constr in self.constr_ieq])
        ieq[ieq < 0] = 0
        eq = np.array([constr(x) for constr in self.constr_eq])
        eq = np.abs(eq)
        eq = eq - self.constr_eq_eps
        f = np.array([obj(x) for obj in self.objs])
        out["F"] = f
        out["G"] = np.concatenate([ieq, eq])

    def _calc_pareto_front(self, *args, **kwargs):
        return self.func_pf(*args, **kwargs)

    def _calc_pareto_set(self, *args, **kwargs):
        return self.func_ps(*args, **kwargs)


def JacketBounds(d_init):
    lbx1 = 4
    ubx1 = 15
    lbx2 = np.ceil(math.pi * (d_init + 0.2) / 0.30)
    ubx2 = np.ceil(math.pi * (d_init + 0.2) / 0.05)
    lbx3 = 3
    ubx3 = 10
    lb = [lbx1, lbx2, lbx3]
    ub = [ubx1, ubx2, ubx3]
    return lb, ub


def JacketObj():
    intVars = [1, 2, 3]
    nvars = 3
    goal = 1.1
    goal2 = 1.05
    # fc of core concrete
    fc_init = 24
    # fc of jacket concrete
    fc_jack = 38
    # fy of core steel
    fy_core = 440
    # fy of jacket steel
    fy_jack = 550
    # diameter of circular column
    d_init = 1.6
    # diameter of longitudinal reinforcement of jacket
    dl_jack = 0.022
    # diameter of transverse reinforcement of jacket
    ds_jack = 0.016
    lb, ub = JacketBounds(d_init)
    rhol_init = 0.008
    rhow_init = 0.0075
    Alunit_jack = math.pi * (dl_jack ** 2) / 4
    Asunit_jack = math.pi * (ds_jack ** 2) / 4
    cost = [1500, 1100, 1100]
    co2 = [0.12, 0.684, 5]

    beta0 = [0.566, 1.167, 0.802, 1.094, -4.213]
    beta1 = [0.059, -0.353, -0.212, -0.274, 4.610]
    beta2 = [0.129, 0.012, -0.013, 0.003, 0.296]
    beta3 = [0.016, 0.007, 0.219, 0.124, -0.044]
    beta4 = [-0.169, 0.002, -0.324, -0.334, 0.097]
    beta5 = [+0.399, +0.164, +0.528, +0.388, 0.253]

    obj = [
        lambda x: (cost[0] * 3.14159 * (2 * 0.02 * x[0] * d_init + (0.02 * x[0]) ** 2) / 4 + cost[1] * (
                Alunit_jack * x[1]) * 7.850 + cost[2] * (Asunit_jack / (0.025 * x[2])) * 3.14159 * (
                           d_init + 0.02 * x[0]) * 7.850) / 1000,

        lambda x: -(beta0[1] + beta4[1] * fc_jack / fc_init + beta5[1] * fy_jack / fy_core + beta1[1] * (
                d_init + 0.02 * 2 * x[0]) / d_init + beta2[0] * (
                            Alunit_jack * x[0] / (d_init + 0.02 * 2 * x[0]) ** 2 / rhol_init) + beta3[1] * (
                            4 * Asunit_jack / (0.025 * x[2] * (d_init - 0.05 + 0.02 * x[0]))) / rhow_init) /
                  (beta0[0] + beta4[0] * fc_jack / fc_init + beta5[0] * fy_jack / fy_core + beta1[0] * (
                          d_init + 0.02 * 2 * x[0]) / d_init + beta2[0] * (
                           Alunit_jack * x[1] / (d_init + 0.02 * 2 * x[0]) ** 2 / rhol_init) + beta3[0] * (
                           4 * Asunit_jack / (0.025 * x[2] * (d_init - 0.05 + 0.02 * x[0]))) / rhow_init),

        lambda x: -(beta0[2] + beta4[2] * fc_jack / fc_init + beta5[2] * fy_jack / fy_core + beta1[2] * (
                d_init + 0.02 * 2 * x[0]) / d_init + beta2[2] * (
                            Alunit_jack * x[1] / (d_init + 0.02 * 2 * x[0]) ** 2 / rhol_init) + beta3[2] * (
                            4 * Asunit_jack / (0.025 * x[2] * (d_init - 0.05 + 0.02 * x[0]))) / rhow_init) /
                  (beta0[0] + beta4[0] * fc_jack / fc_init + beta5[0] * fy_jack / fy_core + beta1[0] * (
                          d_init + 0.02 * 2 * x[0]) / d_init + beta2[0] * (
                           Alunit_jack * x[1] / (d_init + 0.02 * 2 * x[0]) ** 2 / rhol_init) + beta3[0] * (
                           4 * Asunit_jack / (0.025 * x[2] * (d_init - 0.05 + 0.02 * x[0]))) / rhow_init),

        #lambda x: -(beta0[3] + beta4[3] * fc_jack / fc_init + beta5[3] * fy_jack / fy_core + beta1[3] * (
        #        d_init + 0.02 * 2 * x[0]) / d_init + beta2[3] * (Alunit_jack * x[1] / (1.1025 * d_init ** 2)) / (
        #                rhol_init) + beta3[3] * (
        #                    4 * Asunit_jack / (0.025 * x[2] * (d_init - 0.05 + 0.02 * x[0]))) / rhow_init) / (
        #                  beta0[0] + beta4[0] * fc_jack / fc_init + beta5[0] * fy_jack / fy_core + beta1[0] * (
        #                  d_init + 0.02 * 2 * x[0]) / d_init + beta2[0] * (
        #                          Alunit_jack * x[1] / (1.1025 * d_init ** 2)) / (rhol_init) + beta3[0] * (
        #                          4 * Asunit_jack / (
        #                          0.025 * x[2] * (d_init - 0.05 + 0.02 * x[0]))) / rhow_init),
        lambda x: (co2[0] * 3.14159 / 4 * ((d_init + 2 * 0.02 * x[0]) * (d_init - 2 * 0.02 * x[0])) * 2500 + co2[
            1] * Alunit_jack * x[1] * 7850) / 1000]
    constraint = lambda x: -(beta0[4] + beta4[4] * fc_jack / fc_init + beta5[4] * fy_jack / fy_core + beta1[4] * (
            d_init + 0.02 * 2 * x[0]) / d_init + beta2[4] * (Alunit_jack * x[1] / (1.1025 * d_init ** 2)) / rhol_init +
                             beta3[4] * (4 * Asunit_jack / (
                    0.025 * x[2] * (d_init - 0.05 + 0.02 * x[0]))) / rhow_init) + goal2

    if callable(constraint):
        constraint = [constraint]
    problem = FunctionalProblem(3, obj, constr_ieq=constraint, xl=lb, xu=ub)

    method = get_algorithm('nsga2', pop_size=1000)
    res = minimize(problem, method, save_history=False)
    bwm = best_value(res.X, res.F)
    m = min(bwm)
    print(m)
    plot = Scatter()
    plot.add(res.F)
    plot.show()
    # Check if PetalPics folder exists
    petalpics_dir = os.path.join(PROJECT_DIR, 'PetalPics')
    if os.path.exists(petalpics_dir):
        shutil.rmtree(petalpics_dir)
        os.mkdir(petalpics_dir)
    else:
        os.mkdir(petalpics_dir)
    for i in range(len(res.F)):
        plot_petal = Petal(bounds=[-2, 2])
        plot_petal.add(res.F[i])
        plot_petal.save(f'PetalPics/petal_population_{i}.png')

    utilities.create_population_video()


def best_value(x, obj):
    w = [0.45, 0.045, 0.23, 0.17, 0.1]
    bwm = np.zeros(len(x))
    for i in range(len(bwm)):
        bwm[i] = sum(x * y for x, y in zip(w, obj[i]))
    return bwm


if __name__ == '__main__':
    JacketObj()