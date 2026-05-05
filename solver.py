import os
import tempfile

import pyomo.environ as pyo
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.opt import TerminationCondition


def build_model(data):
    m = pyo.ConcreteModel()
    m.FOOD = pyo.Set(initialize=data["foods"])
    m.NUTRIENTS = pyo.Set(initialize=data["nutrients"])
    m.needs = pyo.Param(m.NUTRIENTS, initialize=data["needs"])
    m.content = pyo.Param(m.FOOD, m.NUTRIENTS, initialize=data["content"])
    m.price = pyo.Param(m.FOOD, initialize=data["price"])
    m.eaten = pyo.Var(m.FOOD, domain=pyo.NonNegativeReals)

    def need_def(m, n):
        return sum(m.content[f, n] * m.eaten[f] for f in m.FOOD) >= m.needs[n]

    m.need_constraint = pyo.Constraint(m.NUTRIENTS, rule=need_def)
    m.cost = pyo.Objective(
        expr=sum(m.eaten[f] * m.price[f] for f in m.FOOD),
        sense=pyo.minimize,
    )
    return m


def _solve_capturing(m):
    """Run the solver and return (results, log_text). Captures GLPK's
    subprocess stdout via two mechanisms (FD-level redirect + logfile=)
    so we get output reliably across platforms."""
    fd, log_path = tempfile.mkstemp(suffix=".glpk.log")
    os.close(fd)
    log_text = ""
    try:
        try:
            # capture_fd=True intercepts at the OS file-descriptor level,
            # which catches subprocess output (the GLPK binary) on Linux
            # where it would otherwise bypass Python's sys.stdout.
            with capture_output(capture_fd=True) as buf:
                solver = pyo.SolverFactory("glpk")
                results = solver.solve(m, tee=True, logfile=log_path)
            log_text = buf.getvalue()
        except TypeError:
            # Older Pyomo versions don't accept capture_fd=. Fall back to
            # the default (sys.stdout redirect only) plus the logfile.
            with capture_output() as buf:
                solver = pyo.SolverFactory("glpk")
                results = solver.solve(m, tee=True, logfile=log_path)
            log_text = buf.getvalue()

        # If the FD/stdout capture didn't yield anything, fall back to the
        # logfile that Pyomo wrote to.
        if not log_text.strip():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    log_text = f.read()
            except OSError:
                pass
    finally:
        try:
            os.remove(log_path)
        except OSError:
            pass

    return results, log_text


def solve(data):
    if not data["foods"]:
        return {"status": "no_foods", "x": {}, "cost": None, "log": ""}

    m = build_model(data)

    try:
        results, log = _solve_capturing(m)
    except ApplicationError as e:
        return {
            "status": "solver_missing",
            "message": (
                "GLPK solver binary not found. On Streamlit Cloud add "
                "`glpk-utils` to packages.txt at the repo root. "
                f"({e})"
            ),
            "x": {},
            "cost": None,
            "log": "",
        }

    tc = results.solver.termination_condition
    if tc == TerminationCondition.optimal:
        x = {f: float(pyo.value(m.eaten[f])) for f in data["foods"]}
        cost = float(pyo.value(m.cost))
        return {"status": "optimal", "x": x, "cost": cost, "log": log}
    if tc in (
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
    ):
        return {"status": "infeasible", "x": {}, "cost": None, "log": log}
    if tc == TerminationCondition.unbounded:
        return {"status": "unbounded", "x": {}, "cost": None, "log": log}
    return {"status": str(tc), "x": {}, "cost": None, "log": log}
