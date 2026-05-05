import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from pyomo.common.errors import ApplicationError


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


def solve(data):
    if not data["foods"]:
        return {"status": "no_foods", "x": {}, "cost": None}

    m = build_model(data)
    try:
        solver = pyo.SolverFactory("glpk")
        results = solver.solve(m, tee=False)
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
        }

    tc = results.solver.termination_condition
    if tc == TerminationCondition.optimal:
        x = {f: float(pyo.value(m.eaten[f])) for f in data["foods"]}
        cost = float(pyo.value(m.cost))
        return {"status": "optimal", "x": x, "cost": cost}
    if tc in (
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
    ):
        return {"status": "infeasible", "x": {}, "cost": None}
    if tc == TerminationCondition.unbounded:
        return {"status": "unbounded", "x": {}, "cost": None}
    return {"status": str(tc), "x": {}, "cost": None}
