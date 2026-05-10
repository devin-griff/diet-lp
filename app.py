# =============================================================================
# Diet LP Optimizer — a Streamlit tutorial app.
#
# This file builds an interactive web app around the classic Stigler-style
# diet problem: choose how much of each food to buy so that total cost is
# minimized while every nutrient requirement is met.
#
# It is a Linear Program (LP) — variables are continuous, not binary:
#   minimize   sum_f  p_f * x_f                  (cost)
#   subject to sum_f  D_{f,n} * x_f >= r_n  for all nutrients n
#              x_f >= 0
#
# Library roadmap:
#   - streamlit  — the UI framework. Each interaction reruns this script
#                  top-to-bottom; persistent values live in `st.session_state`.
#   - pyomo      — algebraic modeling: sets, params, vars, objective,
#                  constraints. Continuous (NonNegativeReals) variables here.
#   - HiGHS      — the LP solver, called via Pyomo's appsi_highs interface.
#                  Ships as a pip wheel (`highspy`).
#   - pandas     — DataFrame shape for Streamlit's data editor and Altair.
#   - altair     — grouped bars + horizontal "min requirement" markers.
#
# File roadmap:
#   1. Solver       — model definition + HiGHS log capture.
#   2. Constants    — defaults, nutrient labels, slider cap.
#   3. State        — session_state init / reset / slider helpers.
#   4. Utilities    — DataFrame <-> internal-dict conversion, totals, cost.
#   5. LaTeX        — render the current instance as a formatted equation.
#   6. Tabs         — render_data / render_formulation / render_logs / render_optimizer.
#   7. Main         — page config and tab assembly at module bottom.
# =============================================================================

import base64
import copy
import math
from pathlib import Path

import json

import altair as alt
import pandas as pd
import pyomo.environ as pyo
import streamlit as st
import streamlit.components.v1 as components
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.opt import TerminationCondition
from streamlit_vertical_slider import vertical_slider


# ---------- Solver ----------
#
# Standard Pyomo LP. The only twist is `_solve_capturing`, which redirects
# HiGHS's solver output at the OS file-descriptor level so we can show it
# in the Logs tab.

def build_model(data):
    # ConcreteModel: components bound to data at construction time.
    m = pyo.ConcreteModel()

    # Two index sets: foods (F) and nutrients (N).
    m.FOOD = pyo.Set(initialize=data["foods"])
    m.NUTRIENTS = pyo.Set(initialize=data["nutrients"])

    # Parameters:
    #   needs[n]     = required minimum amount r_n of nutrient n
    #   content[f,n] = how much of nutrient n is in one unit of food f (D_{f,n})
    #   price[f]     = unit cost p_f of food f
    m.needs = pyo.Param(m.NUTRIENTS, initialize=data["needs"])
    m.content = pyo.Param(m.FOOD, m.NUTRIENTS, initialize=data["content"])
    m.price = pyo.Param(m.FOOD, initialize=data["price"])

    # Decision variable x_f: how much of each food to buy. Continuous and
    # non-negative — fractional servings are allowed.
    m.eaten = pyo.Var(m.FOOD, domain=pyo.NonNegativeReals)

    # One nutrient constraint per nutrient n: total nutrient delivered must
    # be at least the required amount. `rule=` builds an indexed constraint.
    def need_def(m, n):
        return sum(m.content[f, n] * m.eaten[f] for f in m.FOOD) >= m.needs[n]

    m.need_constraint = pyo.Constraint(m.NUTRIENTS, rule=need_def)

    # Objective: minimize total cost.
    m.cost = pyo.Objective(
        expr=sum(m.eaten[f] * m.price[f] for f in m.FOOD),
        sense=pyo.minimize,
    )
    return m


def _solve_capturing(m):
    """Run the solver and return (results, log_text). Captures HiGHS's
    stdout via Pyomo's capture_output (FD-level redirect on newer Pyomo,
    plain stdout capture on older). HiGHS via appsi_highs doesn't support
    a logfile= kwarg, so the FD-level capture is the only path."""
    log_text = ""
    try:
        with capture_output(capture_fd=True) as buf:
            solver = pyo.SolverFactory("appsi_highs")
            results = solver.solve(m, tee=True)
        log_text = buf.getvalue()
    except TypeError:
        # Older Pyomo without capture_fd — fall back to plain stdout capture.
        with capture_output() as buf:
            solver = pyo.SolverFactory("appsi_highs")
            results = solver.solve(m, tee=True)
        log_text = buf.getvalue()
    return results, log_text


def solve(data):
    # Top-level entrypoint used by the UI. Always returns a plain dict so the
    # caller can stash the result in session_state without holding on to a
    # live Pyomo model.

    # Empty problem — bail before constructing a model with no foods.
    if not data["foods"]:
        return {"status": "no_foods", "x": {}, "cost": None, "log": ""}

    m = build_model(data)

    try:
        results, log = _solve_capturing(m)
    except ApplicationError as e:
        # Pyomo raises ApplicationError when the solver isn't available.
        # HiGHS ships as a pip wheel via highspy, so this normally only
        # fires on a broken install.
        return {
            "status": "solver_missing",
            "message": (
                "HiGHS solver not available. Run `pip install highspy` "
                f"in your environment. ({e})"
            ),
            "x": {},
            "cost": None,
            "log": "",
        }

    # Translate Pyomo's TerminationCondition enum into a small set of stable
    # status strings the UI knows how to render.
    tc = results.solver.termination_condition
    if tc == TerminationCondition.optimal:
        # Pull numeric values out of the model before returning.
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
    # Catch-all (solver error, time limit, etc.).
    return {"status": str(tc), "x": {}, "cost": None, "log": log}


# ---------- Constants ----------
#
# The four nutrients are baked into the schema (column ordering, slider
# cap math, etc.). NUTRIENT_LABELS maps the short codes to display names.

NUTRIENTS = ["P", "C", "F", "V"]
NUTRIENT_LABELS = {"P": "Protein", "C": "Carbs", "F": "Fat", "V": "Vitamins"}

# Hard caps used by the UI. MAX_FOODS truncates the data editor; SLIDER_CAP
# is the absolute upper bound for the per-food sliders even if the model
# could theoretically use more.
MAX_FOODS = 10
SLIDER_CAP = 50.0

# Default instance shown on first load and after the "Reset to defaults"
# button. `content` is keyed by (food, nutrient) tuples.
DEFAULT_DATA = {
    "foods": ["fruit", "vegetables", "meat", "bread", "pasta", "eggs"],
    "nutrients": NUTRIENTS,
    "needs": {"P": 20.0, "C": 40.0, "F": 15.0, "V": 20.0},
    "price": {
        "fruit": 2.0, "vegetables": 2.0, "meat": 6.0,
        "bread": 2.0, "pasta": 3.0, "eggs": 4.0,
    },
    "content": {
        ("fruit", "P"): 1.0, ("fruit", "C"): 4.0, ("fruit", "F"): 0.0, ("fruit", "V"): 5.0,
        ("vegetables", "P"): 2.0, ("vegetables", "C"): 3.0, ("vegetables", "F"): 0.0, ("vegetables", "V"): 6.0,
        ("meat", "P"): 8.0, ("meat", "C"): 0.0, ("meat", "F"): 5.0, ("meat", "V"): 1.0,
        ("bread", "P"): 2.0, ("bread", "C"): 6.0, ("bread", "F"): 1.0, ("bread", "V"): 1.0,
        ("pasta", "P"): 3.0, ("pasta", "C"): 8.0, ("pasta", "F"): 1.0, ("pasta", "V"): 0.0,
        ("eggs", "P"): 6.0, ("eggs", "C"): 1.0, ("eggs", "F"): 4.0, ("eggs", "V"): 2.0,
    },
}


# ---------- State ----------
#
# Streamlit re-executes the whole script on every interaction. Anything that
# must persist between runs lives in `st.session_state`. The keys we use:
#   - data:                the current problem instance (foods/needs/...)
#   - optimal:             the most recent solver result, or None
#   - _pending_reset:      one-shot flag to reset on the next run
#   - slider_<food>:       slider value for each food (the user's diet)
#   - need_<nutrient>:     number_input value for each nutrient requirement
#   - data_editor:         backing key for the data editor widget

def slider_key(food):
    # Stable, food-keyed widget identifier.
    return f"slider_{food}"


def init_state():
    # Idempotent init: only seed defaults the first time, otherwise the
    # user's edits would be wiped on every rerun.
    if "data" not in st.session_state:
        st.session_state.data = copy.deepcopy(DEFAULT_DATA)
    if "optimal" not in st.session_state:
        st.session_state.optimal = None
    # New foods added later need a slider key, but existing keys are
    # preserved so the user's selection survives.
    for f in st.session_state.data["foods"]:
        if slider_key(f) not in st.session_state:
            st.session_state[slider_key(f)] = 0.0
    # The reset button can't directly mutate widget-backed keys without
    # raising a Streamlit error, so it sets a flag and reruns. We then
    # apply the reset *before* widgets are instantiated this run.
    if st.session_state.pop("_pending_reset", False):
        apply_reset()


def apply_reset():
    # Restore the default instance and clear all widget-backed keys.
    st.session_state.data = copy.deepcopy(DEFAULT_DATA)
    st.session_state.optimal = None
    for f in DEFAULT_DATA["foods"]:
        st.session_state[slider_key(f)] = 0.0
    for n in NUTRIENTS:
        st.session_state[f"need_{n}"] = float(DEFAULT_DATA["needs"][n])


def current_slider_values():
    # Read each slider out of session_state. Missing keys default to 0.
    return {
        f: float(st.session_state.get(slider_key(f), 0.0))
        for f in st.session_state.data["foods"]
    }


# ---------- Utilities ----------
#
# Adapters between two data shapes:
#   - Internal dict shape (used by solver and most of the app).
#   - DataFrame shape (used by Streamlit's data editor widget).
# Plus small helpers for slider bounds, nutrient sums, and cost.

def data_to_df(data):
    # Internal -> DataFrame. One row per food, one column per nutrient + price.
    rows = []
    for f in data["foods"]:
        rows.append({
            "Food": f,
            "P": data["content"][(f, "P")],
            "C": data["content"][(f, "C")],
            "F": data["content"][(f, "F")],
            "V": data["content"][(f, "V")],
            "Price": data["price"][f],
        })
    return pd.DataFrame(rows)


def df_to_data(df, needs):
    # DataFrame -> internal. Normalizes whatever the user typed: strip
    # whitespace, drop blanks/duplicates, coerce numerics, clamp to >= 0.
    df = df.copy()
    df["Food"] = df["Food"].astype("string").str.strip()
    df = df.dropna(subset=["Food"])
    df = df[df["Food"] != ""]
    df = df.drop_duplicates(subset=["Food"], keep="first")
    for col in NUTRIENTS + ["Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    foods = df["Food"].tolist()
    price = {row.Food: float(row.Price) for row in df.itertuples()}
    # `content` is keyed by (food, nutrient) tuples to match Pyomo's
    # multi-indexed parameter style.
    content = {}
    for row in df.itertuples():
        for n in NUTRIENTS:
            content[(row.Food, n)] = float(getattr(row, n))
    return {
        "foods": foods,
        "nutrients": NUTRIENTS,
        "needs": {n: float(needs[n]) for n in NUTRIENTS},
        "price": price,
        "content": content,
    }


def slider_upper_bound(food, data):
    # Pick a sensible max for a food's slider: the smallest amount that
    # would single-handedly satisfy any nutrient requirement (no point
    # going higher), capped by SLIDER_CAP and floored at 1.0 for visibility.
    bounds = []
    for n in NUTRIENTS:
        c = data["content"].get((food, n), 0.0)
        if c > 0:
            bounds.append(data["needs"][n] / c)
    if not bounds:
        return SLIDER_CAP
    return float(min(SLIDER_CAP, max(1.0, math.ceil(max(bounds)))))


def nutrient_totals(x, data):
    # For a given diet x, compute total amount of each nutrient delivered.
    return {
        n: sum(data["content"].get((f, n), 0.0) * float(x.get(f, 0.0)) for f in data["foods"])
        for n in NUTRIENTS
    }


def cost_of(x, data):
    # Total cost for a given diet x.
    return sum(data["price"][f] * float(x.get(f, 0.0)) for f in data["foods"])


# ---------- LaTeX (instance formulation) ----------
#
# The Formulation tab shows a static general formulation in math notation
# and a dynamic "instance" formulation that substitutes the user's current
# numbers into the equation. The helpers here build LaTeX source that
# `st.latex` then renders.

_LATEX_ESCAPE = [
    ("\\", r"\textbackslash "),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
]


def _latex_text(s):
    # Wrap a food name in \text{...} so it renders upright, with special
    # characters escaped.
    for raw, esc in _LATEX_ESCAPE:
        s = s.replace(raw, esc)
    return f"\\text{{{s}}}"


def _build_lhs(coefs, foods):
    # Build a "c1 * food1 + c2 * food2 + ..." style sum, skipping zero
    # coefficients and avoiding a leading "+".
    parts = []
    first = True
    for f in foods:
        c = float(coefs.get(f, 0.0))
        if c == 0:
            continue
        c_str = f"{c:g}"
        sep = "" if first else " + "
        parts.append(f"{sep}{c_str} \\, {_latex_text(f)}")
        first = False
    return "".join(parts) if parts else "0"


def build_instance_latex(data):
    # Assemble the full instance formulation as a LaTeX `aligned` block:
    # objective row, one constraint row per nutrient, then non-negativity.
    foods = data["foods"]
    obj = _build_lhs(data["price"], foods)
    rows = [r"\min \quad & " + obj + r" \\"]
    for i, n in enumerate(NUTRIENTS):
        coefs = {f: data["content"].get((f, n), 0.0) for f in foods}
        lhs = _build_lhs(coefs, foods)
        rhs = f"{data['needs'][n]:g}"
        label = NUTRIENT_LABELS[n]
        # First constraint row gets the "s.t." prefix; subsequent rows
        # just align under it.
        prefix = r"\text{s.t.} \quad & " if i == 0 else r"& "
        rows.append(f"{prefix}{lhs} \\ge {rhs} \\quad \\text{{({label})}} \\\\")
    bounds_lhs = ", ".join(_latex_text(f) for f in foods)
    rows.append(f"& {bounds_lhs} \\ge 0")
    body = r"\begin{aligned}" + "\n".join(rows) + r"\end{aligned}"
    # With many foods the line gets long; \small keeps it on screen.
    if len(foods) > 7:
        body = r"\small " + body
    return body


def colored_metric(label, value, color, align="left"):
    # st.metric doesn't support arbitrary value coloring, so we render a
    # metric-shaped block via raw HTML. Used to flag matching/mismatching
    # values (green if your cost equals the optimum, red otherwise).
    # `align` controls text-align so the metric can sit flush against
    # either edge of its column.
    style_color = f"color: {color};" if color else ""
    st.markdown(
        f"<div style='margin: 0.25rem 0 1rem 0; text-align: {align};'>"
        f"<div style='font-size: 0.875rem; color: rgba(49,51,63,0.6); margin-bottom: 0.25rem;'>{label}</div>"
        f"<div style='font-size: 2rem; font-weight: 600; line-height: 1; {style_color}'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------- Tabs ----------
#
# One render_* function per tab. Data lets the user edit foods and nutrient
# requirements; Formulation shows the math; Logs shows HiGHS output;
# Optimizer is the main interactive view (defined last in this file).

def render_data_tab():
    # Top: four side-by-side number inputs for nutrient minimums (r_n).
    # Each is bound to a `need_<n>` session_state key so `apply_reset` can
    # seed it.
    st.subheader("Nutrient requirements")
    cols = st.columns(4)
    needs = st.session_state.data["needs"]
    new_needs = {}
    for col, n in zip(cols, NUTRIENTS):
        new_needs[n] = col.number_input(
            f"{NUTRIENT_LABELS[n]} ({n})",
            min_value=0.0,
            value=float(needs[n]),
            step=1.0,
            key=f"need_{n}",
        )

    # Editable foods table. `num_rows="dynamic"` lets the user add/delete
    # rows. Each column has a configured min so users can't enter negatives.
    st.subheader(f"Foods (max {MAX_FOODS})")
    df = data_to_df(st.session_state.data)
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "Food": st.column_config.TextColumn("Food"),
            "P": st.column_config.NumberColumn("P (Protein)", min_value=0.0),
            "C": st.column_config.NumberColumn("C (Carbs)", min_value=0.0),
            "F": st.column_config.NumberColumn("F (Fat)", min_value=0.0),
            "V": st.column_config.NumberColumn("V (Vitamins)", min_value=0.0),
            "Price": st.column_config.NumberColumn("Price", min_value=0.0),
        },
        key="data_editor",
    )

    # Validate / report on the edited table.
    warnings = []
    if len(edited) > MAX_FOODS:
        warnings.append(f"Capped at {MAX_FOODS} foods; extra rows ignored.")
        edited = edited.head(MAX_FOODS)

    names = edited["Food"].dropna().astype("string").str.strip()
    if names.duplicated().any():
        warnings.append("Duplicate food names were dropped (kept the first).")

    new_data = df_to_data(edited, new_needs)

    # If the cleaned data differs from what we had, commit it to state and
    # rerun so other tabs see the change. Invalidate any prior solver result
    # and seed a slider for any newly-added food.
    if new_data != st.session_state.data:
        st.session_state.data = new_data
        st.session_state.optimal = None
        for f in new_data["foods"]:
            if slider_key(f) not in st.session_state:
                st.session_state[slider_key(f)] = 0.0
        st.rerun()

    for w in warnings:
        st.warning(w)

    # Reset uses the deferred-flag pattern documented in `init_state`.
    if st.button("Reset to defaults"):
        st.session_state["_pending_reset"] = True
        st.rerun()


def render_formulation_tab():
    # Two sub-tabs (matching pinch-analysis / strip-packing / facility-location):
    # General (static reference math + pedagogical content) and Instance
    # (substitutes the user's current foods into the formulation).
    sub_general, sub_instance = st.tabs(["General", "Instance"])

    with sub_general:
        # 3-column grid: Sets/Parameters/Variables stacked on the left, the
        # centered objective + constraints in the middle, empty right padding
        # so the equation lands at the page midline rather than the right half.
        left, right, _ = st.columns([1, 1, 1])
        with left:
            st.markdown(
                "**Sets**  \n"
                r"$\mathcal{F} = \{\text{foods}\}$" "  \n"
                r"$\mathcal{N} = \{\text{nutrients}\}$"
            )
            st.markdown(
                "**Parameters**  \n"
                r"$p_i$ price for food option $i \in \mathcal{F}$" "  \n"
                r"$r_j$ nutrition requirement for nutrient $j \in \mathcal{N}$" "  \n"
                r"$D_{ij}$ nutrition info for food $i \in \mathcal{F}$ and nutrient $j \in \mathcal{N}$"
            )
            st.markdown(
                "**Variables**  \n"
                r"$x_i$ amount of food $i \in \mathcal{F}$ eaten or purchased"
            )
        with right:
            # Title + display math in one centered block. `$$...$$` inside
            # st.markdown (rather than st.latex) lets us wrap both in a single
            # text-align:center div so the equation lines up under the title.
            st.markdown(
                r"""<div style="text-align: center;">

**Objective and Constraints**

$$
\begin{gathered}
\min_x \sum_{i \in \mathcal{F}} x_i p_i \quad \text{(cost)} \\
\text{s.t.} \quad \sum_{i \in \mathcal{F}} D_{ij} x_i \ge r_j \quad \forall j \in \mathcal{N} \quad \text{(nutrient minimums)} \\
x_i \ge 0 \quad \forall i \in \mathcal{F} \quad \text{(lower bounds)}
\end{gathered}
$$

</div>""",
                unsafe_allow_html=True,
            )

        st.markdown("**Stigler's original computation**")
        st.markdown(
            "In 1945, economist George Stigler asked: what is the cheapest "
            "diet that meets every daily nutritional requirement? With nothing "
            "but paper, pencil, and \"a shrewd hunch about the products that "
            "should appear in the optimal diet\" (his words), he found a five "
            "food combination (wheat flour, evaporated milk, cabbage, spinach, "
            "dried navy beans) costing \\$39.93 per year in 1939 dollars."
        )
        st.markdown(
            "Two years later, George Dantzig invented the simplex method and "
            "the diet problem became one of the first LPs ever solved by "
            "computer. The true LP optimum came out to \\$39.69 per year. "
            "Stigler's hand calculation was off by 0.6%."
        )
        st.markdown(
            "His hunch was almost optimal. He had no way to prove it though. "
            "An exact LP solver gives you certified optimality. A heuristic "
            "just gives you a number."
        )

        st.markdown("**Solution method**")
        st.markdown(
            "Solved as an LP via the simplex method, the algorithm Dantzig "
            "invented to solve this exact problem. HiGHS uses dual simplex "
            "by default, with an interior point alternative for very large "
            "instances. For a problem this size (a few foods times a few "
            "nutrients), the solve completes in microseconds. Most of what "
            "you see in the UI is Streamlit's render overhead."
        )
        st.markdown(
            "HiGHS is a modern open source LP/MIP solver from Edinburgh's "
            "ERGO group, distributed as a pip wheel via `highspy`."
        )

        st.markdown("**Companion notebook**")
        st.markdown(
            "See the [companion Jupyter notebook]"
            "(https://github.com/devin-griff/diet_lp/blob/main/diet.ipynb) "
            "for the Pyomo implementation."
        )

        st.markdown("**References**")
        st.markdown(
            "[1] G. J. Stigler, \"The Cost of Subsistence,\" "
            "*Journal of Farm Economics*, vol. 27, no. 2, pp. 303–314, 1945. "
            "[JSTOR](https://www.jstor.org/stable/1231810)"
        )
        st.markdown(
            "[2] G. B. Dantzig, \"The Diet Problem,\" "
            "*Interfaces*, vol. 20, no. 4, pp. 43–47, 1990. "
            "[INFORMS](https://pubsonline.informs.org/doi/abs/10.1287/inte.20.4.43)"
        )
        st.markdown(
            "[3] Q. Huangfu and J. A. J. Hall, \"Parallelizing the dual "
            "revised simplex method,\" *Mathematical Programming Computation*, "
            "vol. 10, no. 1, pp. 119–142, 2018. "
            "[Springer](https://link.springer.com/article/10.1007/s12532-017-0130-5)"
        )

    with sub_instance:
        st.markdown(
            "The current instance, with values configured on the **Data** tab:"
        )
        data = st.session_state.data
        if not data["foods"]:
            st.info("Add at least one food on the Data tab.")
            return
        st.latex(build_instance_latex(data))


def render_logs_tab():
    # Shows whatever HiGHS printed during the last solve. The capture itself
    # happens in `_solve_capturing`; this tab just displays the result.
    optimal = st.session_state.optimal
    if not optimal:
        st.info("Run the optimizer to see solver logs.")
        return
    log = optimal.get("log", "") or ""
    if not log.strip():
        st.info("No solver output captured for the last run.")
        return
    st.code(log, language="text")


def render_optimizer_tab():
    # Layout (top to bottom):
    #   1. Action buttons (Run Optimizer, Set at Optimum)
    #   2. Status banner if the last run was not optimal
    #   3. Side by side: "Your diet" (interactive vertical sliders) on the
    #      left, "Optimal diet" (read only vertical sliders) on the right.
    #      One vertical slider per food in each half.
    #   4. Nutrient bar chart + cost metrics below the slider banks.
    data = st.session_state.data
    if not data["foods"]:
        st.info("Add at least one food on the Data tab.")
        return

    # Action row. Narrow on the left so the buttons stay compact.
    act1, act2, _ = st.columns([1, 1, 6])
    with act1:
        if st.button("Run Optimizer", width="stretch", key="run_btn"):
            st.session_state.optimal = solve(data)
    optimal = st.session_state.optimal
    with act2:
        set_disabled = not (optimal and optimal["status"] == "optimal")
        if st.button(
            "Set at Optimum",
            width="stretch",
            disabled=set_disabled,
            key="set_opt_btn",
        ):
            # Copy the optimal x_f values into the user sliders. Each slider
            # is clamped to its own upper bound, then rounded to match the
            # step=0.1 of the user sliders (otherwise the value badge shows
            # full-precision floats like 1.62878787878). Bumping
            # `user_slider_version` rotates the component keys so the
            # vertical_slider iframes re-mount and actually pick up the new
            # default_value — the package ignores subsequent default_value
            # prop changes once mounted.
            for f in data["foods"]:
                ub = slider_upper_bound(f, data)
                val = float(optimal["x"].get(f, 0.0))
                clamped = max(0.0, min(val, ub))
                st.session_state[slider_key(f)] = round(clamped, 1)
            st.session_state.user_slider_version = (
                st.session_state.get("user_slider_version", 0) + 1
            )
            st.rerun()

    # Inline status messages for non optimal solver outcomes.
    if optimal:
        if optimal["status"] == "solver_missing":
            st.error(optimal["message"])
        elif optimal["status"] == "infeasible":
            st.error("Infeasible. No diet satisfies the requirements with this data.")
        elif optimal["status"] == "unbounded":
            st.error("Unbounded problem.")
        elif optimal["status"] not in ("optimal", "no_foods"):
            st.error(f"Solver returned: {optimal['status']}")

    # Clamp every slider key before any vertical_slider call so widgets
    # never receive an out of range value when the user shrinks a
    # requirement.
    for f in data["foods"]:
        ub = slider_upper_bound(f, data)
        key = slider_key(f)
        existing = float(st.session_state.get(key, 0.0))
        preserved = max(0.0, min(existing, ub))
        if existing != preserved:
            st.session_state[key] = preserved

    # Tighten the gaps between food sub-columns. Streamlit's smallest
    # `gap="small"` still leaves visible spacing in the horizontal block;
    # this scoped rule targets only stHorizontalBlocks that contain a
    # vertical_slider iframe so other column rows on the page (the action
    # button row, the cost metric row) are not affected. The optimal-diet
    # rule disables pointer events on the right column's iframes so those
    # sliders are visible but cannot be dragged.
    st.markdown(
        """
        <style>
        /* Tighten the gap between food sub-columns in any horizontal block
         * that holds an iframe (vertical sliders are the only iframes
         * inside columnar blocks on this page). */
        div[data-testid="stHorizontalBlock"]:has(iframe) {
            gap: 0 !important;
        }
        /* Disable text selection on the slider banks. Without this, dragging
         * a slider thumb (or even just clicking on the column) selects the
         * label text (food name, value badge, "0" tick label) and surrounding
         * UI, leaving an ugly blue highlight that persists until the user
         * clicks elsewhere. -webkit- prefix needed for older Safari. */
        div[data-testid="stHorizontalBlock"]:has(iframe),
        div[data-testid="stHorizontalBlock"]:has(iframe) * {
            -webkit-user-select: none;
            -moz-user-select: none;
            user-select: none;
        }
        /* Optimal-side iframes are read only. The data-readonly attribute
         * is stamped by the components.html script below onto each food
         * sub-column on the right side. */
        [data-readonly] iframe {
            pointer-events: none;
        }
        /* Custom hover tooltip on user-side food sub-columns. Native iframe
         * title tooltips fire unreliably because the inner React content
         * consumes the hover; CSS :hover on the parent column does fire
         * whenever the mouse is anywhere over the column's bounding box,
         * iframe included. The text comes from a data-tooltip attribute
         * the script below writes onto each user-side column. */
        [data-tooltip] {
            position: relative;
        }
        [data-tooltip]:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            left: 50%;
            top: 100%;
            transform: translateX(-50%);
            background: rgba(17, 24, 39, 0.92);
            color: #f9fafb;
            padding: 0.35rem 0.55rem;
            border-radius: 4px;
            font-size: 0.75rem;
            /* `pre-line` honors explicit \n line breaks in the data-tooltip
             * string while collapsing other whitespace, so the per-food
             * tooltip wraps onto two lines after Carb. `width: max-content`
             * stops the absolutely-positioned pseudo-element from inheriting
             * the food sub-column's narrow width and breaking on every
             * space. */
            white-space: pre-line;
            width: max-content;
            text-align: center;
            line-height: 1.35;
            z-index: 1000;
            pointer-events: none;
            margin-top: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Three column body: Your diet (left), nutrient chart (middle), Optimal
    # diet (right). [4,4,4] gives the chart more horizontal room and pulls
    # the six slider sub-columns tighter together on each side.
    your_col, chart_col, opt_col = st.columns([4, 4, 4])

    with your_col:
        st.markdown("**Your diet**")
        food_cols = st.columns(len(data["foods"]), gap="small")
        for c, f in zip(food_cols, data["foods"]):
            ub = slider_upper_bound(f, data)
            key = slider_key(f)
            current = float(st.session_state.get(key, 0.0))
            with c:
                # Version suffix forces re-mount when "Set at Optimum"
                # bumps the counter; otherwise the component's JS state
                # ignores new default_value props.
                version = st.session_state.get("user_slider_version", 0)
                component_key = f"v_{key}_{version}"
                vertical_slider(
                    label=f,
                    key=component_key,
                    height=220,
                    default_value=current,
                    min_value=0.0,
                    max_value=float(ub),
                    step=0.1,
                    slider_color="#FF4B4B",
                    track_color="#E5E9F1",
                    thumb_color="#FF4B4B",
                    value_always_visible=True,
                )
                # The package's return value is broken for drags to exactly
                # zero: its source does `return val if val else default`,
                # so a 0 from the JS frontend gets silently replaced with
                # the previous default_value. Read the raw component value
                # from session_state instead — Streamlit's component
                # machinery stores the actual JS-side value under the
                # component's key, bypassing the package's falsy check.
                raw = st.session_state.get(component_key)
                if raw is not None:
                    st.session_state[key] = float(raw)

    # Read the user sliders and compute the user cost AFTER the left side
    # widgets have written their values back into session_state above.
    slider_vals = current_slider_values()
    user_cost = cost_of(slider_vals, data)

    # Decide cost-metric colors once so both columns paint consistently.
    if optimal and optimal["status"] == "optimal":
        opt_cost = float(optimal["cost"])
        matches = abs(user_cost - opt_cost) < 0.01
        your_color = "#16a34a" if matches else "#dc2626"
        opt_color = "#16a34a"
        opt_value = f"{opt_cost:.2f}"
    else:
        your_color = None
        opt_color = None
        opt_value = "—"

    # Your cost: right-aligned at the bottom of the left column so it
    # visually pairs with the user slider bank.
    with your_col:
        colored_metric("Your cost", f"{user_cost:.2f}", your_color, align="right")

    with chart_col:
        # Middle column: grouped bar chart of nutrient totals (You vs
        # Optimal) with red minimum requirement rules. Bars are slim so
        # the four nutrient groups fit in the narrow middle column.
        st.markdown("**Constraints**")

        user_totals = nutrient_totals(slider_vals, data)
        rows = [
            {"nutrient": NUTRIENT_LABELS[n], "source": "You", "value": user_totals[n]}
            for n in NUTRIENTS
        ]
        if optimal and optimal["status"] == "optimal":
            opt_totals = nutrient_totals(optimal["x"], data)
            rows.extend(
                {"nutrient": NUTRIENT_LABELS[n], "source": "Optimal", "value": opt_totals[n]}
                for n in NUTRIENTS
            )

        chart_df = pd.DataFrame(rows)
        nutrient_order = [NUTRIENT_LABELS[n] for n in NUTRIENTS]

        # Tight You/Optimal pairs inside each nutrient slot:
        # - `paddingInner=0` on the xOffset scale removes the gap between
        #   the two source sub-bands within a nutrient slot.
        # - Bumping `mark_bar(size=20)` so each bar takes up ~all of its
        #   half-slot; the pair effectively touches.
        # - Bumping `paddingInner=0.5` on the x axis preserves space
        #   between adjacent nutrient groups now that the pair itself is
        #   wider.
        bars = (
            alt.Chart(chart_df)
            .mark_bar(size=20)
            .encode(
                x=alt.X(
                    "nutrient:N",
                    sort=nutrient_order,
                    title=None,
                    scale=alt.Scale(paddingInner=0.5),
                ),
                xOffset=alt.XOffset(
                    "source:N",
                    sort=["You", "Optimal"],
                    scale=alt.Scale(paddingInner=0),
                ),
                y=alt.Y("value:Q", title="Total nutrient"),
                color=alt.Color(
                    "source:N",
                    scale=alt.Scale(domain=["You", "Optimal"], range=["#FF4B4B", "#54A24B"]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("nutrient:N"),
                    alt.Tooltip("source:N"),
                    alt.Tooltip("value:Q", format=".2f"),
                ],
            )
        )
        line_df = pd.DataFrame(
            [
                {"nutrient": NUTRIENT_LABELS[n], "value": data["needs"][n],
                 "kind": "Min requirement", "source": s}
                for n in NUTRIENTS for s in ("You", "Optimal")
            ]
        )
        rules = (
            alt.Chart(line_df)
            .mark_line(strokeWidth=2, strokeDash=[3, 3], strokeCap="butt")
            .encode(
                x=alt.X("nutrient:N", sort=nutrient_order),
                xOffset=alt.XOffset("source:N", sort=["You", "Optimal"]),
                y="value:Q",
                detail="nutrient:N",
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(domain=["Min requirement"], range=["#dc2626"]),
                    legend=alt.Legend(
                        title=None,
                        symbolType="stroke",
                        symbolStrokeWidth=2,
                        symbolDash=[3, 3],
                        orient="top",
                    ),
                ),
                tooltip=[alt.Tooltip("value:Q", title="Min requirement")],
            )
        )
        # Chart height matches the slider band visually. The vertical
        # slider iframe is 298px tall (slider 220 + value badge + max/min
        # number labels + food name label), so set the chart height to
        # roughly the same so the bars span the same vertical extent as
        # the sliders flanking the chart.
        chart = (bars + rules).resolve_scale(color="independent").properties(height=300)
        st.altair_chart(chart, width="stretch")

    with opt_col:
        # Right column: always render the 6 sliders. Pre-solve they appear
        # in gray with value 0; post-solve they switch to green with the
        # solver's x_f. Pointer events are disabled via the CSS rule above
        # so they read as informational regardless of state.
        st.markdown("**Optimal diet**")
        solved = bool(optimal and optimal["status"] == "optimal")
        opt_x = optimal["x"] if solved else {}
        food_cols = st.columns(len(data["foods"]), gap="small")
        for c, f in zip(food_cols, data["foods"]):
            ub = slider_upper_bound(f, data)
            if solved:
                val = round(max(0.0, min(float(opt_x.get(f, 0.0)), ub)), 1)
                color = "#54A24B"
            else:
                val = 0.0
                color = "#cbd5e1"
            with c:
                # Key includes val so the component re-mounts and picks up
                # the new default_value when the optimal solution changes.
                # The package's JS state otherwise sticks at whatever it
                # was on first mount, ignoring subsequent default_value
                # props.
                vertical_slider(
                    label=f,
                    key=f"opt_v_{f}_{val:g}",
                    height=220,
                    default_value=val,
                    min_value=0.0,
                    max_value=float(ub),
                    step=0.1,
                    slider_color=color,
                    track_color="#E5E9F1",
                    thumb_color=color,
                    value_always_visible=True,
                )

        # Optimal cost: left-aligned at the bottom of the right column so
        # it visually pairs with the optimal slider bank.
        colored_metric("Optimal cost", opt_value, opt_color, align="left")

    # Wire up hover tooltips on the USER side and mark the OPTIMAL side
    # read only. Each user slider's parent food sub-column (stColumn) gets
    # a `data-tooltip` attribute; the CSS above renders the badge via
    # `:hover::after`. Each optimal sub-column gets a `data-readonly`
    # attribute; the CSS above sets pointer-events: none on iframes
    # underneath. Doing both via JS avoids any extra Streamlit wrapper
    # elements that would push the right column down (the previous marker
    # div trick added ~16-32px of offset on the right side).
    #
    # Streamlit's markdown sanitizer strips <script>, so we run the
    # attribute injection inside a same-origin components.html iframe that
    # reaches into window.parent.document. A MutationObserver re-applies
    # after every DOM mutation so Streamlit reruns and slider re-mounts
    # do not lose the attributes.
    nutrient_label_short = {"P": "Prot", "C": "Carb", "F": "Fat", "V": "Vit"}
    user_tooltips = []
    for f in data["foods"]:
        # Tooltip shows price and nutrient density per unit. Food name is
        # already visible as the slider's own label so we don't repeat it
        # here. Split between Carb and Fat so the tooltip is two short
        # lines instead of one long one. The CSS rule above sets
        # `white-space: pre-line` so the literal \n is rendered as a break.
        n_list = list(data["nutrients"])
        mid = len(n_list) // 2  # 2 for the default [P, C, F, V]
        line1_parts = [f"${data['price'][f]:g}"] + [
            f"{nutrient_label_short.get(n, n)} {data['content'][(f, n)]:g}"
            for n in n_list[:mid]
        ]
        line2_parts = [
            f"{nutrient_label_short.get(n, n)} {data['content'][(f, n)]:g}"
            for n in n_list[mid:]
        ]
        line1 = "  ·  ".join(line1_parts)
        line2 = "  ·  ".join(line2_parts) + "  per unit"
        user_tooltips.append(line1 + "\n" + line2)

    components.html(
        f"""
        <script>
        (function() {{
            var userTooltips = {json.dumps(user_tooltips)};
            var doc = window.parent.document;
            function disableSelectionInside(iframe) {{
                // Inject styles into the iframe's own document. Parent-page
                // CSS does not cross iframe boundaries, so anything that
                // needs to affect the slider's React-rendered content has
                // to live here. Two rules:
                //   1. `user-select: none` everywhere — kills the blue
                //      selection box when the user drags.
                //   2. Hide the slider's min-value label. The package
                //      renders min/max as <label> siblings of the slider
                //      <span>; `span + label` matches the min label only
                //      (the max label comes BEFORE the span). visibility:
                //      hidden keeps its vertical space, so the food-name
                //      label below it does not shift up onto the thumb.
                // Idempotent via a marker attribute.
                try {{
                    var d = iframe.contentDocument;
                    if (!d || !d.head || d.querySelector('style[data-diet-iframe]')) return;
                    var style = d.createElement('style');
                    style.setAttribute('data-diet-iframe', '1');
                    style.textContent = (
                        '* {{ -webkit-user-select: none !important; ' +
                        '-moz-user-select: none !important; ' +
                        'user-select: none !important; }} ' +
                        'span + label {{ visibility: hidden !important; }}'
                    );
                    d.head.appendChild(style);
                }} catch (e) {{ /* cross-origin or not ready */ }}
            }}
            function apply() {{
                // Iframe order: first 6 are user-side, next 6 are optimal.
                var iframes = doc.querySelectorAll('iframe[src*="streamlit_vertical_slider"]');
                var n = userTooltips.length;
                for (var i = 0; i < iframes.length; i++) {{
                    var col = iframes[i].closest('[data-testid="stColumn"]');
                    if (!col) continue;
                    if (i < n) {{
                        col.setAttribute('data-tooltip', userTooltips[i]);
                    }} else {{
                        col.setAttribute('data-readonly', '');
                    }}
                    disableSelectionInside(iframes[i]);
                    // Some iframes load late; retry once after a beat.
                    iframes[i].addEventListener('load', function(e) {{
                        disableSelectionInside(e.target);
                    }}, {{ once: true }});
                }}
            }}
            apply();
            if (window.parent.__dietTooltipObserver) {{
                window.parent.__dietTooltipObserver.disconnect();
            }}
            var timeout;
            var observer = new MutationObserver(function() {{
                clearTimeout(timeout);
                timeout = setTimeout(apply, 50);
            }});
            observer.observe(doc.body, {{childList: true, subtree: true}});
            window.parent.__dietTooltipObserver = observer;
        }})();
        </script>
        """,
        height=0,
    )


# ---------- Main ----------
#
# Module-level code runs on every Streamlit rerun, so this section needs to
# be cheap and idempotent: configure the page, ensure session_state is set
# up, then assemble the four tabs.

# `set_page_config` must be the first Streamlit call; "wide" layout gives
# the two-column optimizer enough horizontal room.
st.set_page_config(page_title="Diet LP Optimizer", page_icon="favicon.png", layout="wide")

# Initialize session_state defaults and apply any pending reset.
init_state()

# Tighten the top of the main block so the title sits closer to the page top
# and the tabs are visible without scrolling. 2.5rem clears the sticky header
# (running-script spinner + «« sidebar toggle in the top-right) without hiding
# the title underneath it. Same value used across the template family — see
# griffith-pse-app-template/app.py.
st.markdown(
    """
    <style>
    .block-container,
    [data-testid="stMainBlockContainer"] {
      padding-top: 2.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Home link: clicking the Griffith PSE logo navigates back to the portfolio
# site. Same-tab navigation since the user is leaving the demo. Pinned to
# the upper-left corner of the page via position:fixed so it stays visible
# while scrolling. Image is embedded from the local favicon.png as a base64
# data URL — the link still navigates to griffith-pse.com when clicked, but
# loading the page itself doesn't make any third-party request.
_FAVICON_DATA_URL = "data:image/png;base64," + base64.b64encode(
    (Path(__file__).parent / "favicon.png").read_bytes()
).decode()
st.markdown(
    """
    <style>
    .home-logo-corner {
        position: fixed;
        top: 0.5rem;
        left: 0.75rem;
        z-index: 999999;
    }
    .home-logo-corner img {
        width: 32px;
        height: 32px;
        border-radius: 4px;
        display: block;
    }
    </style>
    <a href="https://griffith-pse.com" target="_self" class="home-logo-corner">
      <img src="https://griffith-pse.com/images/favicon.png"
           alt="Griffith PSE — home" />
    </a>
    """.replace("https://griffith-pse.com/images/favicon.png", _FAVICON_DATA_URL),
    unsafe_allow_html=True,
)
st.markdown(
    "<h2 style='margin: 0 0 0.25rem 0; padding: 0; font-size: 1.5rem; font-weight: 700;'>"
    "Diet LP Optimizer "
    "<span style='font-size: 1.15rem; font-weight: 400; color: #6b7280;'>"
    "powered by "
    "<a href='https://github.com/ERGO-Code/HiGHS' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>HiGHS</a>"
    "</span>"
    "</h2>",
    unsafe_allow_html=True,
)
_caption_col, _ = st.columns([1, 1])
with _caption_col:
    st.markdown(
        "Try to beat the optimizer: pick food quantities in the **Optimizer** tab "
        "to meet every nutrient minimum at the lowest cost you can, then click "
        "**Run Optimizer** to compare against the solver's cheapest feasible diet. Edit "
        "foods, prices, and nutrient requirements in the **Data** tab; the "
        "**Formulation** and **Logs** tabs show the underlying LP and solver output."
    )

# `st.tabs` returns one container per label, used as a context manager to
# scope subsequent `st.*` calls into that tab.
optimizer_tab, data_tab, formulation_tab, logs_tab = st.tabs(
    ["🎯 Optimizer", "📋 Data", "📐 Formulation", "📜 Logs"]
)
with optimizer_tab:
    render_optimizer_tab()
with data_tab:
    render_data_tab()
with formulation_tab:
    render_formulation_tab()
with logs_tab:
    render_logs_tab()
