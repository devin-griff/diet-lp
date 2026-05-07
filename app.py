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
#   - GLPK       — the LP solver, called as a subprocess via Pyomo.
#   - pandas     — DataFrame shape for Streamlit's data editor and Altair.
#   - altair     — grouped bars + horizontal "min requirement" markers.
#
# File roadmap:
#   1. Solver       — model definition + GLPK log capture.
#   2. Constants    — defaults, nutrient labels, slider cap.
#   3. State        — session_state init / reset / slider helpers.
#   4. Utilities    — DataFrame <-> internal-dict conversion, totals, cost.
#   5. LaTeX        — render the current instance as a formatted equation.
#   6. Tabs         — render_data / render_formulation / render_logs / render_optimizer.
#   7. Main         — page config and tab assembly at module bottom.
# =============================================================================

import copy
import math
import os
import tempfile

import altair as alt
import pandas as pd
import pyomo.environ as pyo
import streamlit as st
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.opt import TerminationCondition


# ---------- Solver ----------
#
# Standard Pyomo LP. The only twist is `_solve_capturing`, which works
# around the fact that GLPK's stdout comes from a subprocess and isn't
# always picked up by ordinary Python stream redirection.

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
    """Run the solver and return (results, log_text). Captures GLPK's
    subprocess stdout via two mechanisms (FD-level redirect + logfile=)
    so we get output reliably across platforms."""
    # Two capture paths run in parallel because each is unreliable on its own:
    #   1. capture_output(capture_fd=True) — redirects at the OS file
    #      descriptor level, which catches output from child processes like
    #      the GLPK binary. The capture_fd kwarg only exists in newer Pyomo,
    #      hence the TypeError fallback.
    #   2. logfile=log_path — asks GLPK itself to write its log to a file.
    #      Used as a backup if the FD capture comes back empty.
    fd, log_path = tempfile.mkstemp(suffix=".glpk.log")
    os.close(fd)
    log_text = ""
    try:
        try:
            with capture_output(capture_fd=True) as buf:
                solver = pyo.SolverFactory("glpk")
                results = solver.solve(m, tee=True, logfile=log_path)
            log_text = buf.getvalue()
        except TypeError:
            # Older Pyomo without capture_fd — fall back to the plain form.
            with capture_output() as buf:
                solver = pyo.SolverFactory("glpk")
                results = solver.solve(m, tee=True, logfile=log_path)
            log_text = buf.getvalue()
        # If the in-memory capture missed the output, read the on-disk log.
        if not log_text.strip():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    log_text = f.read()
            except OSError:
                pass
    finally:
        # Always clean up the temp log file.
        try:
            os.remove(log_path)
        except OSError:
            pass
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
        # Pyomo raises ApplicationError when the solver binary is missing
        # from PATH. Surfaces a friendly message in the UI rather than a
        # Python traceback.
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


def colored_metric(label, value, color):
    # st.metric doesn't support arbitrary value coloring, so we render a
    # metric-shaped block via raw HTML. Used to flag matching/mismatching
    # values (green if your cost equals the optimum, red otherwise).
    style_color = f"color: {color};" if color else ""
    st.markdown(
        f"<div style='margin: 0.25rem 0 1rem 0;'>"
        f"<div style='font-size: 0.875rem; color: rgba(49,51,63,0.6); margin-bottom: 0.25rem;'>{label}</div>"
        f"<div style='font-size: 2rem; font-weight: 600; line-height: 1; {style_color}'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------- Tabs ----------
#
# One render_* function per tab. Data lets the user edit foods and nutrient
# requirements; Formulation shows the math; Logs shows GLPK output;
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
    # Static reference math at the top, dynamic instance math at the bottom.
    # `st.markdown` with `$...$` renders inline LaTeX; `st.latex` renders a
    # display-style block. The general section is split across a 3-column
    # grid: the left column stacks Sets/Parameters/Variables (each as a
    # single markdown block so items stack tightly with no inter-paragraph
    # margin), the middle column holds the centered objective + constraints,
    # and the right column is empty padding so the equation lands at the
    # page midline rather than the right half.
    st.subheader("General Formulation")
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
        # Title + display math in one centered block. Using `$$...$$` inside
        # st.markdown (rather than st.latex in its own component) lets us wrap
        # both in a single text-align:center div so the equation lines up
        # under the centered title.
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

    st.divider()
    st.subheader("Instance Formulation")
    data = st.session_state.data
    if not data["foods"]:
        st.info("Add at least one food on the Data tab.")
        return
    st.latex(build_instance_latex(data))


def render_logs_tab():
    # Shows whatever GLPK printed during the last solve. The capture itself
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
    # Layout: two equal-width columns. Left has controls (buttons + per-food
    # sliders); right has a grouped bar chart and the cost metrics.
    data = st.session_state.data
    if not data["foods"]:
        st.info("Add at least one food on the Data tab.")
        return

    controls_col, chart_col = st.columns([1, 1])

    with controls_col:
        # Two action buttons side-by-side.
        b1, b2 = st.columns(2)
        run_clicked = b1.button("Run Optimizer", width="stretch")
        if run_clicked:
            # Solve and stash the dict result in session_state so subsequent
            # reruns keep showing it.
            st.session_state.optimal = solve(data)

        optimal = st.session_state.optimal
        set_disabled = not (optimal and optimal["status"] == "optimal")
        if b2.button("Set at Optimum", width="stretch", disabled=set_disabled):
            # Copy the optimal x_f values into the sliders. Each slider is
            # clamped to its own upper bound to avoid out-of-range errors.
            # `st.rerun()` forces an immediate redraw with the new slider
            # values.
            for f in data["foods"]:
                ub = slider_upper_bound(f, data)
                val = float(optimal["x"].get(f, 0.0))
                st.session_state[slider_key(f)] = max(0.0, min(val, ub))
            st.rerun()

        # Inline status messages for non-optimal solver outcomes.
        if optimal:
            if optimal["status"] == "solver_missing":
                st.error(optimal["message"])
            elif optimal["status"] == "infeasible":
                st.error("Infeasible — no diet satisfies the requirements with this data.")
            elif optimal["status"] == "unbounded":
                st.error("Unbounded problem.")
            elif optimal["status"] not in ("optimal", "no_foods"):
                st.error(f"Solver returned: {optimal['status']}")

        # One slider per food. Each slider's max depends on the current
        # data: if the user shrinks a nutrient requirement, an existing
        # slider value might exceed the new upper bound — clamp it before
        # constructing the widget so Streamlit doesn't raise.
        st.markdown("**Your diet**")
        for f in data["foods"]:
            ub = slider_upper_bound(f, data)
            key = slider_key(f)
            existing = float(st.session_state.get(key, 0.0))
            preserved = max(0.0, min(existing, ub))
            if existing != preserved:
                st.session_state[key] = preserved
            # Slider label includes price plus per-unit nutrient content for
            # the food, so the user can reason about each food's tradeoff
            # without flipping to the Data tab.
            nutrient_parts = [
                f"{NUTRIENT_LABELS.get(n, n).lower()} {data['content'][(f, n)]:g}"
                for n in data["nutrients"]
            ]
            label_str = (
                f"{f}  (price {data['price'][f]:g}, "
                + ", ".join(nutrient_parts)
                + ")"
            )
            st.slider(
                label_str,
                min_value=0.0,
                max_value=float(ub),
                value=preserved,
                step=0.1,
                key=key,
            )

        # Read the current sliders and compute the user's cost.
        slider_vals = current_slider_values()
        user_cost = cost_of(slider_vals, data)

    with chart_col:
        # Right column: a grouped bar chart of nutrient totals (You vs
        # Optimal), short red rules across each bar pair showing the
        # required minimum, and two cost metrics underneath.

        # One row per (nutrient, source) for the bars.
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
        req_df = pd.DataFrame(
            [{"nutrient": NUTRIENT_LABELS[n], "value": data["needs"][n], "kind": "Min requirement"}
             for n in NUTRIENTS]
        )

        # Display order on the x-axis matches the NUTRIENTS list.
        nutrient_order = [NUTRIENT_LABELS[n] for n in NUTRIENTS]

        # Layer 1: grouped bars. `xOffset` encoding produces side-by-side
        # bars for the two sources within each nutrient group.
        bars = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("nutrient:N", sort=nutrient_order, title=None),
                xOffset=alt.XOffset("source:N", sort=["You", "Optimal"]),
                y=alt.Y("value:Q", title="Total nutrient amount"),
                color=alt.Color(
                    "source:N",
                    scale=alt.Scale(domain=["You", "Optimal"], range=["#4C78A8", "#54A24B"]),
                    legend=alt.Legend(title=None),
                ),
                tooltip=[
                    alt.Tooltip("nutrient:N"),
                    alt.Tooltip("source:N"),
                    alt.Tooltip("value:Q", format=".2f"),
                ],
            )
        )
        # Layer 2: horizontal segments showing the minimum requirement for
        # each nutrient. Two endpoints per nutrient (one above each bar in
        # the pair) connected by a thick line, using the same xOffset scale
        # so the segment lines up with the two bars.
        line_df = pd.DataFrame(
            [
                {"nutrient": NUTRIENT_LABELS[n], "value": data["needs"][n],
                 "kind": "Min requirement", "source": s}
                for n in NUTRIENTS for s in ("You", "Optimal")
            ]
        )
        rules = (
            alt.Chart(line_df)
            .mark_line(strokeWidth=5, strokeCap="round")
            .encode(
                x=alt.X("nutrient:N", sort=nutrient_order),
                xOffset=alt.XOffset("source:N", sort=["You", "Optimal"]),
                y="value:Q",
                detail="nutrient:N",
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(domain=["Min requirement"], range=["#dc2626"]),
                    legend=alt.Legend(title=None, symbolType="stroke", symbolStrokeWidth=3),
                ),
                tooltip=[alt.Tooltip("value:Q", title="Min requirement")],
            )
        )
        # `resolve_scale(color="independent")` gives the two layers separate
        # color scales so the bar legend and rule legend coexist.
        chart = (bars + rules).resolve_scale(color="independent").properties(height=380)
        st.altair_chart(chart, width="stretch")

        # Two cost metrics centered under the chart. Color the user's cost
        # green if it matches the optimum (within $0.01), red otherwise.
        _, m1, m2, _ = st.columns([1, 1, 1, 1])
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

        with m1:
            colored_metric("Your cost", f"{user_cost:.2f}", your_color)
        with m2:
            colored_metric("Optimal cost", opt_value, opt_color)


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
# and the tabs are visible without scrolling. The minimum here is determined
# by Streamlit's sticky header (~3.75rem); going smaller hides the title
# underneath it.
st.markdown(
    """
    <style>
    .block-container,
    [data-testid="stMainBlockContainer"] {
      padding-top: 4rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Home link: clicking the Griffith PSE logo navigates back to the portfolio
# site. Same-tab navigation since the user is leaving the demo. Pinned to
# the upper-left corner of the page via position:fixed so it stays visible
# while scrolling. Image is loaded from griffith-pse.com so a single
# CDN-served copy is the source of truth across all apps.
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
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<h2 style='margin: 0 0 0.25rem 0; padding: 0; font-size: 1.5rem; font-weight: 700;'>"
    "Diet LP Optimizer"
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
