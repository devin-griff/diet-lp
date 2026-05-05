import copy
import math

import altair as alt
import pandas as pd
import streamlit as st

from solver import solve

NUTRIENTS = ["P", "C", "F", "V"]
NUTRIENT_LABELS = {"P": "Protein", "C": "Carbs", "F": "Fat", "V": "Vitamins"}
MAX_FOODS = 10
SLIDER_CAP = 50.0

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


def slider_key(food):
    return f"slider_{food}"


def init_state():
    if "data" not in st.session_state:
        st.session_state.data = copy.deepcopy(DEFAULT_DATA)
    if "optimal" not in st.session_state:
        st.session_state.optimal = None
    for f in st.session_state.data["foods"]:
        if slider_key(f) not in st.session_state:
            st.session_state[slider_key(f)] = 0.0
    if st.session_state.pop("_pending_reset", False):
        apply_reset()


def apply_reset():
    st.session_state.data = copy.deepcopy(DEFAULT_DATA)
    st.session_state.optimal = None
    for f in DEFAULT_DATA["foods"]:
        st.session_state[slider_key(f)] = 0.0
    for n in NUTRIENTS:
        st.session_state[f"need_{n}"] = float(DEFAULT_DATA["needs"][n])


def current_slider_values():
    return {
        f: float(st.session_state.get(slider_key(f), 0.0))
        for f in st.session_state.data["foods"]
    }


def data_to_df(data):
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
    df = df.copy()
    df["Food"] = df["Food"].astype("string").str.strip()
    df = df.dropna(subset=["Food"])
    df = df[df["Food"] != ""]
    df = df.drop_duplicates(subset=["Food"], keep="first")
    for col in NUTRIENTS + ["Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    foods = df["Food"].tolist()
    price = {row.Food: float(row.Price) for row in df.itertuples()}
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
    bounds = []
    for n in NUTRIENTS:
        c = data["content"].get((food, n), 0.0)
        if c > 0:
            bounds.append(data["needs"][n] / c)
    if not bounds:
        return SLIDER_CAP
    return float(min(SLIDER_CAP, max(1.0, math.ceil(max(bounds)))))


def nutrient_totals(x, data):
    return {
        n: sum(data["content"].get((f, n), 0.0) * float(x.get(f, 0.0)) for f in data["foods"])
        for n in NUTRIENTS
    }


def cost_of(x, data):
    return sum(data["price"][f] * float(x.get(f, 0.0)) for f in data["foods"])


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
    for raw, esc in _LATEX_ESCAPE:
        s = s.replace(raw, esc)
    return f"\\text{{{s}}}"


def _build_lhs(coefs, foods):
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
    foods = data["foods"]
    obj = _build_lhs(data["price"], foods)
    rows = [r"\min \quad & " + obj + r" \\"]
    for i, n in enumerate(NUTRIENTS):
        coefs = {f: data["content"].get((f, n), 0.0) for f in foods}
        lhs = _build_lhs(coefs, foods)
        rhs = f"{data['needs'][n]:g}"
        label = NUTRIENT_LABELS[n]
        prefix = r"\text{s.t.} \quad & " if i == 0 else r"& "
        rows.append(f"{prefix}{lhs} \\ge {rhs} \\quad \\text{{({label})}} \\\\")
    bounds_lhs = ", ".join(_latex_text(f) for f in foods)
    rows.append(f"& {bounds_lhs} \\ge 0")
    body = r"\begin{aligned}" + "\n".join(rows) + r"\end{aligned}"
    if len(foods) > 7:
        body = r"\small " + body
    return body


def colored_metric(label, value, color):
    style_color = f"color: {color};" if color else ""
    st.markdown(
        f"<div style='margin: 0.25rem 0 1rem 0;'>"
        f"<div style='font-size: 0.875rem; color: rgba(49,51,63,0.6); margin-bottom: 0.25rem;'>{label}</div>"
        f"<div style='font-size: 2rem; font-weight: 600; line-height: 1; {style_color}'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_data_tab():
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

    warnings = []
    if len(edited) > MAX_FOODS:
        warnings.append(f"Capped at {MAX_FOODS} foods; extra rows ignored.")
        edited = edited.head(MAX_FOODS)

    names = edited["Food"].dropna().astype("string").str.strip()
    if names.duplicated().any():
        warnings.append("Duplicate food names were dropped (kept the first).")

    new_data = df_to_data(edited, new_needs)

    if new_data != st.session_state.data:
        st.session_state.data = new_data
        st.session_state.optimal = None
        for f in new_data["foods"]:
            if slider_key(f) not in st.session_state:
                st.session_state[slider_key(f)] = 0.0
        st.rerun()

    for w in warnings:
        st.warning(w)

    if st.button("Reset to defaults"):
        st.session_state["_pending_reset"] = True
        st.rerun()


def render_formulation_tab():
    st.subheader("General Formulation")

    st.markdown("**Sets**")
    st.markdown(r"$\mathcal{F} = \{\text{foods}\}$")
    st.markdown(r"$\mathcal{N} = \{\text{nutrients}\}$")

    st.markdown("**Parameters**")
    st.markdown(r"$p_i$ price for food option $i \in \mathcal{F}$")
    st.markdown(r"$r_j$ nutrition requirement for nutrient $j \in \mathcal{N}$")
    st.markdown(r"$D_{ij}$ nutrition info for food $i \in \mathcal{F}$ and nutrient $j \in \mathcal{N}$")

    st.markdown("**Variables**")
    st.markdown(r"$x_i$ amount of food $i \in \mathcal{F}$ eaten or purchased")

    st.markdown("**Objective and Constraints**")
    st.latex(r"""
    \begin{gathered}
    \min_x \sum_{i \in \mathcal{F}} x_i p_i \quad \text{(cost)} \\
    \text{s.t.} \quad \sum_{i \in \mathcal{F}} D_{ij} x_i \ge r_j \quad \forall j \in \mathcal{N} \quad \text{(nutrient minimums)} \\
    x_i \ge 0 \quad \forall i \in \mathcal{F} \quad \text{(lower bounds)}
    \end{gathered}
    """)

    st.divider()
    st.subheader("Instance Formulation")
    data = st.session_state.data
    if not data["foods"]:
        st.info("Add at least one food on the Data tab.")
        return
    st.latex(build_instance_latex(data))


def render_logs_tab():
    st.subheader("GLPK solver output")
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
    data = st.session_state.data
    if not data["foods"]:
        st.info("Add at least one food on the Data tab.")
        return

    controls_col, chart_col = st.columns([1, 1])

    with controls_col:
        b1, b2 = st.columns(2)
        run_clicked = b1.button("Run Optimizer", width="stretch")
        if run_clicked:
            st.session_state.optimal = solve(data)

        optimal = st.session_state.optimal
        set_disabled = not (optimal and optimal["status"] == "optimal")
        if b2.button("Set at Optimum", width="stretch", disabled=set_disabled):
            for f in data["foods"]:
                ub = slider_upper_bound(f, data)
                val = float(optimal["x"].get(f, 0.0))
                st.session_state[slider_key(f)] = max(0.0, min(val, ub))
            st.rerun()

        if optimal:
            if optimal["status"] == "solver_missing":
                st.error(optimal["message"])
            elif optimal["status"] == "infeasible":
                st.error("Infeasible — no diet satisfies the requirements with this data.")
            elif optimal["status"] == "unbounded":
                st.error("Unbounded problem.")
            elif optimal["status"] not in ("optimal", "no_foods"):
                st.error(f"Solver returned: {optimal['status']}")

        st.markdown("**Your diet**")
        for f in data["foods"]:
            ub = slider_upper_bound(f, data)
            key = slider_key(f)
            existing = float(st.session_state.get(key, 0.0))
            preserved = max(0.0, min(existing, ub))
            if existing != preserved:
                st.session_state[key] = preserved
            st.slider(
                f"{f}  (price {data['price'][f]:g})",
                min_value=0.0,
                max_value=float(ub),
                value=preserved,
                step=0.1,
                key=key,
            )

        slider_vals = current_slider_values()
        user_cost = cost_of(slider_vals, data)

    with chart_col:
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

        nutrient_order = [NUTRIENT_LABELS[n] for n in NUTRIENTS]
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
        # Horizontal segment centered on each nutrient's bar pair: connect points at
        # the You-bar and Optimal-bar centers via the same xOffset scale as the bars.
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
        chart = (bars + rules).resolve_scale(color="independent").properties(height=380)
        st.altair_chart(chart, width="stretch")

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


st.set_page_config(page_title="Diet LP Optimizer", layout="wide")
init_state()
st.title("Diet LP Optimizer")
optimizer_tab, data_tab, formulation_tab, logs_tab = st.tabs(
    ["Optimizer", "Data", "Formulation", "Logs"]
)
with optimizer_tab:
    render_optimizer_tab()
with data_tab:
    render_data_tab()
with formulation_tab:
    render_formulation_tab()
with logs_tab:
    render_logs_tab()
