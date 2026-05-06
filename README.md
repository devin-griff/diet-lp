# Diet LP Optimizer

A Streamlit app for the classic diet linear program (Pyomo + GLPK): minimize
food cost subject to nutrient minimums. Tune your diet with sliders; compare
cost against the LP optimum.

**Live demo:** https://dietlp.streamlit.app/

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

GLPK must be on PATH. On Streamlit Cloud, packages.txt handles `glpk-utils`.

## Files

- `app.py` — Streamlit UI, Pyomo model, GLPK wrapper
- `diet.ipynb` — formulation in a notebook
- `requirements.txt`, `packages.txt` — Python deps and system packages
