# Diet LP Optimizer

A Streamlit app for the classic diet linear program (Pyomo + GLPK): minimize
food cost subject to nutrient minimums. Tune your diet with sliders; compare
cost against the LP optimum.

**Live demo:** https://diet.griffith-pse.com  
**Home:** https://griffith-pse.com

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

GLPK must be on PATH:
- Debian/Ubuntu: `apt-get install glpk-utils`
- macOS: `brew install glpk`

## Deployment

Auto-deploys to Fly.io on every push to `main` via
`.github/workflows/deploy.yml`. The `Dockerfile` builds a Python 3.12 image
with `glpk-utils` and the app dependencies; `fly.toml` configures auto-stop
machines (idle = $0/mo). Custom domain wired through Cloudflare DNS.

## Files

- `app.py` — Streamlit UI, Pyomo model, GLPK wrapper
- `diet.ipynb` — formulation in a notebook
- `requirements.txt`, `packages.txt` — Python and system deps
- `Dockerfile`, `fly.toml`, `.dockerignore` — Fly.io production image config
- `.github/workflows/deploy.yml` — auto-deploy pipeline
