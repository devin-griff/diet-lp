# Diet LP Optimizer

A Streamlit app for the classic diet linear program (Pyomo + HiGHS): minimize
food cost subject to nutrient minimums. Tune your diet with sliders; compare
cost against the LP optimum.

**Live demo:** https://diet.griffith-pse.com  
**Home:** https://griffith-pse.com

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

HiGHS ships as a pip wheel (`highspy`), so `pip install` covers everything —
no separate solver install needed.

## Deployment

Auto-deploys to Fly.io on every push to `main` via
`.github/workflows/deploy.yml`. The `Dockerfile` builds a Python 3.12 image
and installs everything from `requirements.txt`; `fly.toml` configures
auto-stop machines. Custom domain wired through Cloudflare DNS.

- **Machine**: `shared-cpu-1x` · 1 GB RAM · single region (`ord`) · `min_machines_running=0` (auto-stops on idle).
- **Cost ceiling**: ~$3.89/mo if traffic kept the VM awake 24/7.[^fly-pricing] Realistic on idle-heavy demo traffic: well under $1/mo per app. Bandwidth is effectively free under Fly's 100 GB/mo egress allowance.

[^fly-pricing]: Fly.io pricing as of 2026-05; published rates may shift. See https://fly.io/docs/about/pricing/.

## Files

- `app.py` — Streamlit UI, Pyomo model, HiGHS wrapper
- `diet.ipynb` — formulation in a notebook
- `requirements.txt` — Python deps
- `Dockerfile`, `fly.toml`, `.dockerignore` — Fly.io production image config
- `.github/workflows/deploy.yml` — auto-deploy pipeline
