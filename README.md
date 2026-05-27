# Diet LP Optimizer

A Streamlit app for the classic diet linear program (Pyomo + HiGHS): minimize
food cost subject to nutrient minimums. Tune your diet with sliders; compare
cost against the LP optimum. The **📐 Formulation** tab in the app walks
through Stigler's 1945 hand calculation, Dantzig's LP improvement, and the
references. See [References](#references) below.

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
- **Cost ceiling**: ~$3.89/mo if traffic kept the VM awake 24/7. Realistic on idle-heavy demo traffic: well under $1/mo per app. Bandwidth is effectively free under Fly's 100 GB/mo egress allowance.

## Files

- `app.py` — Streamlit UI, Pyomo model, HiGHS wrapper
- `diet.ipynb` — formulation in a notebook
- `requirements.txt` — Python deps
- `Dockerfile`, `fly.toml`, `.dockerignore` — Fly.io production image config
- `.github/workflows/deploy.yml` — auto-deploy pipeline

## References

[1] G. J. Stigler, "The Cost of Subsistence," *Journal of Farm Economics*,
vol. 27, no. 2, pp. 303–314, 1945.
[JSTOR](https://www.jstor.org/stable/1231810)

[2] G. B. Dantzig, "The Diet Problem," *Interfaces*, vol. 20, no. 4,
pp. 43–47, 1990.
[INFORMS](https://pubsonline.informs.org/doi/abs/10.1287/inte.20.4.43)

[3] Q. Huangfu and J. A. J. Hall, "Parallelizing the dual revised simplex
method," *Mathematical Programming Computation*, vol. 10, no. 1,
pp. 119–142, 2018.
[Springer](https://link.springer.com/article/10.1007/s12532-017-0130-5)

[4] M. L. Bynum, G. A. Hackebeil, W. E. Hart, C. D. Laird, B. L. Nicholson,
J. D. Siirola, J.-P. Watson, and D. L. Woodruff, *Pyomo — Optimization
Modeling in Python*, 3rd ed. Cham: Springer, 2021.
[Springer](https://link.springer.com/book/10.1007/978-3-030-68928-5)
