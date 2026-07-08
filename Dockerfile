# Same shape as Knapsack — Streamlit + Pyomo + HiGHS on Python 3.12 slim.
# HiGHS ships as a pip wheel (`highspy`); no system deps needed.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py favicon.png ./
# Streamlit reads .streamlit/config.toml from the working directory; the
# theme.primaryColor in there paints user-side slider thumbs the Wong
# blue that matches the "You" series in the Constraints chart. Without
# this copy, production falls back to Streamlit's default red.
COPY .streamlit/ ./.streamlit/

# Overwrite Streamlit's default static index.html: title, favicon, and
# inject Open Graph + Twitter Card meta tags so links to this app on
# *.griffith-pse.com unfurl as a rich card on LinkedIn / Slack / iMessage.
RUN STATIC=$(python -c "import streamlit, os; print(os.path.join(os.path.dirname(streamlit.__file__), 'static'))") \
    && sed -i 's|<title>Streamlit</title>|<title>Diet</title>|' "$STATIC/index.html" \
    && sed -i 's|</head>|<link rel="icon" type="image/png" href="./favicon.png"/><meta property="og:type" content="website"/><meta property="og:title" content="Diet LP Optimizer"/><meta property="og:description" content="Stigler diet LP: minimum-cost meal plan under nutrient constraints. Pyomo + HiGHS, runs in your browser."/><meta property="og:image" content="https://griffith-pse.com/images/diet.png"/><meta property="og:site_name" content="Griffith PSE"/><meta name="twitter:card" content="summary_large_image"/><meta name="twitter:title" content="Diet LP Optimizer"/><meta name="twitter:description" content="Stigler diet LP: minimum-cost meal plan under nutrient constraints. Pyomo + HiGHS, runs in your browser."/><meta name="twitter:image" content="https://griffith-pse.com/images/diet.png"/></head>|' "$STATIC/index.html" \
    && cp /app/favicon.png "$STATIC/favicon.png" \
    && cp /app/favicon.png "$STATIC/favicon.ico"

# Run as a non-root user. If a future Streamlit (or transitive dep) RCE
# lands in the container, the attacker doesn't get root. Defense in depth.
RUN useradd -m -u 1000 streamlit && chown -R streamlit:streamlit /app
USER streamlit

EXPOSE 8080
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
