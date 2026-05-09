# Same shape as Knapsack — Streamlit + Pyomo + HiGHS on Python 3.12 slim.
# HiGHS ships as a pip wheel (`highspy`); no system deps needed.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py favicon.png ./

# Overwrite Streamlit's default static index.html title and favicon so the
# initial render — before the React app boots and applies set_page_config —
# already shows our app name and the blackletter-G favicon, instead of the
# default "Streamlit" title flashing for ~1s before being replaced.
RUN STATIC=$(python -c "import streamlit, os; print(os.path.join(os.path.dirname(streamlit.__file__), 'static'))") \
    && sed -i 's|<title>Streamlit</title>|<title>Diet LP Optimizer</title>|' "$STATIC/index.html" \
    && cp /app/favicon.png "$STATIC/favicon.png"

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
