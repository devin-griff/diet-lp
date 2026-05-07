# Same shape as Knapsack — Streamlit + Pyomo + GLPK on Python 3.12 slim.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends glpk-utils \
    && rm -rf /var/lib/apt/lists/*

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

EXPOSE 8080
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
