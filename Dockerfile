# Same shape as Knapsack — Streamlit + Pyomo + GLPK on Python 3.12 slim.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends glpk-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

EXPOSE 8080
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
