FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements*.txt ./

# base requirements
FROM base AS lite-build

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV TELEMETRY_ENABLED=false

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# lite mode
FROM base AS full-build

RUN pip install --no-cache-dir -r requirements-full.txt

COPY . .

EXPOSE 8000 8001


ENV TELEMETRY_ENABLED=true
ENV TELEMETRY_METRICS_PORT=8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

#dev mode
FROM full-build AS dev-build

RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

EXPOSE 8000 8001

ENV DEBUG=true
ENV RELOAD=true
ENV TELEMETRY_ENABLED=true
ENV TELEMETRY_CONSOLE_EXPORT=true
ENV TELEMETRY_METRICS_PORT=8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

FROM full-build AS final