# Match the previously working FR3Py container (Ubuntu 22.04/Python 3.10,
# libfranka 0.13.3) while keeping the MolmoBot robot service isolated.
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG FRANKY_VERSION=1.1.4
ARG LIBFRANKA_WHEEL_VERSION=0-13-3

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3 python3-pip unzip wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
        numpy==1.26.4 \
        fastapi==0.116.1 \
        pydantic==2.11.7 \
        uvicorn==0.35.0 \
        websockets==15.0.1 \
    && wget -q \
        "https://github.com/TimSchneider42/franky/releases/download/v${FRANKY_VERSION}/libfranka_${LIBFRANKA_WHEEL_VERSION}_wheels.zip" \
        -O /tmp/franky-wheels.zip \
    && unzip -q /tmp/franky-wheels.zip -d /tmp/franky-wheels \
    && python3 -m pip install --no-cache-dir --no-index \
        --find-links=/tmp/franky-wheels/dist "franky-control==${FRANKY_VERSION}" \
    && rm -rf /tmp/franky-wheels /tmp/franky-wheels.zip

WORKDIR /app
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "uvicorn", "--app-dir", "/app/services", "franky_service:app", "--host", "0.0.0.0", "--port", "54321", "--workers", "1"]
