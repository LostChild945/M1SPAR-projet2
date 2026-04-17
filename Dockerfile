FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY dataset/requirements.txt /tmp/dataset-requirements.txt
COPY analyse/requirements.txt  /tmp/analyse-requirements.txt
RUN pip install --no-cache-dir \
    -r /tmp/dataset-requirements.txt \
    -r /tmp/analyse-requirements.txt

WORKDIR /app/analyse
