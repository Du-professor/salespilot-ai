FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/salespilot

COPY pyproject.toml ./
COPY app ./app
COPY entrypoint.sh ./entrypoint.sh

RUN pip install --upgrade pip \
    && pip install ".[rag,enterprise]" \
    && sed -i 's/\r$//' /srv/salespilot/entrypoint.sh \
    && chmod +x /srv/salespilot/entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)" || exit 1

ENTRYPOINT ["/srv/salespilot/entrypoint.sh"]
