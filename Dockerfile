# syntax=docker/dockerfile:1

FROM python:3.13-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN /opt/venv/bin/pip install .

FROM python:3.13-slim
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1
RUN useradd --system --uid 10001 --no-create-home gateway
COPY --from=build /opt/venv /opt/venv
COPY gateway.example.toml /etc/llm-gateway/gateway.toml
USER gateway
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)"]
ENTRYPOINT ["llm-gateway"]
CMD ["--config", "/etc/llm-gateway/gateway.toml", "--host", "0.0.0.0", "--port", "8080"]
