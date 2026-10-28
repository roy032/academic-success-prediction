# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.12-slim

# Stage 1: train. Pass --build-arg DATA=path/in/context.csv to train on the real file;
# without it the image is trained on synthetic data with the UCI schema.
FROM ${PYTHON_IMAGE} AS train
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
ARG DATA=""
COPY . /ctx
RUN if [ -n "$DATA" ]; then asp train --data "/ctx/$DATA" --out /artifacts; \
    else asp train --out /artifacts; fi

# Stage 2: serve. No training data or build tooling in the final image.
FROM ${PYTHON_IMAGE}
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]" && useradd --create-home app
COPY --from=train /artifacts/model.joblib /app/model.joblib
USER app
ENV ASP_MODEL=/app/model.joblib
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "academic_success.serve:app", "--host", "0.0.0.0", "--port", "8000"]
