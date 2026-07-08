FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

COPY app ./app
COPY corpus ./corpus
COPY scripts ./scripts
COPY docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
