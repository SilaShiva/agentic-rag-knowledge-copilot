FROM python:3.12-slim

WORKDIR /app

# System deps (git needed for reading the knowledge-base clone)
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY templates ./templates
COPY static ./static
COPY run.py .

# data/ (chroma index, audit db) is mounted as a volume in production
ENV PORT=8000
EXPOSE 8000

CMD ["python", "run.py"]
