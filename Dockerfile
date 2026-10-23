FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GROUNDCHECK_HOST=0.0.0.0 \
    GROUNDCHECK_PORT=8000

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data

EXPOSE 8000

CMD ["python3", "-m", "groundcheck.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
