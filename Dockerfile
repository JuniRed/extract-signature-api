FROM python:3.11-slim

RUN apt-get update && apt-get install -y poppler-utils

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
