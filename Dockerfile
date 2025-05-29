FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/app ./src/app

RUN python3 -m src.app.download_models

COPY ./src/features/utils.py ./src/features/utils.py

EXPOSE 8000

CMD ["uvicorn", "src.app.app:app", "--reload", "--port", "8000", "--host", "0.0.0.0"]