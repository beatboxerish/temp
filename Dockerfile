FROM mcr.microsoft.com/playwright/python:v1.45.0-jammy

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN playwright install --with-deps chromium
RUN python -m nltk.downloader punkt_tab
RUN python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33')"

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
