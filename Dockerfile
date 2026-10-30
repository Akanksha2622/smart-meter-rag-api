FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Build the vector index at image build time (also downloads the embedding model)
RUN python -m app.ingest
EXPOSE 7860
# Set GROQ_API_KEY as an environment variable / secret when running
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
