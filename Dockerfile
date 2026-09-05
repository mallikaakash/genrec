# GenRec API: verbalization + ranking head. Does NOT hold the LLM.
# The backbone lives in the vllm container; this talks to it over HTTP.
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir \
      "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
      "transformers>=4.44" numpy "fastapi[standard]" prometheus-client \
      httpx pytest

COPY src/ /app/src/
COPY tests/ /app/tests/
ENV PYTHONPATH=/app/src

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
