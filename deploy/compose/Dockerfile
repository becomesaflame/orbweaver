FROM python:3.12-slim
WORKDIR /app
COPY backend /app
COPY web /web
ENV PYTHONUNBUFFERED=1
ENV ORBWEAVER_WEB_DIR=/web
RUN pip install --no-cache-dir .
CMD ["python", "-m", "orbweaver.cli", "serve"]
