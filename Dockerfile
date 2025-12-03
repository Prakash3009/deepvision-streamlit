FROM python:3.10-slim

WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_cloud.py .
COPY yolov8n.pt .
COPY alert_sound.mp3 .

EXPOSE 8501

CMD ["streamlit", "run", "app_cloud.py", "--server.port=8501", "--server.address=0.0.0.0"]