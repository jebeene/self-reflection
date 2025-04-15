FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    ffmpeg \
    libgtk2.0-dev \
    libgtk-3-dev \
    tk \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY .. /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    tensorflow==2.13.0 \
    opencv-python-headless \
    deepface \
    numpy \
    pillow

# Set display environment variable for GUI
ENV DISPLAY=:0

# Run the script
CMD ["python", "self-reflection.py"]
