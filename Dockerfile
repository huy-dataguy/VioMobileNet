FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    ffmpeg \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone repo + pull model
RUN git lfs install && \
    git clone https://huggingface.co/dataguychill/MoViNet4Violence-Detection-Backup && \
    cd MoViNet4Violence-Detection-Backup && \
    git lfs pull && \
    cd .. && \
    mv MoViNet4Violence-Detection-Backup/trained_models_dropout_autolr_trlayers_NoAug ./models && \
    rm -rf MoViNet4Violence-Detection-Backup

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
