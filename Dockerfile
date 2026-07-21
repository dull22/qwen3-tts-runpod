FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Pasang dependensi
RUN pip install -U qwen-tts runpod soundfile

# flash-attn dipasang terpisah; kalau gagal build, baris ini bisa dihapus
# (handler tetap jalan tanpa flash-attn, lihat catatan di panduan)
RUN pip install -U flash-attn --no-build-isolation

# Pre-download bobot model ke dalam image (menghindari cold start lama)
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base'); \
    snapshot_download('Qwen/Qwen3-TTS-Tokenizer-12Hz')"

# Salin kode handler
COPY handler.py /app/handler.py

# Jalankan handler saat worker start
CMD ["python", "-u", "handler.py"]
