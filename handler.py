import torch, io, base64
import soundfile as sf
import runpod
from qwen_tts import Qwen3TTSModel

# ============================================================
# Dimuat SEKALI saat worker start (bukan tiap request).
# Ini yang bikin request kedua dan seterusnya jadi cepat.
# ============================================================
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

def handler(event):
    inp = event["input"]

    # Terima teks tunggal (string) ATAU banyak klausa (list)
    texts = inp["text"]
    is_batch = isinstance(texts, list)
    if not is_batch:
        texts = [texts]

    # Bahasa: boleh satu untuk semua, atau list per klausa
    langs = inp.get("language", "Indonesian")
    if not isinstance(langs, list):
        langs = [langs] * len(texts)

    # Jalankan voice clone (batch sekaligus kalau banyak)
    wavs, sr = model.generate_voice_clone(
        text=texts,
        language=langs,
        ref_audio=inp["ref_audio"],   # URL / path / base64 audio referensi
        ref_text=inp["ref_text"],     # transkrip persis audio referensi
    )

    # Ubah tiap hasil audio jadi base64
    results = []
    for w in wavs:
        buf = io.BytesIO()
        sf.write(buf, w, sr, format="WAV")
        results.append(base64.b64encode(buf.getvalue()).decode())

    # Kembalikan list kalau input batch, single kalau input tunggal
    return {
        "audio_base64": results if is_batch else results[0],
        "sample_rate": sr,
    }

# Mulai worker serverless
runpod.serverless.start({"handler": handler})