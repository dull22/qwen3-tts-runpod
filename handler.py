import torch, io, base64, tempfile, os
import soundfile as sf
import runpod
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

def _siapkan_ref_audio(ref_audio):
    if ref_audio.startswith("http://") or ref_audio.startswith("https://"):
        return ref_audio, None
    audio_bytes = base64.b64decode(ref_audio)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".audio")
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name, tmp.name

def handler(event):
    inp = event["input"]

    texts = inp["text"]
    is_batch = isinstance(texts, list)
    if not is_batch:
        texts = [texts]

    langs = inp.get("language", "auto")
    if not isinstance(langs, list):
        langs = [langs] * len(texts)

    ref_audio_path, tmp_path = _siapkan_ref_audio(inp["ref_audio"])
    ref_text = inp["ref_text"]

    try:
        # Ekstrak fitur suara referensi SEKALI, pakai ulang untuk semua beat
        clone_prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=ref_text,
        )

        results = []
        # Proses satu per satu (menghindari bug batch decode)
        for teks, lang in zip(texts, langs):
            wavs, sr = model.generate_voice_clone(
                text=teks,
                language=lang,
                voice_clone_prompt=clone_prompt,
            )
            buf = io.BytesIO()
            sf.write(buf, wavs[0], sr, format="WAV")
            results.append(base64.b64encode(buf.getvalue()).decode())
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "audio_base64": results if is_batch else results[0],
        "sample_rate": sr,
    }

runpod.serverless.start({"handler": handler})
