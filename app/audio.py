import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/ogg",
}

def encode_audio(audio: np.ndarray, sample_rate: int, fmt: str, mp3_quality: int = 0) -> bytes:
    if fmt in {"wav", "flac"}:
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format=fmt.upper())
        return buffer.getvalue()

    with tempfile.TemporaryDirectory(prefix="typingo-kokoro-") as tmp:
        tmp_dir = Path(tmp)
        wav_path = tmp_dir / "input.wav"
        out_path = tmp_dir / ("output.mp3" if fmt == "mp3" else "output.opus")

        sf.write(wav_path, audio, sample_rate, format="WAV")

        codec_args = (
            ["-codec:a", "libmp3lame", "-q:a", str(mp3_quality)]
            if fmt == "mp3"
            else ["-codec:a", "libopus", "-b:a", "64k"]
        )

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(wav_path),
                "-threads", "1",
                *codec_args,
                str(out_path),
            ],
            check=True,
        )

        return out_path.read_bytes()
