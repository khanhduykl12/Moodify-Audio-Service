import os
import shutil
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audio-analysis")

# Essentia relies on compiled C++ binaries and runs inside Docker (Linux).
# On Windows hosts without Essentia installed, we gracefully fallback.
try:
    import essentia.standard as es  # type: ignore
    HAS_ESSENTIA = True
    logger.info("Essentia engine successfully loaded.")
except ImportError:
    es = None
    HAS_ESSENTIA = False
    logger.warning("Essentia is not installed on this host environment. Falling back to lightweight mode (use Docker for full Essentia C++ DSP extraction).")


app = FastAPI(title="Moodify Audio Feature Analysis Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "essentia-audio-extractor",
        "engine": "essentia" if HAS_ESSENTIA else "fallback"
    }

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ")

    suffix = os.path.splitext(file.filename)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        if not HAS_ESSENTIA:
            # Fallback khi test trực tiếp trên Windows không qua Docker
            return {
                "bpm": 124.0,
                "keySignature": "C Major",
                "energy": 0.65,
                "danceability": 0.58,
                "valence": 0.52,
                "acousticness": 0.25,
                "durationMs": 180000,
                "note": "Extracted in fallback mode. Run with Docker for full Essentia C++ DSP processing."
            }

        # 1. Đọc file âm thanh thành tín hiệu mono (44100Hz)
        loader = es.MonoLoader(filename=temp_path, sampleRate=44100)
        audio = loader()


        if len(audio) == 0:
            raise HTTPException(status_code=400, detail="File âm thanh rỗng hoặc định dạng không hỗ trợ")

        duration_sec = len(audio) / 44100.0
        duration_ms = int(duration_sec * 1000)

        # 2. BPM / Nhịp độ (RhythmExtractor2013)
        try:
            rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
            bpm, _, _, _, _ = rhythm_extractor(audio)
            bpm = round(float(bpm), 1)
        except Exception:
            bpm = 120.0

        # 3. Tone / Điệu tính (KeyExtractor)
        try:
            key_extractor = es.KeyExtractor()
            key, scale, _ = key_extractor(audio)
            key_signature = f"{key} {scale.capitalize()}"
        except Exception:
            key_signature = "C Major"
            scale = "major"

        # 4. Danceability (Độ nảy & bắt tai của nhịp điệu)
        try:
            dance_extractor = es.Danceability()
            danceability, _ = dance_extractor(audio)
            danceability = round(min(max(float(danceability), 0.0), 1.0), 3)
        except Exception:
            danceability = 0.5

        # 5. Energy (Năng lượng âm thanh)
        try:
            energy_val = float(es.Energy()(audio))
            normalized_energy = round(min(energy_val / (len(audio) * 0.02 + 1e-6), 1.0), 3)
            normalized_energy = max(0.05, normalized_energy)
        except Exception:
            normalized_energy = 0.6

        # 6. Valence (Sắc thái cảm xúc vui/buồn: 0.0 = buồn/u tối, 1.0 = vui/sáng)
        try:
            spec_centroid = float(es.SpectralCentroidTime()(audio))
            spectral_ratio = min(max(spec_centroid / 4000.0, 0.0), 1.0)

            # Điệu Major thiên về sáng/vui hơn Minor
            mode_weight = 0.6 if scale.lower() == "major" else 0.4
            valence = round(0.4 * spectral_ratio + 0.3 * danceability + 0.3 * mode_weight, 3)
            valence = min(max(valence, 0.05), 0.95)
        except Exception:
            valence = 0.5

        # 7. Acousticness (Độ mộc: nhạc acoustic vs nhạc điện tử tổng hợp)
        try:
            high_energy_factor = normalized_energy * 0.7 + (1.0 if bpm > 125 else 0.0) * 0.2
            acousticness = round(max(0.05, min(1.0 - high_energy_factor, 0.95)), 3)
        except Exception:
            acousticness = 0.3

        return {
            "bpm": bpm,
            "keySignature": key_signature,
            "energy": normalized_energy,
            "danceability": danceability,
            "valence": valence,
            "acousticness": acousticness,
            "durationMs": duration_ms
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi trích xuất đặc trưng âm học: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
