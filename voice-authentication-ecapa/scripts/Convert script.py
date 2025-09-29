import subprocess
from pathlib import Path
import sys

# ==== НАСТРОЙКИ ====
SRC_DIR = Path("Low versions voice/Voice Dadam/voice_messages")   # исходные файлы
DST_DIR = Path("Data/Dadam")    # папка для готовых WAV
PREFIX = "D"                        # префикс имени
START = 1                           # с какого номера начинать
RATE = 16000                        # частота дискретизации
CHANNELS = 1                        # количество каналов (1 = моно)
# ====================

INPUT_EXTS = {".ogg", ".oga", ".opus", ".mp3", ".m4a", ".wav", ".flac", ".aac", ".wma"}

def has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def convert_file(src_path: Path, dst_path: Path, rate: int = 16000, channels: int = 1):
    """Конвертация файла в WAV (PCM) через ffmpeg"""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_path),
        "-ac", str(channels),
        "-ar", str(rate),
        "-vn",
        "-map_metadata", "-1",
        str(dst_path)
    ]
    subprocess.run(cmd, check=True)

def collect_files(src_dir: Path):
    files = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in INPUT_EXTS]
    files.sort(key=lambda p: p.stat().st_mtime)
    return files

def main():
    if not SRC_DIR.exists() or not SRC_DIR.is_dir():
        print(f"❌ Папка не найдена: {SRC_DIR}")
        sys.exit(1)
    DST_DIR.mkdir(parents=True, exist_ok=True)

    if not has_ffmpeg():
        print("❌ ffmpeg не найден. Установите ffmpeg и добавьте в PATH.")
        sys.exit(1)

    files = collect_files(SRC_DIR)
    if not files:
        print("❗ В папке нет подходящих файлов.")
        sys.exit(0)

    idx = START
    for f in files:
        out_name = f"{PREFIX}{idx}.wav"
        out_path = DST_DIR / out_name
        convert_file(f, out_path, RATE, CHANNELS)
        print(f"✅ {f.name}  →  {out_name}")
        idx += 1

    print(f"\nГотово! Файлы сохранены в {DST_DIR.resolve()}")

if __name__ == "__main__":
    main()
