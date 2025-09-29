# pip install speechbrain==0.5.16 soundfile numpy pandas huggingface_hub

from pathlib import Path
import os, subprocess, tempfile, random, shutil, pathlib
import numpy as np, pandas as pd, soundfile as sf
import torch

# ===== 0) Патчируем symlink: любая ссылка -> копирование (и не падаем, если src нет) =====
_orig_symlink_to = pathlib.Path.symlink_to
def _copy_instead_of_symlink(self, target, *args, **kwargs):
    src = Path(target)
    dst = self
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src.exists():
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                if dst.exists():
                    try:
                        if dst.is_dir(): shutil.rmtree(dst)
                        else: dst.unlink()
                    except Exception:
                        pass
                shutil.copy2(src, dst)
        else:
            # если источника нет — тихо пропускаем (SpeechBrain всё равно затем загрузит готовый файл)
            return
    except Exception:
        # на всякий случай — не валимся
        return
pathlib.Path.symlink_to = _copy_instead_of_symlink

# ===== 1) Отключаем предупреждения HF про симлинки/телеметрию =====
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from huggingface_hub import snapshot_download
from speechbrain.pretrained import SpeakerRecognition

# ================= НАСТРОЙКИ =================
TIMA_DIR  = Path("")      # папка с WAV Тимы
ULUG_DIR  = Path("")      # папка с WAV Улуга

USE_FIRST_N_TIMA_FOR_ENROLL = 10        # сколько файлов Тимы на "enroll"
MIN_SEC = 2.0                           # минимальная длительность после чистки (сек)
THRESH_GRID = np.linspace(-5, 5, 401)   # сетка порогов для verify_score
SHUFFLE = True
RANDOM_SEED = 42

SAVE_CSV = True
CSV_PATH = "ecapa_verify_scores.csv"

FFMPEG_BIN = "ffmpeg"                   # при необходимости: r"C:\ffmpeg\bin\ffmpeg.exe"
MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
LOCAL_MODEL_DIR = Path("models") / "ecapa_local"  # локальная папка с моделью
# ===========================================

# -------- устройство --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] Using: {DEVICE} | torch {torch.__version__} | cuda={torch.version.cuda}")

# -------- helper: гарантируем наличие ключевых файлов модели --------
NEEDED_FILES = {
    "hyperparams.yaml",
    "custom.py",               # критичный файл — из-за него падали
    "embedding_model.ckpt",    # названия в репо могут отличаться, но этот есть
    "classifier.ckpt",
    "mean_var_norm_emb.ckpt",
    "mean_var_norm_classifier.ckpt",
    "label_encoder.txt",
}

def ensure_model_files(cache_path: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    # копируем всё содержимое снапшота (на всякий случай)
    shutil.copytree(cache_path, dst_dir, dirs_exist_ok=True)
    # дополнительно проверяем нужные файлы и пытаемся найти/скопировать их по имени
    for fname in NEEDED_FILES:
        dst = dst_dir / fname
        if not dst.exists():
            candidates = list(cache_path.rglob(fname))
            if candidates:
                shutil.copy2(candidates[0], dst)

# -------- загрузка модели без настоящих symlink --------
def load_ecapa_model():
    cache_path = Path(snapshot_download(repo_id=MODEL_ID))
    ensure_model_files(cache_path, LOCAL_MODEL_DIR)
    spk = SpeakerRecognition.from_hparams(
        source=str(LOCAL_MODEL_DIR),
        savedir=str(LOCAL_MODEL_DIR),     # любые "symlink" превратятся в копирование
        run_opts={"device": DEVICE}
    )
    return spk

spk = load_ecapa_model()

# -------- препроцесс: срез тишины + 16kHz mono --------
def ffmpeg_trim_silence(in_path: str) -> str:
    fd, out_path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    af = (
        "silenceremove=start_periods=1:start_threshold=-35dB:start_silence=0.3:detection=peak,"
        "areverse,silenceremove=start_periods=1:start_threshold=-35dB:start_silence=0.3:detection=peak,areverse"
    )
    cmd = [FFMPEG_BIN, "-y", "-i", in_path, "-ac", "1", "-ar", "16000", "-af", af, out_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_path

def valid_wav(path: str) -> bool:
    try:
        wav, sr = sf.read(path)
        secs = len(wav)/sr if sr else 0.0
        return secs >= MIN_SEC
    except:
        return False

# -------- скоринг пары файлов через ECAPA --------
def score_pair(a_path: str, b_path: str) -> float:
    """
    Возвращает числовой скор похожести.
    Предпочтительно используем score_files (возвращает только скор).
    Делаем fallback на verify_files (который может вернуть (score, decision)).
    """
    try:
        s = spk.score_files(a_path, b_path)  # тензор со скором
    except Exception:
        out = spk.verify_files(a_path, b_path)
        # out может быть тензором или кортежем (score, decision)
        s = out[0] if isinstance(out, (tuple, list)) else out

    # аккуратно приводим к float
    if hasattr(s, "detach"):
        s = s.detach()
    if hasattr(s, "cpu"):
        s = s.cpu()
    try:
        return float(np.array(s).squeeze())
    except Exception:
        return float(s)

def main():
    if SHUFFLE:
        random.seed(RANDOM_SEED)

    tima_files = sorted(TIMA_DIR.glob("*.wav"))
    ulug_files = sorted(ULUG_DIR.glob("*.wav"))
    if SHUFFLE:
        random.shuffle(tima_files)
        random.shuffle(ulug_files)

    assert len(tima_files) >= USE_FIRST_N_TIMA_FOR_ENROLL, \
        f"Нужно минимум {USE_FIRST_N_TIMA_FOR_ENROLL} файлов Тимы, найдено {len(tima_files)}"

    enroll_src = tima_files[:USE_FIRST_N_TIMA_FOR_ENROLL]
    tima_test  = tima_files[USE_FIRST_N_TIMA_FOR_ENROLL:]
    ulug_test  = ulug_files

    print(f"[INFO] Tima: всего {len(tima_files)}, enroll {len(enroll_src)}, test {len(tima_test)}")
    print(f"[INFO] Ulug: всего {len(ulug_files)}, test {len(ulug_test)}")

    # готовим очищенные временные версии
    def prep_list(files):
        out = []
        for p in files:
            tmp = ffmpeg_trim_silence(str(p))
            if valid_wav(tmp):
                out.append(tmp)
            else:
                try: os.remove(tmp)
                except: pass
        return out

    enroll = prep_list(enroll_src)
    tima   = prep_list(tima_test)
    ulug   = prep_list(ulug_test)

    if len(enroll) == 0 or (len(tima) + len(ulug)) == 0:
        raise RuntimeError("После чистки слишком мало валидных файлов")

    # считаем средний verify_score по всем enroll-сэмплам
    rows = []
    for path in tima:
        s = float(np.mean([score_pair(e, path) for e in enroll]))
        rows.append({"file": Path(path).name, "speaker": "Tima", "score": s, "label": 1})
    for path in ulug:
        s = float(np.mean([score_pair(e, path) for e in enroll]))
        rows.append({"file": Path(path).name, "speaker": "Ulug", "score": s, "label": 0})

    df = pd.DataFrame(rows)
    pos = df[df.label == 1]["score"].values
    neg = df[df.label == 0]["score"].values

    # подбор порога (минимум |FAR - FRR|, при равенстве — больше F1)
    best = None
    for thr in THRESH_GRID:
        tp = int(np.sum(pos >= thr)); fn = int(np.sum(pos < thr))
        fp = int(np.sum(neg >= thr)); tn = int(np.sum(neg < thr))
        far = fp / max(len(neg), 1)
        frr = fn / max(len(pos), 1)
        eer = abs(far - frr)
        f1  = tp / max(tp + 0.5*(fp + fn), 1e-9)
        cand = {"thr":thr,"far":far,"frr":frr,"eer":eer,"tp":tp,"fn":fn,"fp":fp,"tn":tn,"f1":f1}
        if best is None or cand["eer"] < best["eer"] or (cand["eer"] == best["eer"] and cand["f1"] > best["f1"]):
            best = cand

    thr = best["thr"]
    print("\n=== ECAPA — подобранный порог ===")
    print(f"Threshold: {thr:.2f} | FAR: {best['far']:.3f} | FRR: {best['frr']:.3f} | F1≈{best['f1']:.3f}")
    print(f"TP={best['tp']}, FN={best['fn']}, FP={best['fp']}, TN={best['tn']}")

    # разбор ошибок
    df["pred"] = (df["score"] >= thr).astype(int)
    errors = df[df["pred"] != df["label"]]
    fa = errors[(errors["label"] == 0) & (errors["pred"] == 1)]
    fr = errors[(errors["label"] == 1) & (errors["pred"] == 0)]

    print("\n— ЛОЖНЫЕ ПРИЁМЫ (FA): топ 10")
    print(fa.sort_values("score", ascending=False).head(10)[["file","speaker","score"]].to_string(index=False) if not fa.empty else "нет")
    print("\n— ОТКАЗЫ СВОИМ (FR): топ 10")
    print(fr.sort_values("score", ascending=True).head(10)[["file","speaker","score"]].to_string(index=False) if not fr.empty else "нет")

    if SAVE_CSV:
        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"\n[CSV] Сохранено: {CSV_PATH}")

    # уборка временных файлов
    for p in enroll + tima + ulug:
        try: os.remove(p)
        except: pass

if __name__ == "__main__":
    main()

