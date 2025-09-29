# pip install speechbrain==0.5.16 numpy pandas scikit-learn soundfile joblib huggingface_hub torchaudio

from pathlib import Path
import os, subprocess, tempfile, shutil, pathlib, json, random
import numpy as np, pandas as pd, soundfile as sf, torch, torchaudio
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
from huggingface_hub import snapshot_download
from speechbrain.pretrained import SpeakerRecognition

# ---- анти-symlink (Windows) ----
_orig_symlink_to = pathlib.Path.symlink_to
def _copy_instead_of_symlink(self, target, *args, **kwargs):
    src = Path(target); dst = self
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src.exists():
            if src.is_dir(): shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                if dst.exists():
                    try:
                        if dst.is_dir(): shutil.rmtree(dst)
                        else: dst.unlink()
                    except Exception: pass
                shutil.copy2(src, dst)
    except Exception:
        return
pathlib.Path.symlink_to = _copy_instead_of_symlink
os.environ["HF_HUB_DISABLE_SYMLINKS"]="1"; os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"

# ---- пути/параметры ----
DATA_ROOT = Path("Data")                 # корень, где лежат папки-спикеры
ART_DIR   = Path("models") / "sid_multi"
ART_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ART_DIR / "sid_classifier.joblib"
LABELMAP_PATH = ART_DIR / "label_map.json"
EMB_CACHE_CSV = ART_DIR / "embeddings.csv"

FFMPEG_BIN = "ffmpeg"
MIN_SEC = 2.5
RANDOM_SEED = 42
CV_FOLDS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
LOCAL_ECAPA = Path("models") / "ecapa_local"

print(f"[DEVICE] {DEVICE} | torch {torch.__version__} | cuda={torch.version.cuda}")

def load_spk_model():
    cache = Path(snapshot_download(repo_id=MODEL_ID))
    LOCAL_ECAPA.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache, LOCAL_ECAPA, dirs_exist_ok=True)
    spk = SpeakerRecognition.from_hparams(source=str(LOCAL_ECAPA), savedir=str(LOCAL_ECAPA), run_opts={"device": DEVICE})
    return spk
spk = load_spk_model()

def ffmpeg_trim(in_path: str) -> str:
    fd, out_path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    af = ("silenceremove=start_periods=1:start_threshold=-35dB:start_silence=0.3:detection=peak,"
          "areverse,silenceremove=start_periods=1:start_threshold=-35dB:start_silence=0.3:detection=peak,areverse")
    cmd = [FFMPEG_BIN, "-y", "-i", in_path, "-ac", "1", "-ar", "16000", "-af", af, out_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_path

def valid_wav(path: str) -> bool:
    try:
        wav, sr = sf.read(path); secs = len(wav)/sr if sr else 0.0
        return secs >= MIN_SEC
    except: return False

def embed_file(path: str) -> np.ndarray:
    sig, sr = torchaudio.load(path)
    if sr != 16000: sig = torchaudio.functional.resample(sig, sr, 16000)
    sig = sig.to(DEVICE)
    with torch.no_grad():
        emb = spk.encode_batch(sig)   # [1,1,D]
    return emb.squeeze(0).squeeze(0).detach().cpu().numpy()

def discover_speakers(root: Path):
    speakers = [d for d in sorted(root.iterdir()) if d.is_dir()]
    assert speakers, f"В {root} не найдены папки со спикерами"
    print("[SPK]", ", ".join(d.name for d in speakers))
    return {d.name: d for d in speakers}

def collect_embeddings(roots: dict) -> pd.DataFrame:
    rows = []
    for label, folder in roots.items():
        wavs = sorted(folder.glob("*.wav"))
        print(f"[LOAD] {label}: {len(wavs)} файлов")
        for p in wavs:
            tmp = ffmpeg_trim(str(p))
            try:
                if not valid_wav(tmp): continue
                e = embed_file(tmp)
                rows.append({"speaker": label, "file": p.name, "path": str(p),
                             **{f"e{i}": float(v) for i, v in enumerate(e)}})
            finally:
                try: os.remove(tmp)
                except: pass
    return pd.DataFrame(rows)

def main():
    random.seed(RANDOM_SEED)
    roots = discover_speakers(DATA_ROOT)
    df = collect_embeddings(roots)
    if df.empty: raise RuntimeError("Нет валидных эмбеддингов.")

    df.to_csv(EMB_CACHE_CSV, index=False, encoding="utf-8-sig")
    print(f"[CACHE] -> {EMB_CACHE_CSV}")

    feat_cols = [c for c in df.columns if c.startswith("e")]
    X = df[feat_cols].values
    y_text = df["speaker"].values

    le = LabelEncoder()
    y = le.fit_transform(y_text)
    label_map = {int(i): cls for i, cls in enumerate(le.classes_)}
    with open(LABELMAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] label map -> {LABELMAP_PATH}")

    # Стандартизация + логрег с вероятностями (подходит для N классов)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", multi_class="auto"))
    ])

    # Кросс-валидация
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    preds, trues = [], []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", multi_class="auto"))
        ])
        model.fit(X[tr], y[tr])
        p = model.predict(X[va])
        preds.extend(p.tolist()); trues.extend(y[va].tolist())
        acc = np.mean(p == y[va])
        print(f"[CV] fold {fold} acc={acc:.3f}")

    print("\n[CV] classification report:")
    print(classification_report(trues, preds, target_names=le.classes_))

    # Дообучим на всех данных и сохраним
    clf.fit(X, y)
    dump({"pipeline": clf, "label_map": label_map, "feat_cols": feat_cols}, MODEL_PATH)
    print(f"[SAVE] model -> {MODEL_PATH}")

    yhat = clf.predict(X)
    print("\n[ALL] confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y, yhat))

if __name__ == "__main__":
    main()
