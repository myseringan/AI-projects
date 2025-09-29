# Реальный режим идентификации (Tima vs Ulug) с микрофона.
# pip install sounddevice webrtcvad-wheels joblib torchaudio speechbrain==0.5.16 huggingface_hub

import os, sys, queue, time, shutil, json, pathlib
from pathlib import Path
import numpy as np
import sounddevice as sd
import webrtcvad
import torch, torchaudio
from joblib import load
from huggingface_hub import snapshot_download
from speechbrain.pretrained import SpeakerRecognition

# ====== Анти-symlink для Windows (как в твоих скриптах) ======
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
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ====== Пути/модель ======
SID_MODEL = Path("models") / "sid_multi" / "sid_classifier.joblib"
MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
LOCAL_DIR = Path("models") / "ecapa_local"

# ====== Аудио/онлайн-параметры ======
SR = 16000                # частота дискретизации
CHANNELS = 1              # моно
FRAME_MS = 20             # VAD фрейм (20мс = 320 сэмплов)
VAD_AGGR = 2              # агрессивность VAD: 0..3 (выше = строже к голосу)
WINDOW_SEC = 2.0          # длина окна речи для инференса
MIN_SPEECH_SEC = 1.0      # минимальная речевая длительность, чтобы считать окно валидным
SILENCE_HANG_SEC = 0.4    # сколько тишины подряд считаем концом utterance
SMOOTH_WINDOWS = 5        # сглаживание по последним N окнам (голосование/среднее)
VOTE_FRAC = 0.6           # доля голосов, чтобы зафиксировать метку

DEVICE_TORCH = "cuda" if torch.cuda.is_available() else "cpu"

def load_spk_model():
    cache = Path(snapshot_download(repo_id=MODEL_ID))
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache, LOCAL_DIR, dirs_exist_ok=True)
    spk = SpeakerRecognition.from_hparams(
        source=str(LOCAL_DIR), savedir=str(LOCAL_DIR),
        run_opts={"device": DEVICE_TORCH}
    )
    return spk

def embed_signal(spk: SpeakerRecognition, sig: torch.Tensor, sr: int) -> np.ndarray:
    """sig: torch.FloatTensor [1, T] @ sr (ожидаем 16k)"""
    if sr != 16000:
        sig = torchaudio.functional.resample(sig, sr, 16000)
        sr = 16000
    sig = sig.to(DEVICE_TORCH)
    with torch.no_grad():
        emb = spk.encode_batch(sig)  # [1, 1, D]
    return emb.squeeze(0).squeeze(0).detach().cpu().numpy()

def main():
    if not SID_MODEL.exists():
        print(f"❌ Не найдена обученная модель: {SID_MODEL}\nСначала запусти: python train_speaker_id.py")
        sys.exit(1)

    print(f"[DEVICE] torch={torch.__version__} | cuda={torch.version.cuda} | running on {DEVICE_TORCH}")

    spk = load_spk_model()
    clf_obj = load(SID_MODEL)
    clf = clf_obj["pipeline"]; label_map = clf_obj["label_map"]
    id2label = {int(k): v for k, v in label_map.items()}

    vad = webrtcvad.Vad(VAD_AGGR)

    frame_len = int(SR * FRAME_MS / 1000)  # 320
    window_len = int(SR * WINDOW_SEC)
    min_len = int(SR * MIN_SPEECH_SEC)
    hang_len = int(SR * SILENCE_HANG_SEC / (FRAME_MS / 1000))

    audio_q = queue.Queue()

    # ========= аудио-коллбек =========
    def sd_callback(indata, frames, time_info, status):
        if status:
            # можно вывести warning, но не роняем стрим
            pass
        # indata shape: [frames, channels]
        mono = np.mean(indata, axis=1).astype(np.float32)
        audio_q.put(mono.copy())

    # ========= основной цикл =========
    stream = sd.InputStream(
        channels=CHANNELS, samplerate=SR, dtype="float32",
        blocksize=frame_len, callback=sd_callback
    )
    stream.start()

    print("🎙  Реальный режим запущен. Говори в микрофон.  (Ctrl+C для выхода)\n")

    voiced_buf = []          # список 20мс фреймов с речью (float32)
    silence_count = 0
    last_preds = []          # последние результаты по окнам (id)
    last_probas = []         # последние усреднённые вероятности по окнам [C]

    try:
        while True:
            # забираем 20мс фрейм
            frame = audio_q.get()

            # WebRTC VAD ожидает int16 PCM
            pcm16 = np.clip(frame * 32768.0, -32768, 32767).astype(np.int16).tobytes()
            is_speech = vad.is_speech(pcm16, SR)

            if is_speech:
                voiced_buf.append(frame)
                silence_count = 0
            else:
                silence_count += 1

            # если накопили достаточно речи — делаем предсказание на «окне»
            total_len = sum(len(f) for f in voiced_buf)
            if total_len >= min_len and total_len >= window_len:
                # соберём ровно WINDOW_SEC (последние сэмплы)
                concat = np.concatenate(voiced_buf, axis=0)
                if len(concat) > window_len:
                    concat = concat[-window_len:]
                # тензор [1, T]
                sig = torch.from_numpy(concat).unsqueeze(0)
                emb = embed_signal(spk, sig, SR)
                proba = clf.predict_proba(emb.reshape(1, -1))[0]  # [C]
                pred_id = int(np.argmax(proba))
                pred_label = id2label[pred_id]

                # сглаживание
                last_preds.append(pred_id)
                last_probas.append(proba)
                if len(last_preds) > SMOOTH_WINDOWS:
                    last_preds = last_preds[-SMOOTH_WINDOWS:]
                    last_probas = last_probas[-SMOOTH_WINDOWS:]

                # голосование и средняя уверенность
                votes = np.bincount(last_preds, minlength=len(proba))
                winner = int(np.argmax(votes))
                need = max(1, int(np.ceil(VOTE_FRAC * len(last_preds))))
                passed = votes[winner] >= need
                mean_proba = np.mean(np.vstack(last_probas), axis=0)
                mean_conf = float(mean_proba[winner])

                # печать статуса (одна строка)
                msg = (f"\r→ pred: {id2label[winner]:<5} | conf(avg)={mean_conf:0.3f} "
                       f"| votes={votes.tolist()} need≥{need}/{len(last_preds)}      ")
                sys.stdout.write(msg)
                sys.stdout.flush()

                # «сдвигаем» окно — оставим хвост речи, чтобы быстрее давать следующее
                # берем последний 1 сек для стабильности
                keep = int(SR * 1.0)
                if len(concat) > keep:
                    tail = concat[-keep:]
                    voiced_buf = [tail]
                else:
                    voiced_buf = [concat]

            # если долго нет речи — сбрасываем буфер (конец utterance)
            if silence_count >= hang_len and voiced_buf:
                voiced_buf.clear()
                # сбросим окно, но сглаживающие предсказания оставим — приятнее UX
                sys.stdout.write("\r(тишина)".ljust(80) + "\r")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n🛑 Остановлено пользователем.")
    finally:
        stream.stop(); stream.close()

if __name__ == "__main__":
    main()
