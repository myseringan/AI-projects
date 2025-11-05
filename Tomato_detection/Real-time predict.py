import time
import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Настройки
MODEL_PATH = "best_model.pth"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["ripe", "unripe"]
CONF_THRESHOLD = 0.65   # менять по результатам eval
SKIP_FRAME = 5          # обрабатывать каждый 2-й кадр (чтобы поднять fps)

# Трансформ
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Загрузка модели
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Вспомогательные функции
def predict_frame(frame_bgr):
    # frame: BGR numpy (OpenCV)
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    x = transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx])

# Запуск камеры
cap = cv2.VideoCapture(0)  # 0 или индекс камеры
if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру")
    exit(1)

last_label = ""
last_conf = 0.0
frame_idx = 0
fps_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    frame_count += 1

    # Предсказание с пропуском кадров
    if frame_idx % SKIP_FRAME == 0:
        label, conf = predict_frame(frame)
        last_label, last_conf = label, conf

    # Оверлей результата
    text = f"{last_label} {last_conf:.2f}"
    # цвет: ripe -> зелёный, unripe -> синий/желтый
    color = (0,200,0) if last_label=="ripe" else (0,140,255)
    if last_conf < CONF_THRESHOLD:
        color = (0,0,255)  # low confidence -> red
        text += " (check)"

    cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("PomidorVision - realtime", frame)

    # FPS print каждую секунду
    if time.time() - fps_time >= 1.0:
        print("FPS:", frame_count / (time.time()-fps_time))
        fps_time = time.time()
        frame_count = 0

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC чтобы выйти
        break

cap.release()
cv2.destroyAllWindows()
