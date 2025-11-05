import os
import time
import json

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# --- Параметры ---
DATA_DIR = "Datasets"
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"
CLASSES = ["ripe", "unripe"]
PATIENCE = 4
NUM_WORKERS = 0  # <- на Windows можно оставить 0, либо поставить 2 если всё в main()

# --- Трансформации ---
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
    transforms.RandomRotation(12),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def build_model(num_classes):
    # Современный способ загрузки весов (torchvision >= 0.13)
    try:
        weights_enum = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights_enum)
    except Exception:
        # fallback для старых версий torchvision
        model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def main():
    # Если запускаешь на Windows и хочешь многопроцессную загрузку датасета,
    # ставь NUM_WORKERS > 0. Пока оставляем 0 для стабильности.
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR,"train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR,"validation"), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model(len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    best_val_acc = 0.0
    no_improve = 0
    history = {"train_loss":[], "val_acc":[], "val_loss":[]}

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_ds)

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item() * imgs.size(0)
                probs = torch.softmax(out, dim=1)
                preds = probs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        val_loss = val_loss / len(val_ds)
        val_acc = correct / total
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={time.time()-t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("Saved best model")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping")
                break

    # save history
    with open("train_history.json","w") as f:
        json.dump(history, f)

    # Final report on validation
    print("Final validation classification report:")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    print(classification_report(y_true, y_pred, target_names=CLASSES))
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    # Для Windows: безопасный старт дочерних процессов (если используешь spawn)
    try:
        import multiprocessing
        multiprocessing.freeze_support()
    except Exception:
        pass
    main()
