# 🎙️ Voice Authentication Project (ECAPA-TDNN + Custom Classifier)

---

## 📌 English

### Project Overview
This repository demonstrates **speaker verification** and **multi-speaker identification** using:
- Pretrained ECAPA-TDNN embeddings (SpeechBrain)
- Custom trained classifier (Logistic Regression / SVM)
- Real-time recognition with microphone

### Repository Structure
```
Data/                # Example dataset (contains one folder per speaker, e.g. Tima/)
models/sid_multi/    # Saved trained classifier and label map
scripts/             # Training, evaluation, real-time and preprocessing scripts
requirements.txt     # Python dependencies
.gitignore           # Ignore rules
README.md            # Documentation (this file)
```

### Scripts
- **Convert script.py** → Converts audio files to WAV 16kHz mono, trims silence  
- **train_speaker_id.py** → Extracts embeddings and trains classifier on voices  
- **evaluate_voice_verification.py** → Evaluates verification metrics (FAR/FRR, threshold)  
- **realtime_speaker.py** → Real-time speaker recognition with microphone  

### Quick Start
1. Put your audio in `Data/<SpeakerName>/` (each folder = 1 person).  
   Example: `Data/Tima/T1.wav`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the classifier:
   ```bash
   python scripts/train_speaker_id.py
   ```
4. Test in real time:
   ```bash
   python scripts/realtime_speaker.py
   ```

---

## 📌 Русский

### Описание проекта
Этот репозиторий демонстрирует **верификацию голоса** и **идентификацию нескольких спикеров** с помощью:
- Предобученной модели ECAPA-TDNN (SpeechBrain)
- Собственного классификатора (Logistic Regression / SVM)
- Работы в реальном времени через микрофон

### Структура
```
Data/                # Датасет (каждый спикер в отдельной папке, например Tima/)
models/sid_multi/    # Обученный классификатор и карта меток
scripts/             # Скрипты обучения, теста, реального времени и конвертации
requirements.txt     # Зависимости Python
.gitignore           # Игнорируемые файлы
README.md            # Документация
```

### Скрипты
- **Convert script.py** → Конвертация аудио в WAV 16kHz mono, удаление тишины  
- **train_speaker_id.py** → Извлечение эмбеддингов и обучение классификатора  
- **evaluate_voice_verification.py** → Оценка метрик (FAR/FRR, подбор порога)  
- **realtime_speaker.py** → Распознавание в реальном времени через микрофон  

### Быстрый старт
1. Поместите аудио в `Data/<ИмяСпикера>/` (одна папка = один человек).  
   Пример: `Data/Tima/T1.wav`
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Обучите модель:
   ```bash
   python scripts/train_speaker_id.py
   ```
4. Запустите распознавание:
   ```bash
   python scripts/realtime_speaker.py
   ```

---

## 📌 O‘zbekcha

### Loyihaning tavsifi
Ushbu loyiha **ovozni tekshirish** va **bir nechta spikerlarni aniqlash** imkoniyatini ko‘rsatadi:
- Oldindan o‘qitilgan ECAPA-TDNN modelidan foydalanadi (SpeechBrain)
- Maxsus klassifikator (Logistic Regression / SVM)
- Mikrofon orqali real vaqt rejimida ishlash

### Tuzilishi
```
Data/                # Dastlabki dataset (har bir spiker alohida papkada, masalan Tima/)
models/sid_multi/    # O‘qitilgan klassifikator va label_map
scripts/             # O‘qitish, test, real vaqt va konvertatsiya skriptlari
requirements.txt     # Python kutubxonalari
.gitignore           # Git uchun ignore fayli
README.md            # Hujjat
```

### Skriptlar
- **Convert script.py** → Audio fayllarni WAV 16kHz mono ga o‘tkazish, sokinlikni kesish  
- **train_speaker_id.py** → Embeddinglarni ajratish va klassifikatorni o‘qitish  
- **evaluate_voice_verification.py** → Baholash metrikalari (FAR/FRR, threshold)  
- **realtime_speaker.py** → Mikrofon orqali real vaqt rejimida tanib olish  

### Tezkor start
1. Audio fayllarni `Data/<SpeakerName>/` ga joylang (har bir papka = bitta odam).  
   Masalan: `Data/Tima/T1.wav`
2. Kutubxonalarni o‘rnating:
   ```bash
   pip install -r requirements.txt
   ```
3. Modelni o‘qiting:
   ```bash
   python scripts/train_speaker_id.py
   ```
4. Real vaqt rejimida sinab ko‘ring:
   ```bash
   python scripts/realtime_speaker.py
   ```

---

## 👤 Author
**Temur Eshmurodov**
