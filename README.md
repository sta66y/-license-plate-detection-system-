# Система детекции номерных знаков

## Информация об авторе

- **ФИО**: Гультяев, Захарова
- **Группа**: 972403

## Описание

Проект представляет собой систему обнаружения автомобильных номерных знаков на основе нейросетевой архитектуры YOLOv8. Приложение поддерживает два режима работы:
- **Обработка видеофайла** — покадровая обработка с сохранением результата
- **Веб-камера** — детекция в реальном времени

## Возможности

- Детекция номерных знаков в реальном времени
- Настраиваемые пороги уверенности (confidence) и IoU
- Поддержка кастомных моделей
- Логирование всех событий в файл
- Docker-контейнеризация для простого развёртывания
- CLI-интерфейс для интеграции

## Структура проекта

```
.
├── src/
│   ├── __init__.py         # Экспорт пакета
│   ├── logger.py           # Модуль логирования (singleton)
│   ├── detector.py         # YOLODetector класс
│   ├── visualizer.py       # Функция отрисовки детекций
│   ├── model_impl.py       # My_LicensePlate_Model (публичный API)
│   └── main.py             # CLI приложение
├── models/
│   └── best.pt             # Веса обученной модели
├── data/
│   └── log_file.log        # Файл логов (создаётся при запуске)
├── videos/                 # Входные видеофайлы
├── output/                 # Обработанные видео
├── pyproject.toml          # Зависимости Poetry
├── Dockerfile              # Образ Docker
├── docker-compose.yaml     # Конфигурация Docker Compose
└── README.md               # Этот файл
```

## Требования

### Для локального запуска
- Python 3.10+
- Poetry (рекомендуется) или pip

### Для Docker
- Docker Desktop
- Docker Compose
- NVIDIA GPU + CUDA (опционально, для ускорения)

## Установка

### Вариант 1: Poetry (рекомендуется)

```bash
poetry install --only main

# macOS (CPU):
pip install torch torchvision

# Linux/Windows с NVIDIA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

poetry shell
```

### Вариант 2: pip

```bash
pip install ultralytics opencv-python-headless numpy pyyaml pillow

pip install torch torchvision

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Вариант 3: Docker

```bash
docker build -t license-plate-detector .

docker-compose up license-plate-video
```

## Запуск приложения

### Режим обработки видео

```bash
# Базовый запуск
python src/main.py --video videos/input.mp4 --output output/result.mp4

# С кастомной моделью
python src/main.py --video videos/input.mp4 --output output/result.mp4 --model models/best.pt

# С порогом уверенности 0.6
python src/main.py --video videos/input.mp4 --output output/result.mp4 --confidence 0.6

# Без окна предпросмотра
python src/main.py --video videos/input.mp4 --output output/result.mp4 --no-preview
```

### Режим веб-камеры

```bash
python src/main.py --camera

python src/main.py --camera --camera-id 1

```

### CLI-параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--video` | Путь к входному видео | — |
| `--camera` | Режим веб-камеры | — |
| `--model` | Путь к весам модели (.pt) | None |
| `--confidence` | Порог уверенности (0.0–1.0) | 0.5 |
| `--iou` | Порог IoU для NMS (0.0–1.0) | 0.45 |
| `--device` | Устройство: `cuda` или `cpu` | auto |
| `--output` | Путь к выходному видео | `output.mp4` |
| `--no-preview` | Отключить окно предпросмотра | False |
| `--camera-id` | ID камеры | 0 |

## Docker Compose

### Обработка видео

```bash
# 1. Положи видео в ./videos/input.mp4
# 2. Положи модель в ./models/best.pt
# 3. Запусти контейнер
docker-compose up license-plate-video

# Результат будет в ./output/output.mp4
```

### Веб-камера (только Linux)

```bash
# 1. Дай доступ к камере
xhost +local:docker

# 2. Запусти контейнер
docker-compose up license-plate-camera
```

## Логирование

Все события записываются в `./data/log_file.log`:
- Инициализация модели
- Статистика обработки кадров
- Ошибки и предупреждения

Просмотр логов в реальном времени:
```bash
tail -f data/log_file.log
```

## Обучение модели

Для тренировки модели на своём датасете:

1. Подготовь датасет в формате YOLO:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   ├── labels/
   │   ├── train/
   │   └── val/
   └── data.yaml
   ```

2. Запусти обучение в Google Colab или локально:
   ```python
   from ultralytics import YOLO

   model = YOLO("yolov8n.pt")
   model.train(data="data.yaml", epochs=100, imgsz=640)
   ```

3. Скопируй веса `best.pt` в папку `models/`

### Полезные ссылки

- [Документация Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Наш датасет на Roboflow](https://app.roboflow.com/ksenia-zakharova-s-workspace/cvat-rzlad/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

## Демонстрация

### Видеообработка

![Demo](./demo.gif)

## Метрики

| Метрика | Целевое значение |
|---------|------------------|
| mAP     |       0.489 


