import os
from typing import Optional, TypedDict

import numpy as np
from ultralytics import YOLO


DEFAULT_MODEL = "yolov8n.pt"


class Detection(TypedDict):
    """Структура результата детекции одного объекта."""
    bbox: list[int]
    confidence: float
    class_id: int


class YOLODetector:
    """
    Обёртка над YOLO для детекции объектов на кадре.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Путь к .pt файлу весов. None — загрузит дефолтную модель.
            confidence_threshold: Минимальная уверенность детекции (0.0-1.0).
            iou_threshold: Порог IoU для NMS (0.0-1.0).
            device: 'cuda', 'cpu' или None для автовыбора.
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: Optional[str]) -> YOLO:
        """
        Загружает YOLO модель из файла или дефолтную.

        Args:
            model_path: Путь к весам или None.

        Returns:
            Загруженная YOLO модель.

        Raises:
            RuntimeError: Если модель не удалось загрузить.
        """
        try:
            path = model_path if (model_path and os.path.exists(model_path)) else DEFAULT_MODEL
            model = YOLO(path)

            if self.device:
                model.to(self.device)

            return model

        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить модель '{model_path}': {e}") from e

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Выполняет детекцию объектов на одном кадре.

        Args:
            frame: Кадр в формате BGR (numpy array).

        Returns:
            Список детекций. Пустой список если ничего не найдено.

        Raises:
            RuntimeError: Если predict завершился с ошибкой.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        return [
            Detection(
                bbox=[int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                confidence=float(conf),
                class_id=int(cls),
            )
            for box, conf, cls in zip(boxes, confidences, classes)
        ]