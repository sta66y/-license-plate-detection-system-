from typing import Optional

import numpy as np

from logger import setup_logger
from detector import YOLODetector, Detection
from visualizer import draw_detections


class My_LicensePlate_Model:
    """
    Модель детекции номерных знаков на основе YOLO.

    Объединяет детектор (YOLODetector) и визуализатор (draw_detections)
    в единый удобный интерфейс.

    Пример использования:
        model = My_LicensePlate_Model(model_path="models/best.pt")
        detections = model.detect_plates(frame)
        annotated, detections = model.process_frame(frame)
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
        self.logger = setup_logger()

        self.detector = YOLODetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            device=device,
        )

        self.logger.info("My_LicensePlate_Model успешно инициализирована")

    def detect_plates(self, frame: np.ndarray) -> list[Detection]:
        """
        Детектирует номерные знаки на одном кадре.

        Args:
            frame: Кадр изображения в формате BGR.

        Returns:
            Список детекций, каждая содержит:
                - bbox: [x1, y1, x2, y2]
                - confidence: вероятность (0.0-1.0)
                - class_id: id класса из модели
            Возвращает пустой список если ничего не найдено или произошла ошибка.
        """
        try:
            detections = self.detector.detect(frame)
            self.logger.debug(f"Обнаружено номеров: {len(detections)}")
            return detections
        except Exception as e:
            self.logger.error(f"Ошибка во время детекции: {e}")
            return []

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        """
        Обрабатывает кадр: детектирует номера и рисует рамки.

        Args:
            frame: Входной кадр в формате BGR.

        Returns:
            Кортеж из:
                - кадр с нарисованными рамками и подписями
                - список детекций
        """
        detections = self.detect_plates(frame)
        annotated_frame = draw_detections(frame.copy(), detections)
        return annotated_frame, detections