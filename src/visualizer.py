import cv2
import numpy as np

from detector import Detection


BBOX_COLOR = (0, 255, 0)
BBOX_THICKNESS = 2
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.5
LABEL_OFFSET_Y = 10


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """
    Рисует рамки и подписи на кадре.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)

        label = f"Plate: {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - LABEL_OFFSET_Y),
            LABEL_FONT,
            LABEL_FONT_SCALE,
            BBOX_COLOR,
            BBOX_THICKNESS,
        )

    return frame
