import argparse
import logging
import time

import cv2

from logger import setup_logger
from model_impl import My_LicensePlate_Model


MAX_CAMERA_FAILURES = 10


def process_video_file(
    model: My_LicensePlate_Model,
    video_path: str,
    output_path: str,
    show_preview: bool = False,
) -> None:
    """
    Обрабатывает видеофайл покадрово и сохраняет результат.

    Args:
        model: Инициализированная модель детекции.
        video_path: Путь к входному видео.
        output_path: Путь куда сохранить обработанное видео.
        show_preview: Показывать ли окно предпросмотра во время обработки.

    Raises:
        ValueError: Если не удалось открыть видео или создать выходной файл.
    """
    logger = logging.getLogger("license_plate_detector")
    logger.info(f"Обработка видеофайла: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видеофайл: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Параметры видео: {width}x{height} @ {fps:.2f} FPS, кадров: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Не удалось создать выходной файл: {output_path}")

    frame_count = 0
    detection_total = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            processed_frame, detections = model.process_frame(frame)
            detection_total += len(detections)
            out.write(processed_frame)

            if show_preview:
                cv2.imshow("License Plate Detection", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Обработка прервана пользователем")
                    break

            if frame_count % 50 == 0:
                progress = 100 * frame_count / total_frames
                logger.info(
                    f"Прогресс: {progress:.1f}% "
                    f"({frame_count}/{total_frames} кадров), "
                    f"найдено номеров: {detection_total}"
                )

    except Exception as e:
        logger.error(f"Ошибка при обработке видео: {e}")
        raise
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    logger.info("=" * 50)
    logger.info("ОБРАБОТКА ЗАВЕРШЕНА")
    logger.info(f"Вход: {video_path} → Выход: {output_path}")
    logger.info(f"Обработано кадров: {frame_count}, найдено номеров: {detection_total}")
    logger.info("=" * 50)


def process_webcam_stream(
    model: My_LicensePlate_Model,
    camera_id: int = 0,
    show_preview: bool = True,
) -> None:
    """
    Обрабатывает живой поток с веб-камеры.

    Args:
        model: Инициализированная модель детекции.
        camera_id: ID камеры (обычно 0 для встроенной).
        show_preview: Показывать ли окно предпросмотра.

    Raises:
        ValueError: Если не удалось открыть камеру.
        RuntimeError: Если камера перестала отвечать.
    """
    logger = logging.getLogger("license_plate_detector")
    logger.info(f"Запуск потока с камеры ID: {camera_id}")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть камеру: {camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    detection_total = 0
    consecutive_failures = 0
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                consecutive_failures += 1
                logger.warning(
                    f"Не удалось получить кадр "
                    f"({consecutive_failures}/{MAX_CAMERA_FAILURES})"
                )
                if consecutive_failures >= MAX_CAMERA_FAILURES:
                    raise RuntimeError("Камера перестала отвечать")
                continue

            consecutive_failures = 0
            frame_count += 1

            processed_frame, detections = model.process_frame(frame)
            detection_total += len(detections)

            # Считаем реальный FPS через время между кадрами
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            cv2.putText(
                processed_frame,
                f"FPS: {fps:.1f} | Detections: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if show_preview:
                cv2.imshow("License Plate Detection - Live", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Поток остановлен пользователем")
                    break

    except KeyboardInterrupt:
        logger.info("Поток прерван пользователем (Ctrl+C)")
    except Exception as e:
        logger.error(f"Ошибка во время обработки потока: {e}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()

    logger.info(
        f"Поток завершён. Кадров: {frame_count}, найдено номеров: {detection_total}"
    )


def _validate_threshold(value: str) -> float:
    """
    Валидатор для аргументов --confidence и --iou.
    Гарантирует что значение находится в диапазоне [0.0, 1.0].
    """
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(
            f"Значение должно быть между 0.0 и 1.0, получено: {v}"
        )
    return v


def main() -> None:
    """Точка входа CLI приложения."""
    parser = argparse.ArgumentParser(
        description="Система детекции номерных знаков на основе YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --video input.mp4 --output output.mp4
  python main.py --camera
  python main.py --video input.mp4 --model models/best.pt --output output.mp4
  python main.py --camera --confidence 0.6
        """,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--video", type=str, help="Путь к входному видеофайлу")
    mode_group.add_argument("--camera", action="store_true", help="Режим веб-камеры")

    parser.add_argument("--model", type=str, default=None,
                        help="Путь к файлу весов .pt")
    parser.add_argument("--confidence", type=_validate_threshold, default=0.5,
                        help="Порог уверенности (0.0-1.0, по умолчанию: 0.5)")
    parser.add_argument("--iou", type=_validate_threshold, default=0.45,
                        help="Порог IoU для NMS (0.0-1.0, по умолчанию: 0.45)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None,
                        help="Устройство вычислений: cuda или cpu")
    parser.add_argument("--output", type=str, default="output.mp4",
                        help="Путь к выходному файлу (по умолчанию: output.mp4)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Отключить окно предпросмотра")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="ID камеры (по умолчанию: 0)")

    args = parser.parse_args()

    # Логгер инициализируется один раз до создания модели
    setup_logger()

    model = My_LicensePlate_Model(
        model_path=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        device=args.device,
    )

    if args.video:
        process_video_file(
            model=model,
            video_path=args.video,
            output_path=args.output,
            show_preview=not args.no_preview,
        )
    elif args.camera:
        process_webcam_stream(
            model=model,
            camera_id=args.camera_id,
            show_preview=not args.no_preview,
        )


if __name__ == "__main__":
    main()