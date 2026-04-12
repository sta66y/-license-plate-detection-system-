import logging
import os
from pathlib import Path


def setup_logger(log_dir: str = "./data") -> logging.Logger:
    """
    Настраивает и возвращает логгер (singleton паттерн).
    Пишет одновременно в файл и консоль.
    """
    logger = logging.getLogger("license_plate_detector")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    log_path = Path(os.environ.get("LOG_DIR", log_dir))
    log_path.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_path / "log_file.log")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
