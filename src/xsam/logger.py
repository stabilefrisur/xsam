import logging
import shutil
from datetime import datetime
from pathlib import Path

from xsam.constants import TIMESTAMP_FORMAT


def initialize_loggers():
    file_logger = get_file_logger("file_logger")
    action_logger = get_action_logger("action_logger")
    return file_logger, action_logger


def get_action_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Disable propagation
    return logger


def get_file_logger(name: str, file_path: str = "file_log.log") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.FileHandler(file_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Disable propagation
    return logger


def clear_nonexistent_entries(
    log_path: str = "file_log.log", action_logger: logging.Logger = None
) -> None:
    action_logger = (
        get_action_logger("action_logger") if action_logger is None else action_logger
    )
    log_path = Path(log_path)

    if not log_path.exists():
        action_logger.info(f"File log {log_path} does not exist.")
        return

    with log_path.open("r") as file:
        lines = file.readlines()

    with log_path.open("w") as file:
        for line in lines:
            entry = line.strip()
            _, _, file_path = entry.split(",")
            file_path = Path(file_path)
            if file_path.exists():
                file.write(line)
            else:
                action_logger.info(f"File log removed nonexistent entry: {entry}")


def backup_log_file(
    log_path: str = "file_log.log",
    archive_dir: str = "log_archive",
    action_logger: logging.Logger = None,
) -> None:
    if action_logger is None:
        action_logger = get_action_logger("action_logger")

    log_path = Path(log_path)
    archive_dir = Path(archive_dir)

    if not log_path.exists():
        action_logger.info(f"File log {log_path} does not exist.")
        return

    if not archive_dir.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    log_file_name = log_path.name
    backup_file_name = f"{timestamp}_{log_file_name}"
    backup_file_path = archive_dir / backup_file_name

    shutil.copy2(log_path, backup_file_path)
    action_logger.info(f"File log backup created at {backup_file_path}")


if __name__ == "__main__":
    # initialize_loggers()
    # backup_log_file()
    clear_nonexistent_entries()
