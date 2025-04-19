import logging
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from xsam.constants import TIMESTAMP_FORMAT, TIMESTAMP_FORMAT_LOG


class ActionLogger:
    """A class to log actions to standard output."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ActionLogger, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.logger = logging.getLogger("ActionLogger")
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            # Don't propagate to the root logger
            self.logger.propagate = False
            self.initialized = True

    def info(self, message: str):
        """Log an informational message.

        Args:
            message (str): Message to log.
        """
        self.logger.info(message)


class FileLogger:
    """A class to log file paths to a file and retrieve them by log ID or file name."""

    _instances = {}

    def __new__(cls, log_file: str, *args, **kwargs):
        if log_file not in cls._instances:
            cls._instances[log_file] = super(FileLogger, cls).__new__(cls)
        return cls._instances[log_file]

    def __init__(self, log_file: str):
        if not hasattr(self, "initialized"):
            self.log_file = Path(log_file).with_suffix(".log")
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.logger = logging.getLogger(f"FileLogger_{log_file}")
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.initialized = True

    def log_file_path(self, file_path: str):
        """Log the file path to the log file.

        Args:
            file_path (str): File path to log.
        """
        log_id = uuid.uuid4()
        timestamp = datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT_LOG)
        absolute_path = str(Path(file_path).resolve())
        message = f"{log_id} | {timestamp} | {absolute_path}"
        self.logger.info(message)

    def get_file_path(self, log_id: str = None, file_name: str = None) -> str:
        """Retrieve the file path by log ID or file name.

        Args:
            log_id (str, optional): Log ID. Defaults to None.
            file_name (str, optional): File name. Defaults to None. If file_name is provided, it will return the last exact match or pattern match.

        Returns:
            str: File path or message.
        """
        assert log_id or file_name, "Either log_id or file_name must be provided."

        if not self.log_file.exists():
            return "Log file does not exist."

        with self.log_file.open("r") as f:
            lines = f.readlines()

        if log_id:
            for line in lines:
                log_id_record, timestamp, path = line.strip().split(" | ")
                if log_id_record == log_id:
                    return path
            return "Log ID not found."

        if file_name:
            last_exact_match = None
            last_pattern_match = None
            for line in lines:
                log_id_record, timestamp, path = line.strip().split(" | ")
                if Path(path).name == file_name:
                    last_exact_match = path
                elif re.search(file_name, Path(path).name):
                    last_pattern_match = path

            if last_exact_match:
                return last_exact_match
            if last_pattern_match:
                return last_pattern_match
            return "File name not found."

        return "Either log_id or file_name must be provided."

    def get_logs(self) -> list[str]:
        """Retrieve all logs."""
        if not self.log_file.exists():
            return []
        with self.log_file.open("r") as f:
            return f.readlines()

    def search_logs(self, keyword: str) -> list:
        """Search logs for a specific keyword.

        Args:
            keyword (str): Keyword to search for in the logs.

        Returns:
            list: List of log entries containing the keyword.
        """
        if not self.log_file.exists():
            return []

        with self.log_file.open("r") as f:
            return [line for line in f if keyword in line]

    def backup_logs(self, backup_dir: str):
        """Backup the current log file to the specified directory.

        Args:
            backup_dir (str): Directory to backup the log file.
        """
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)
        backup_file = backup_path / f"{self.log_file.stem}_{timestamp}.log"
        shutil.copy(str(self.log_file), backup_file)

    def clean_logs(self):
        """Clean the log file of entries where the file path no longer exists."""
        if not self.log_file.exists():
            return

        with self.log_file.open("r") as f:
            lines = f.readlines()

        valid_lines = []
        for line in lines:
            log_id_record, timestamp, path = line.strip().split(" | ")
            if Path(path).exists():
                valid_lines.append(line)

        with self.log_file.open("w") as f:
            f.writelines(valid_lines)


def set_log_path(new_path: Path | str):
    """Set a custom path for the log file."""
    global file_logger
    file_logger = FileLogger(Path(new_path))
    # General logger for actions
    action_logger = ActionLogger()
    action_logger.info(f"Log file path updated to {new_path}")
