import logging
import os
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

    def warning(self, message: str):
        """Log a warning message.

        Args:
            message (str): Message to log.
        """
        self.logger.warning(message)


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

    def get_logs(self) -> list[str]:
        """Retrieve all logs as lines without trailing newlines."""
        if not self.log_file.exists():
            return []
        with self.log_file.open("r") as f:
            return [line.rstrip("\n") for line in f]

    def search_logs(self, keyword: str, return_type: str = "log_entry") -> list[str]:
        """Search logs for a specific keyword.

        Args:
            keyword (str): Keyword to search for in the logs.

        Returns:
            list: List of log entries, log IDs, timestamps, or paths containing the keyword.
        """
        assert return_type in ["log_entry", "log_id", "timestamp", "path"], (
            "Invalid return type. Must be 'log_entry', 'log_id', 'timestamp', or 'path'."
        )

        log_entries = [line for line in self.get_logs() if keyword in line]
        if not log_entries:
            return []

        if return_type == "log_entry":
            return log_entries
        elif return_type == "log_id":
            return [line.split(" | ")[0] for line in log_entries]
        elif return_type == "timestamp":
            return [line.split(" | ")[1] for line in log_entries]
        elif return_type == "path":
            return [Path(line.split(" | ")[2]) for line in log_entries]

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
        """Clean the log file of entries where the file path no longer exists, and remove duplicate file log entries (keep only the latest for each path)."""
        if not self.log_file.exists():
            return

        with self.log_file.open("r") as f:
            lines = f.readlines()

        # Keep only the latest entry for each path
        path_to_line = {}
        for line in lines:
            log_id_record, timestamp, path = line.strip().split(" | ")
            if Path(path).exists():
                path_to_line[path] = line  # Overwrite to keep the latest

        valid_lines = list(path_to_line.values())

        with self.log_file.open("w") as f:
            f.writelines(valid_lines)


file_logger = None


def get_file_logger():
    """Retrieve the current FileLogger instance."""
    # Singleton pattern to ensure only one instance of FileLogger exists
    global file_logger
    if file_logger is None:
        default_log_path = Path.home() / ".logs" / "file_log.log"
        custom_log_path = Path(os.getenv("XSAM_LOG_PATH", default_log_path))
        file_logger = FileLogger(custom_log_path)
    return file_logger


def set_log_path(new_path: Path | str):
    """Set a custom path for the log file."""
    global file_logger
    file_logger = FileLogger(Path(new_path))
    # General logger for actions
    action_logger = ActionLogger()
    action_logger.info(f"Log file path updated to {new_path}")
