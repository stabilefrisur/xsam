import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from xsam.constants import TIMESTAMP_FORMAT, TIMESTAMP_FORMAT_LOG, LOG_DELIMITER


class ActionLogger:
    """A singleton class to log actions to standard output and to a log file using standard logging."""

    _instance = None
    log_file = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ActionLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_file: str):
        if not hasattr(self, "initialized"):
            self.logger = logging.getLogger("ActionLogger")
            self.logger.setLevel(logging.INFO)
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
            self.logger.propagate = False
            self.log_file = Path(log_file).with_suffix(".log")
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.initialized = True

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def get_logs(self) -> list[str]:
        """Retrieve all logs as lines without trailing newlines."""
        if not self.log_file.exists():
            return []
        with self.log_file.open("r") as f:
            return [line.rstrip("\n") for line in f]


class FileLogger:
    """A singleton class to log file paths to a log file and retrieve them by log ID or file name."""

    _instance = None
    log_file = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FileLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_file: str):
        if not hasattr(self, "initialized"):
            self.logger = logging.getLogger("FileLogger")
            self.logger.setLevel(logging.INFO)
            self.log_file = Path(log_file).with_suffix(".log")
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.initialized = True

    def log_a_file(self, file_path: str) -> str:
        """Log the file path to the log file.

        Args:
            file_path (str): File path to log.
        """
        log_id = uuid.uuid4()
        timestamp = datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT_LOG)
        absolute_path = str(Path(file_path).resolve())
        log_entry = LOG_DELIMITER.join([str(log_id), timestamp, absolute_path])
        self.logger.info(log_entry)
        return log_entry

    def get_logs(self) -> list[str]:
        """Retrieve all logs as lines without trailing newlines."""
        if not self.log_file.exists():
            return []
        with self.log_file.open("r") as f:
            return [line.rstrip("\n") for line in f]

    def search_logs(self, keywords: str | list[str], return_type: str = "log_entry") -> list[str]:
        """Search logs for one or more keywords. Only entries containing all keywords are returned.

        Args:
            keyword (str | list[str]): Keyword or list of keywords to search for in the logs.

        Returns:
            list: List of log entries, log IDs, timestamps, or paths containing all of the keywords.
        """
        assert return_type in ["log_entry", "log_id", "timestamp", "path"], (
            "Invalid return type. Must be 'log_entry', 'log_id', 'timestamp', or 'path'."
        )

        keywords = [keywords] if isinstance(keywords, str) else keywords

        log_entries = [
            line for line in self.get_logs()
            if all(k in line for k in keywords)
        ]
        if not log_entries:
            return []

        if return_type == "log_entry":
            return log_entries
        elif return_type == "log_id":
            return [line.split(LOG_DELIMITER)[0] for line in log_entries]
        elif return_type == "timestamp":
            return [line.split(LOG_DELIMITER)[1] for line in log_entries]
        elif return_type == "path":
            return [Path(line.split(LOG_DELIMITER)[2]) for line in log_entries]

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
            log_id_record, timestamp, path = line.strip().split(LOG_DELIMITER)
            if Path(path).exists():
                path_to_line[path] = line  # Overwrite to keep the latest

        valid_lines = list(path_to_line.values())

        with self.log_file.open("w") as f:
            f.writelines(valid_lines)


_action_logger = None


def get_action_logger() -> ActionLogger:
    """Retrieve the current ActionLogger instance."""
    global _action_logger
    if _action_logger is None:
        default_log_path = Path.home() / ".logs" / "action_log.log"
        custom_log_path = Path(os.getenv("XSAM_LOG_PATH", default_log_path))
        _action_logger = ActionLogger(log_file=custom_log_path)
    return _action_logger


_file_logger = None


def get_file_logger() -> FileLogger:
    """Retrieve the current FileLogger instance."""
    global _file_logger
    if _file_logger is None:
        default_log_path = Path.home() / ".logs" / "file_log.log"
        custom_log_path = Path(os.getenv("XSAM_LOG_PATH", default_log_path))
        _file_logger = FileLogger(log_file=custom_log_path)
    return _file_logger


def set_log_path(new_dir: Path | str):
    global _file_logger, _action_logger
    # Always reset the singletons and their class-level _instance attributes
    if _file_logger is not None:
        _file_logger.logger.handlers.clear()
        _file_logger.initialized = False
    if _action_logger is not None:
        _action_logger.logger.handlers.clear()
        _action_logger.initialized = False
    _file_logger = None
    _action_logger = None
    FileLogger._instance = None
    ActionLogger._instance = None
    # Now re-instantiate
    new_dir = Path(new_dir)
    new_dir.mkdir(parents=True, exist_ok=True)
    action_log_path = new_dir / "action_log.log"
    file_log_path = new_dir / "file_log.log"
    _action_logger = ActionLogger(log_file=action_log_path)
    _file_logger = FileLogger(log_file=file_log_path)


if __name__ == "__main__":
    # Example usage
    # set_log_path("logs")

    action_logger = get_action_logger()
    action_logger.info("This is an info message.")
    action_logger.warning("This is a warning message.")
    action_logger.error("This is an error message.")
    action_logger.critical("This is a critical message.")
    action_logger.debug("This is a debug message.")
    print(action_logger.get_logs())
    print(action_logger.log_file)

    file_logger = get_file_logger()
    file_logger.log_a_file("example.txt")
    print(file_logger.get_logs())
    print(file_logger.log_file)
