import os
import pickle
import re
from pathlib import Path

import pandas as pd

from xsam.logger import ActionLogger, FileLogger

# General logger for actions
action_logger = ActionLogger()

# Allow users to define a custom log file path via an environment variable
default_log_path = Path.home() / ".logs" / "file_log.log"
custom_log_path = Path(os.getenv("XSAM_LOG_PATH", default_log_path))
file_logger = FileLogger(custom_log_path)


def load(
    file_name: str = None,
    file_format: str = None,
    full_file_path: Path | str = None,
    log_id: str = None,
) -> pd.DataFrame | pd.Series | dict | None:
    """Load a DataFrame, Series, or dictionary from a file.

    Args:
        file_name (str): File name without extension.
        file_format (str): File format. Supported formats are 'csv', 'xlsx', and 'pickle'.
        full_file_path (Path | str): Full file path including the file name and extension.
        log_id (str): Unique file ID from the log file.

    Returns:
        pd.DataFrame | pd.Series | dict | None: Loaded object.

    Raises:
        ValueError: If the format is not supported.
        FileNotFoundError: If the specified path does not exist.
    """
    logs = file_logger.get_logs()
    log_entries = _parse_log_entries(logs)

    if full_file_path:
        return _load_from_full_path(full_file_path)
    elif file_name or file_format:
        return _load_from_name_and_format(log_entries, file_name, file_format)
    elif log_id:
        return _load_from_log_id(log_entries, log_id)
    else:
        raise ValueError("Insufficient parameters provided for loading the file.")


def _load_from_full_path(full_path: Path | str):
    path = Path(full_path)
    if path.exists():
        return _load_file(path)
    else:
        raise FileNotFoundError(f"The file {full_path} does not exist.")


def _load_from_name_and_format(log_entries, file_name, file_format):
    pattern = re.compile(file_name) if file_name else None
    extension = "p" if file_format == "pickle" else file_format
    matches = [
        entry
        for entry in log_entries
        if (not pattern or pattern.search(Path(entry["full_path"]).stem))
        and (not file_format or entry["file_format"] == extension)
    ]
    if matches:
        return _load_file(matches[-1]["full_path"])  # Return only the latest match
    else:
        raise FileNotFoundError("No matching file found in the log.")


def _load_from_log_id(log_entries, log_id):
    for entry in reversed(log_entries):
        if entry["log_id"] == log_id:
            return _load_file(entry["full_path"])
    raise FileNotFoundError(f"No file found with log_id {log_id}.")


def _load_file(full_path: Path | str):
    path = Path(full_path)  # Ensure full_path is a Path object
    extension = path.suffix[1:]  # Remove the leading dot
    file_format = "pickle" if extension == "p" else extension
    action_logger.info(f"Loading {file_format} from {path}")

    if file_format == "csv":
        return pd.read_csv(path)
    elif file_format == "xlsx":
        return _load_excel(path)
    elif file_format == "pickle":
        with path.open("rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {file_format}")


def _parse_log_entries(log_entries: list[str]) -> list[dict]:
    parsed_entries = []
    for entry in log_entries:
        log_id, timestamp, full_path = entry.strip().split(" | ")
        full_path = Path(full_path)
        file_format = full_path.suffix[1:]  # Remove the leading dot
        parsed_entries.append(
            {
                "log_id": log_id,
                "timestamp": timestamp,
                "full_path": str(full_path),
                "file_format": file_format,
            }
        )
    return parsed_entries


def _load_excel(full_path: Path) -> pd.DataFrame | dict:
    excel_data = pd.read_excel(full_path, sheet_name=None)
    action_logger.info(f"Excel file loaded from {full_path}")
    if len(excel_data) == 1:
        return next(iter(excel_data.values()))
    return excel_data
