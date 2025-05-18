import pickle
import re
from enum import Enum
from pathlib import Path
from typing import Callable

import pandas as pd

from xsam.logger import get_action_logger, get_file_logger

# General logger for actions
action_logger = get_action_logger()


class ImportFormat(Enum):
    CSV = "csv"
    XLSX = "xlsx"
    PICKLE = "p"


def import_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def import_xlsx(path: Path) -> pd.DataFrame | dict:
    excel_data = pd.read_excel(path, sheet_name=None)
    action_logger.info(f"Excel file imported from {path}")
    if len(excel_data) == 1:
        return next(iter(excel_data.values()))
    return excel_data


def import_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


ImportFunction = Callable[[Path], pd.DataFrame | pd.Series | dict]

IMPORT_FUNCTIONS: dict[str, ImportFunction] = {
    ImportFormat.CSV.value: import_csv,
    ImportFormat.XLSX.value: import_xlsx,
    ImportFormat.PICKLE.value: import_pickle,
}


def get_importer(file_extension: str) -> ImportFunction:
    """Return the import function for the given file extension."""
    valid_extensions = {e.value for e in ImportFormat}
    if file_extension not in valid_extensions:
        raise ValueError(f"Unsupported file extension: {file_extension}. Supported extensions: {valid_extensions}")
    importer = IMPORT_FUNCTIONS.get(file_extension)
    if importer is None:
        raise ValueError(f"No import function registered for extension: {file_extension}")
    return importer


def import_obj(
    file_name: str = None,
    file_extension: str = None,
    file_path: Path | str = None,
    log_id: str = None,
) -> pd.DataFrame | pd.Series | dict | None:
    """Import a DataFrame, Series, or dictionary from a file.

    Args:
        file_name (str): File name without extension.
        file_extension (str): File extension. Supported extensions are 'csv', 'xlsx', and 'p'.
        full_file_path (Path | str): Full file path including the file name and extension.
        log_id (str): Unique file ID from the log file.

    Returns:
        pd.DataFrame | pd.Series | dict | None: Imported object.

    Raises:
        ValueError: If the extension is not supported.
        FileNotFoundError: If the specified path does not exist.
    """
    file_logger = get_file_logger()
    logs = file_logger.get_logs()
    log_entries = _parse_log_entries(logs)

    if file_path:
        return _import_from_full_path(file_path)
    elif file_name or file_extension:
        return _import_from_name_and_extension(log_entries, file_name, file_extension)
    elif log_id:
        return _import_from_log_id(log_entries, log_id)
    else:
        raise ValueError("Insufficient parameters provided for importing the file.")


def _import_from_full_path(full_path: Path | str):
    path = Path(full_path)
    if path.exists():
        return _import_file(path)
    else:
        raise FileNotFoundError(f"The file {full_path} does not exist.")


def _import_from_name_and_extension(log_entries, file_name, file_extension):
    pattern = re.compile(file_name) if file_name else None
    matches = [
        entry
        for entry in log_entries
        if (not pattern or pattern.search(Path(entry["full_path"]).stem))
        and (not file_extension or entry["file_extension"] == file_extension)
    ]
    if matches:
        return _import_file(matches[-1]["full_path"])  # Return only the latest match
    else:
        raise FileNotFoundError("No matching file found in the log.")


def _import_from_log_id(log_entries, log_id):
    for entry in reversed(log_entries):
        if entry["log_id"] == log_id:
            return _import_file(entry["full_path"])
    raise FileNotFoundError(f"No file found with log_id {log_id}.")


def _import_file(full_path: Path | str):
    path = Path(full_path)
    file_extension = path.suffix[1:]
    action_logger.info(f"Importing {file_extension} from {path}")
    importer = get_importer(file_extension)
    return importer(path)


def _parse_log_entries(log_entries: list[str]) -> list[dict]:
    parsed_entries = []
    for entry in log_entries:
        parts = entry.strip().split(" | ")
        if len(parts) != 3:
            continue  # skip malformed lines
        log_id, timestamp, full_path = parts
        parsed_entries.append(
            {
                "log_id": log_id,
                "timestamp": timestamp,
                "full_path": Path(full_path),
                "file_extension": Path(full_path).suffix[1:],
            }
        )
    return parsed_entries
