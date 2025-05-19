import pickle
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


def import_pickle(path: Path) -> any:
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
    file_name: str | list[str] = None,
    file_extension: str = None,
    file_path: Path | str = None,
    log_id: str = None,
) -> pd.DataFrame | pd.Series | dict | None:
    """Import a DataFrame, Series, or dictionary from a file.

    Args:
        file_name (str): File name without extension.
        file_extension (str): File extension. Supported extensions are 'csv', 'xlsx', and 'p'.
        file_path (Path | str): Full file path including the file name and extension.
        log_id (str): Unique file ID from the log file.

    Returns:
        pd.DataFrame | pd.Series | dict | None: Imported object.

    Raises:
        ValueError: If the extension is not supported.
        FileNotFoundError: If the specified path does not exist.
    """
    file_logger = get_file_logger()
    if file_path:
        return _import_from_full_path(file_path)
    log_entry = _find_log_entry(file_logger, file_name, file_extension, log_id)
    if log_entry:
        return _import_file(log_entry)
    raise FileNotFoundError("No matching file found in the log.")


def _import_from_full_path(full_path: Path | str) -> pd.DataFrame | pd.Series | dict:
    """Import a file from the given full path."""
    path = Path(full_path)
    if path.exists():
        return _import_file(path)
    else:
        raise FileNotFoundError(f"The file {full_path} does not exist.")


def _find_log_entry(file_logger, file_name, file_extension, log_id) -> str | None:
    """Find the log entry for the given file name, extension, or log ID."""
    # Search by log_id
    if log_id:
        log_entries = file_logger.search_logs(log_id, return_type="path")
        if log_entries:
            return log_entries[-1]
        return None
    # Search by file_name and/or extension
    keywords = []
    if file_name:
        file_name = [file_name] if isinstance(file_name, str) else file_name
        keywords.extend(file_name)
    if file_extension:
        keywords.append(file_extension)
    log_entries = file_logger.search_logs(keywords, return_type="path") if keywords else []
    if log_entries:
        return log_entries[-1]
    return None


def _import_file(full_path: Path | str) -> pd.DataFrame | pd.Series | dict:
    """Import a file from the given path."""
    path = Path(full_path)
    file_extension = path.suffix[1:]
    action_logger.info(f"Importing {file_extension} from {path}")
    importer = get_importer(file_extension)
    return importer(path)
