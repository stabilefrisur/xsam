import functools
import pickle
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd

from xsam.constants import TIMESTAMP_FORMAT
from xsam.logger import ActionLogger, get_file_logger
from xsam.utilities import flatten_dict

# General logger for actions
action_logger = ActionLogger()


class ExportFormat(Enum):
    CSV = "csv"
    XLSX = "xlsx"
    PICKLE = "p"
    PNG = "png"
    SVG = "svg"


ExportFunction = Callable[[pd.DataFrame | pd.Series | dict | plt.Figure, Path], None]


def get_exporter(file_extension: str) -> ExportFunction:
    """Get the appropriate export function based on the file extension.

    Args:
        file_extension (str): File extension to determine the export function.
        Supported extensions are 'csv', 'xlsx', 'p', 'png', and 'svg'.

    Raises:
        ValueError: If the file extension is not supported.
        ValueError: If no export function is registered for the extension.

    Returns:
        ExportFunction: The export function corresponding to the file extension.
    """
    # Validate against ExportFormat Enum for typo safety
    valid_extensions = {e.value for e in ExportFormat}
    if file_extension not in valid_extensions:
        raise ValueError(f"Unsupported file extension: {file_extension}. Supported extensions: {valid_extensions}")
    exporter = EXPORT_FUNCTIONS.get(file_extension)
    if exporter is None:
        raise ValueError(f"No export function registered for extension: {file_extension}")
    return exporter


def export_obj(
    obj: pd.DataFrame | pd.Series | dict | plt.Figure,
    file_name: str,
    file_extension: str = ExportFormat.PICKLE.value,
    file_path: Path | str = Path.home() / "output",
    add_timestamp: bool = True,
) -> None:
    """Export a DataFrame, Series, dictionary, or Figure to a file.

    This function will attempt to export the object to the specified directory. If the export fails due to a permission or OS error, it will retry in the system's temporary directory and log a warning.

    Args:
        obj (pd.DataFrame | pd.Series | dict | plt.Figure): Object to export.
        file_name (str): File name without extension.
        file_extension (str): File extension. Supported extensions are 'csv', 'xlsx', 'p', 'png', and 'svg'. Default is 'p'.
        file_path (Path | str): Directory path where the file will be exported. Default is 'output'.
        add_timestamp (bool): Whether to add a timestamp to the file name. Default is True.

    Raises:
        ValueError: If the extension is not supported.
    """
    path = Path(file_path)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    file_name = f"{timestamp}_{file_name}" if add_timestamp else file_name
    full_path = path / f"{file_name}.{file_extension}"

    export_function = get_exporter(file_extension)
    file_logger = get_file_logger()

    try:
        export_function(obj, full_path)
        file_logger.log_file_path(str(full_path))
    except (PermissionError, OSError) as e:
        # Handle errors by exporting to a temporary directory
        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / f"{file_name}.{file_extension}"
        action_logger.warning(
            f"Failed to export file to {full_path} due to {type(e).__name__}: {e}. "
            f"Exporting to temporary directory: {temp_path}"
        )
        export_function(obj, temp_path)
        file_logger.log_file_path(str(temp_path))


def export_csv(obj, full_path: Path) -> None:
    if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        obj.to_csv(full_path, index=True, encoding="utf-8")
        action_logger.info(f"CSV exported to {full_path}")
    elif isinstance(obj, dict):
        flattened_dict = flatten_dict(obj)
        for key, value in flattened_dict.items():
            if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                key_file_path = full_path.with_name(f"{full_path.stem}_{key.replace('.', '_')}{full_path.suffix}")
                value.to_csv(key_file_path, index=True, encoding="utf-8")
                action_logger.info(f"CSV exported to {key_file_path}")
            else:
                raise TypeError("CSV format is only supported for DataFrame or Series in dictionary values")
    else:
        raise TypeError("CSV format is only supported for DataFrame, Series, or dict")


def export_xlsx(obj, full_path: Path) -> None:
    if isinstance(obj, pd.DataFrame):
        obj.to_excel(full_path, index=True)
    elif isinstance(obj, pd.Series):
        obj.to_frame().to_excel(full_path, index=True)
    elif isinstance(obj, dict):
        flattened_dict = flatten_dict(obj)
        with pd.ExcelWriter(full_path) as writer:
            for key, value in flattened_dict.items():
                if isinstance(value, pd.DataFrame):
                    value.to_excel(
                        writer, sheet_name=key[:31], index=True
                    )  # Excel sheet names are limited to 31 characters
                elif isinstance(value, pd.Series):
                    value.to_frame().to_excel(writer, sheet_name=key[:31], index=True)
                else:
                    raise TypeError("Unsupported type in dictionary for xlsx format")
    else:
        raise TypeError("XLSX format is only supported for DataFrame, Series, or dict")

    action_logger.info(f"XLSX exported to {full_path}")


def export_pickle(obj, full_path: Path) -> None:
    with open(full_path, "wb") as f:
        pickle.dump(obj, f)

    action_logger.info(f"Pickle exported to {full_path}")


def export_figure(obj, full_path: Path, format: str) -> None:
    """Generic function to export matplotlib figures in a given format."""
    if isinstance(obj, plt.Figure):
        obj.savefig(full_path, format=format)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, plt.Figure):
                key_file_path = full_path.with_name(f"{full_path.stem}_{key.replace('.', '_')}{full_path.suffix}")
                value.savefig(key_file_path, format=format)
                action_logger.info(f"{format.upper()} exported to {key_file_path}")
            else:
                raise TypeError(f"{format.upper()} format is only supported for Figure in dictionary values")
    else:
        raise TypeError(f"{format.upper()} format is only supported for Figure")

    action_logger.info(f"{format.upper()} exported to {full_path}")


def export_png(obj, full_path: Path) -> None:
    export_figure(obj, full_path, "png")


def export_svg(obj, full_path: Path) -> None:
    export_figure(obj, full_path, "svg")


EXPORT_FUNCTIONS: dict[str, ExportFunction] = {
    ExportFormat.CSV.value: export_csv,
    ExportFormat.XLSX.value: export_xlsx,
    ExportFormat.PICKLE.value: export_pickle,
    ExportFormat.PNG.value: export_png,
    ExportFormat.SVG.value: export_svg,
}


def export_decorator(
    export: bool = True,
    file_name: str = "output",
    file_extension: str = "p",
    file_path: Path | str = Path("output"),
    add_timestamp: bool = True,
):
    """Decorator to export the output of a function."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if export:
                export(
                    result,
                    file_name=kwargs.get("file_name", file_name),
                    file_extension=kwargs.get("file_extension", file_extension),
                    file_path=kwargs.get("file_path", file_path),
                    add_timestamp=kwargs.get("add_timestamp", add_timestamp),
                )
            return result

        return wrapper

    return decorator
