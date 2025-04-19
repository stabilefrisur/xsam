import functools
import os
import pickle
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from xsam.constants import TIMESTAMP_FORMAT
from xsam.logger import ActionLogger, FileLogger
from xsam.utilities import flatten_dict

# General logger for actions
action_logger = ActionLogger()

# Allow users to define a custom log file path via an environment variable
default_log_path = Path.home() / ".logs" / "file_log.log"
custom_log_path = Path(os.getenv("XSAM_LOG_PATH", default_log_path))
file_logger = FileLogger(custom_log_path)


def save(
    obj: pd.DataFrame | pd.Series | dict | plt.Figure,
    file_name: str,
    file_format: str = "pickle",
    file_path: Path | str = Path.home() / "output",
    add_timestamp: bool = True,
) -> None:
    """Save a DataFrame, Series, dictionary, or Figure to a file.

    Args:
        obj (pd.DataFrame | pd.Series | dict | plt.Figure): Object to save.
        file_name (str): File name without extension.
        file_format (str): File format. Supported formats are 'csv', 'xlsx', 'pickle', 'png', and 'svg'. Default is 'pickle'.
        file_path (Path | str): Directory path where the file will be saved. Default is 'output'.
        add_timestamp (bool): Whether to add a timestamp to the file name. Default is True.

    Raises:
        ValueError: If the format is not supported.
    """
    path = Path(file_path)
    path.mkdir(parents=True, exist_ok=True)  # Ensure the path exists
    extension = "p" if file_format == "pickle" else file_format
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    file_name = f"{timestamp}_{file_name}" if add_timestamp else file_name
    full_path = path / f"{file_name}.{extension}"

    file_logger.log_file_path(full_path)
    # action_logger.info(f"Saving {type(obj).__name__} as {file_format} to {full_path}")

    # Map file formats to their respective save functions
    save_functions = {
        "csv": save_csv,
        "xlsx": save_xlsx,
        "p": save_pickle,
        "png": save_png,
        "svg": save_svg,
    }

    save_function = save_functions.get(extension)
    if save_function is None:
        raise ValueError(f"Unsupported file format: {file_format}")

    try:
        save_function(obj, full_path)
    except (PermissionError, OSError) as e:
        # Handle errors by saving to a temporary directory
        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / f"{file_name}.{extension}"
        action_logger.warning(
            f"Failed to save file to {full_path} due to {type(e).__name__}: {e}. "
            f"Saving to temporary directory: {temp_path}"
        )
        save_function(obj, temp_path)


def save_csv(obj, full_path: Path) -> None:
    if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        obj.to_csv(full_path, index=False, encoding="utf-8")
        action_logger.info(f"CSV saved to {full_path}")
    elif isinstance(obj, dict):
        flattened_dict = flatten_dict(obj)
        for key, value in flattened_dict.items():
            if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                key_file_path = full_path.with_name(f"{full_path.stem}_{key.replace('.', '_')}{full_path.suffix}")
                value.to_csv(key_file_path, index=False, encoding="utf-8")
                action_logger.info(f"CSV saved to {key_file_path}")
            else:
                raise TypeError("CSV format is only supported for DataFrame or Series in dictionary values")
    else:
        raise TypeError("CSV format is only supported for DataFrame, Series, or dict")


def save_xlsx(obj, full_path: Path) -> None:
    if isinstance(obj, pd.DataFrame):
        obj.to_excel(full_path, index=False)
    elif isinstance(obj, pd.Series):
        obj.to_frame().to_excel(full_path, index=False)
    elif isinstance(obj, dict):
        flattened_dict = flatten_dict(obj)
        with pd.ExcelWriter(full_path) as writer:
            for key, value in flattened_dict.items():
                if isinstance(value, pd.DataFrame):
                    value.to_excel(
                        writer, sheet_name=key[:31], index=False
                    )  # Excel sheet names are limited to 31 characters
                elif isinstance(value, pd.Series):
                    value.to_frame().to_excel(writer, sheet_name=key[:31], index=False)
                else:
                    raise TypeError("Unsupported type in dictionary for xlsx format")
    else:
        raise TypeError("XLSX format is only supported for DataFrame, Series, or dict")

    action_logger.info(f"XLSX saved to {full_path}")


def save_pickle(obj, full_path: Path) -> None:
    with open(full_path, "wb") as f:
        pickle.dump(obj, f)

    action_logger.info(f"Pickle saved to {full_path}")


def save_figure(obj, full_path: Path, format: str) -> None:
    """Generic function to save matplotlib figures in a given format."""
    if isinstance(obj, plt.Figure):
        obj.savefig(full_path, format=format)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, plt.Figure):
                key_file_path = full_path.with_name(f"{full_path.stem}_{key.replace('.', '_')}{full_path.suffix}")
                value.savefig(key_file_path, format=format)
                action_logger.info(f"{format.upper()} saved to {key_file_path}")
            else:
                raise TypeError(f"{format.upper()} format is only supported for Figure in dictionary values")
    else:
        raise TypeError(f"{format.upper()} format is only supported for Figure")

    action_logger.info(f"{format.upper()} saved to {full_path}")


def save_png(obj, full_path: Path) -> None:
    save_figure(obj, full_path, "png")


def save_svg(obj, full_path: Path) -> None:
    save_figure(obj, full_path, "svg")


def save_decorator(
    save: bool = True,
    file_name: str = "output",
    file_format: str = "pickle",
    file_path: Path | str = Path("output"),
    add_timestamp: bool = True,
):
    """Decorator to save the output of a function."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if save:
                save(
                    result,
                    file_name=kwargs.get("file_name", file_name),
                    file_format=kwargs.get("file_format", file_format),
                    file_path=kwargs.get("file_path", file_path),
                    add_timestamp=kwargs.get("add_timestamp", add_timestamp),
                )
            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    import pandas as pd

    data = {
        "A": [1, 2, 3],
        "B": [4, 5, 6],
    }
    df = pd.DataFrame(data)

    save(df, "data", "csv", add_timestamp=False)
    save(df, "data", "xlsx", add_timestamp=False)
    save(df, "data", "pickle", add_timestamp=False)

    print(file_logger.log_file)
    from xsam.logger import set_log_path
    set_log_path(file_logger.log_file.parent / "custom_log.log")
    save(df, "data", "csv", add_timestamp=False)
    save(df, "data", "xlsx", add_timestamp=False)

