import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from xsam.constants import TIMESTAMP_FORMAT
from xsam.logger import ActionLogger, FileLogger

# Initialize loggers
action_logger = ActionLogger()
file_logger = FileLogger("file_log.log")

def save(
    obj: pd.DataFrame | pd.Series | dict | plt.Figure,
    file_name: str,
    file_format: str = "pickle",
    file_path: Path | str = Path("output"),
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
        TypeError: If the object type is not supported.
        ValueError: If the format is not supported.
    """
    path = Path(file_path)
    path.mkdir(parents=True, exist_ok=True)  # Ensure the path exists
    extension = "p" if file_format == "pickle" else file_format
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    file_name = f"{timestamp}_{file_name}" if add_timestamp else file_name
    full_path = path / f"{file_name}.{extension}"

    file_logger.log_file_path(full_path)
    action_logger.info(f"Saving {type(obj).__name__} as {file_format} to {full_path}")

    if isinstance(obj, pd.DataFrame):
        _save_dataframe(obj, full_path, file_format)
    elif isinstance(obj, pd.Series):
        _save_series(obj, full_path, file_format)
    elif isinstance(obj, dict):
        _save_dict(obj, full_path)
    elif isinstance(obj, plt.Figure):
        _save_figure(obj, full_path, file_format)
    else:
        raise TypeError("Unsupported object type")


def _save_dataframe(df: pd.DataFrame, full_path: Path, file_format: str) -> None:
    if file_format == "csv":
        df.to_csv(full_path, index=False)
    elif file_format == "xlsx":
        df.to_excel(full_path, index=False)
    elif file_format == "pickle":
        with open(full_path, "wb") as f:
            pickle.dump(df, f)
    else:
        raise ValueError(f"Unsupported format for DataFrame: {file_format}")

    action_logger.info(f"DataFrame saved to {full_path}")


def _save_series(series: pd.Series, full_path: Path, file_format: str) -> None:
    if file_format == "csv":
        series.to_csv(full_path, index=False)
    elif file_format == "xlsx":
        series.to_frame().to_excel(full_path, index=False)
    elif file_format == "pickle":
        with open(full_path, "wb") as f:
            pickle.dump(series, f)
    else:
        raise ValueError(f"Unsupported format for Series: {file_format}")

    action_logger.info(f"Series saved to {full_path}")


def _save_dict(d: dict, full_path: Path) -> None:
    if all(isinstance(v, (pd.DataFrame, pd.Series, dict)) for v in d.values()):
        with pd.ExcelWriter(full_path) as writer:
            for key, value in d.items():
                if isinstance(value, pd.DataFrame):
                    value.to_excel(writer, sheet_name=key, index=False)
                elif isinstance(value, pd.Series):
                    value.to_frame().to_excel(writer, sheet_name=key, index=False)
                elif isinstance(value, dict):
                    nested_path = full_path.parent / f"{key}.xlsx"
                    _save_dict(value, nested_path)
    else:
        raise ValueError(
            "All values in the dictionary must be DataFrame, Series, or nested dictionaries"
        )

    action_logger.info(f"Dictionary saved to {full_path}")


def _save_figure(fig: plt.Figure, full_path: Path, file_format: str) -> None:
    if file_format in ["png", "svg"]:
        fig.savefig(full_path)
    else:
        raise ValueError(f"Unsupported format for Figure: {file_format}")

    action_logger.info(f"Figure saved to {full_path}")


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
