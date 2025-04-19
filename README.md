# XSAM

XSAM is a Python package designed to assist with management and analysis of financial data.

## Installation

To install the package, use the following command:

```sh
pip install xsam
```

If you want to run the package's tests, install dev dependencies for pytest
```sh
pip install xsam[dev]
pytest
```

## Usage

### Customizing the Log File Location

By default, the `xsam` package writes logs to `file_log.log` in the user's home directory. To specify a custom location for the log file, you can either set the `XSAM_LOG_PATH` environment variable or configure it programmatically using the `set_log_path` function.

#### Example (Programmatically):
```python
from pathlib import Path
from xsam.logger import set_log_path

# Set a custom log file path
custom_log_path = Path("C:/path/to/custom/file_log.log")
set_log_path(custom_log_path)
```

#### Example (Linux/Mac):
```bash
export XSAM_LOG_PATH=/path/to/custom/file_log.log
```

#### Example (Windows, PowerShell):
```bash
$env:XSAM_LOG_PATH = "C:\path\to\custom\file_log.log"
```

#### Example (Windows, Command Prompt):
```bash
set XSAM_LOG_PATH=C:\path\to\custom\file_log.log
```

### Entry Point
Run the main program like so

```sh
xsam
```

Or run the main module

```sh
python -m xsam.main
```

### Saving Data
You can save a DataFrame, Series, dictionary, or Figure to a file using the `save` function:
```python
from pathlib import Path
import pandas as pd
from xsam.output import save

# Create a DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# Save the DataFrame to a CSV filev
save(df, 'data', 'csv', Path('data'))
```

#### Additional Examples
1. **Save a Series to a Pickle File:**
    ```python
    series = pd.Series([1, 2, 3, 4, 5])
    save(series, 'series_data', 'pickle', Path('data'))
    ```

2. **Save a Dictionary of DataFrames to an Excel File:**
    ```python
    data_dict = {
        'Sheet1': pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
        'Sheet2': pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
    }
    save(data_dict, 'data_dict', 'xlsx', Path('data'))
    ```

3. **Save a Matplotlib Figure to a SVG File:**
    ```python
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    save(fig, 'figure', 'svg', Path('data'))
    ```

### Loading Data
You can load a DataFrame, Series, or dictionary from a file using the `load` function:
```python
from xsam.input import load

# Load the latest file in the log
loaded_df = load(file_name='data')

# Print the loaded DataFrame
print(loaded_df)
```

#### Arguments for `load`
- `file_name` (str): The name of the file to load. If not provided, the function will search the log for the latest file.
- `file_format` (str): The format of the file to load. Supported formats are 'csv', 'xlsx', and 'pickle'.
- `full_file_path` (Path | str): The path to the file. If not provided, the function will search the log for the latest file.
- `log_id` (str): The unique file ID from the log file. If provided, the function will use this to locate the file.

#### Examples
1. **Load by Name:**
    ```python
    loaded_df = load(file_name='data')
    ```

2. **Load by Name and Format:**
    ```python
    loaded_df = load(file_name='data', file_format='csv')
    ```

3. **Load by Full File Path:**
    ```python
    loaded_df = load(full_file_path='data/data.csv')
    ```

4. **Load by Log ID:**
    ```python
    loaded_df = load(log_id='unique-log-id')
    ```

### Aggregating Fields by Label

The `aggregate_fields_by_label` function aggregates values in a DataFrame by label, with optional regex matching and multipliers.

#### Parameters

- `df` (pd.DataFrame): DataFrame containing the data to aggregate.
- `id_column` (str): Column containing IDs to be associated with groups.
- `weight_column` (str, optional): Column to use for weighted averages. Defaults to None.
- `field_columns` (list[str], optional): List of field columns to be aggregated. Defaults to None.
- `label_column` (str, optional): Column containing labels to use for grouping. Defaults to None.
- `label_regex` (dict, optional): Dictionary of regex patterns and multipliers to use for grouping. Defaults to None.
- `method` (str, optional): Method to use for aggregation. Options are "sum", "wsum", "avg", "wavg". Defaults to "sum".
- `preliminary` (bool, optional): Whether to return the DataFrame before summing the values for each group. Defaults to False.

#### Returns

- `pd.DataFrame`: DataFrame with aggregated values for each group.

#### Example

```python
import pandas as pd
from xsam.aggregation import aggregate_fields_by_label

data = {
    "id": ["A0", "B0", "C0", "A1", "B1", "C1", "A2", "B2", "C2"],
    "value1": [10, 20, 30, 40, 50, 60, 70, 80, 90],
    "value2": [15, 25, 35, 45, 55, 65, 75, 85, 95],
    "weight": [1, 2, 3, 4, 5, 6, 7, 8, 9],
}
df = pd.DataFrame(data)
regex_dict = {
    "A_group": {"long": ("A.*", 1), "short": ("A1", -1), "other": ("A1", 1)},
    "B_group": {"long": ("B.*", 1), "short": ("B1", -1)},
    "C_group": {"long": ("C.*", 1), "short": ("C1", -1)},
}
result = aggregate_fields_by_label(df, "id", "weight", ["value1", "value2"], label_regex=regex_dict, method="wavg")
print(result)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors

- Stabile Frisur - [stabilefrisur@proton.me](mailto:stabilefrisur@proton.me)

## Contributing

No need to contribute at this point. Thank you!
