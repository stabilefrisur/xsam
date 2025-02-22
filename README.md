
# XSAM

XSAM is a Python package designed to assist with data aggregation tasks. This README provides an overview of the `aggregation.py` module, which contains functions for aggregating data in a pandas DataFrame.

## Installation

To install the package, use the following command:

```sh
pip install xsam
```

## Usage

### aggregate_fields_by_label

The `aggregate_fields_by_label` function aggregates values in a DataFrame by label, with optional regex matching and multipliers.

#### Parameters

- `df` (pd.DataFrame): DataFrame containing the data to aggregate.
- `id_column` (str): Column containing IDs to be associated with groups.
- `weight_column` (str): Column to use for weighted averages.
- `field_columns` (list[str]): List of field columns to be aggregated as weighted average by group.
- `label_column` (str, optional): Column containing labels to use for grouping. Defaults to None.
- `label_regex` (dict, optional): Dictionary of regex patterns and multipliers to use for grouping. Defaults to None.

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
result = aggregate_fields_by_label(df, "id", "weight", ["value1", "value2"], label_regex=regex_dict)
print(result)
```

## Testing

To run the tests, use the following command:

```sh
pytest
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors

- Stabile Frisur - [stabilefrisur@proton.me](mailto:stabilefrisur@proton.me)

## Contributing

No need to contribute at this point. Thank you!
