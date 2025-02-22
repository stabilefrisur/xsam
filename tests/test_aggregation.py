import pandas as pd
from xsam.aggregation import aggregate_fields_by_label

def test_aggregate_fields_by_label_with_regex() -> None:
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

    test_result = pd.DataFrame(
        {
            "group": ["A_group", "B_group", "C_group"],
            "aggregated_value1": [55.0, 12.0, 10.0],
            "aggregated_value2": [60.0, 12.0, 10.0],
        }
    )
    assert result.equals(test_result), f"Expected {test_result}, but got {result}"

def test_aggregate_fields_by_label_with_label() -> None:
    data = {
        "id": ["A0", "B0", "C0", "A1", "B1", "C1", "A2", "B2", "C2"],
        "label": ["foo", "foo", "bar", "bar", "baz", "baz", "foo", "bar", "baz"],
        "value1": [10, 20, 30, 40, 50, 60, 70, 80, 90],
        "value2": [15, 25, 35, 45, 55, 65, 75, 85, 95],
        "weight": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    df = pd.DataFrame(data)
    result = aggregate_fields_by_label(df, "id", "weight", ["value1", "value2"], label_column="label")
    print(result)

    test_result = pd.DataFrame(
        {
            "group": ["bar", "baz", "foo"],
            "aggregated_value1": [890/15, 71.0, 54.0],
            "aggregated_value2": [965/15, 76.0, 59.0],
        }
    )
    assert result.equals(test_result), f"Expected {test_result}, but got {result}"

def test_aggregate_fields_by_label_no_label_no_regex() -> None:
    data = {
        "id": ["A0", "B0", "C0", "A1", "B1", "C1", "A2", "B2", "C2"],
        "value1": [10, 20, 30, 40, 50, 60, 70, 80, 90],
        "value2": [15, 25, 35, 45, 55, 65, 75, 85, 95],
        "weight": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    df = pd.DataFrame(data)
    result = aggregate_fields_by_label(df, "id", "weight", ["value1", "value2"])
    print(result)

    test_result = pd.DataFrame(
        {
            "group": ["A0", "A1", "A2", "B0", "B1", "B2", "C0", "C1", "C2"],
            "aggregated_value1": [10.0, 40.0, 70.0, 20.0, 50.0, 80.0, 30.0, 60.0, 90.0],
            "aggregated_value2": [15.0, 45.0, 75.0, 25.0, 55.0, 85.0, 35.0, 65.0, 95.0],
        }
    )
    assert result.equals(test_result), f"Expected {test_result}, but got {result}"

if __name__ == "__main__":
    test_aggregate_fields_by_label_with_regex()
    test_aggregate_fields_by_label_with_label()
    test_aggregate_fields_by_label_no_label_no_regex()
