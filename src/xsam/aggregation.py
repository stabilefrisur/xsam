import pandas as pd
import re

def aggregate_by_label(
    df: pd.DataFrame, 
    label_column: str, 
    weight_column: str, 
    agg_column: str, 
    regex_dict: dict[str, dict[str, tuple[str, int]]] = None
) -> pd.DataFrame:
    """Aggregate values in a DataFrame by label, with optional regex matching and multipliers.

    Args:
        df (pd.DataFrame): DataFrame containing the data to aggregate.
        label_column (str): Label column to group by.
        weight_column (str): Weight column to use for weighted average.
        agg_column (str): Aggregation column to calculate the weighted average.
        regex_dict (dict): Dictionary of regex patterns and multipliers. Defaults to None.

        regex_dict example:
        {
            "group1": {
                "subgroup1": ("regex1", multiplier1),
                "subgroup2": ("regex2", multiplier2),
            },
            "group2": {
                "subgroup1": ("regex3", multiplier3),
                "subgroup2": ("regex4", multiplier4),
            },
        }

    Returns:
        pd.DataFrame: DataFrame with aggregated values for each group.
    """
    if regex_dict:
        # Create new columns 'group', 'subgroup', and 'multiplier' based on regex matching
        def match_label(
            label: str,
        ) -> list[tuple[str, str, int]]:
            matches = []
            for group, subgroups in regex_dict.items():
                for subgroup, (regex, multiplier) in subgroups.items():
                    if re.match(regex, label):
                        matches.append((group, subgroup, multiplier))
            return matches if matches else [(None, None, None)]

        expanded_rows = []
        for _, row in df.iterrows():
            matches = match_label(row[label_column])
            for match in matches:
                new_row = row.copy()
                new_row["group"], new_row["subgroup"], new_row["multiplier"] = match
                expanded_rows.append(new_row)
        df = pd.DataFrame(expanded_rows)
    else:
        # Use the label column as the group and set subgroup as 'default' with multiplier 1
        df["group"] = df[label_column]
        df["subgroup"] = "default"
        df["multiplier"] = 1

    # Drop rows where 'group' is None (no match found)
    df = df.dropna(subset=["group"])

    # Calculate weighted value for each row
    df["weighted_value"] = df[agg_column] * df[weight_column]

    # Calculate weighted sum for each group and subgroup
    def calculate_weighted_sum(group: pd.DataFrame) -> float:
        return group["weighted_value"].sum() / group[weight_column].sum()

    aggregated_df = (
        df.groupby(["group", "subgroup"]).apply(calculate_weighted_sum).reset_index()
    )
    aggregated_df.columns = ["group", "subgroup", "aggregated_value"]

    # Create a unique index for the subgroup multipliers
    multiplier_dict = {
        f"{group}_{subgroup}": multiplier
        for group, subgroups in regex_dict.items()
        for subgroup, (_, multiplier) in subgroups.items()
    }

    # Apply multipliers to the aggregated values
    aggregated_df["aggregated_value"] *= aggregated_df.apply(
        lambda row: multiplier_dict[f"{row['group']}_{row['subgroup']}"], axis=1
    )

    # Sum the values for each group
    final_df = aggregated_df.groupby("group")["aggregated_value"].sum().reset_index()

    return final_df

# Test function
def test_aggregate_by_label() -> None:
    data = {
        "label": ["A0", "B0", "C0", "A1", "B1", "C1", "A2", "B2", "C2"],
        "value": [10, 20, 30, 40, 50, 60, 70, 80, 90],
        "weight": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    df = pd.DataFrame(data)
    regex_dict = {
        "A_group": {"long": ("A.*", 1), "short": ("A1", -1)},
        "B_group": {"long": ("B.*", 1), "short": ("B1", -1)},
        "C_group": {"long": ("C.*", 1), "short": ("C1", -1)},
    }
    result = aggregate_by_label(df, "label", "weight", "value", regex_dict)
    print(result)

    test_result = pd.DataFrame(
        {
            "group": ["A_group", "B_group", "C_group"],
            "aggregated_value": [15.0, 12.0, 10.0],
        }
    )
    assert result.equals(test_result), f"Expected {test_result}, but got {result}"

if __name__ == "__main__":
    test_aggregate_by_label()
