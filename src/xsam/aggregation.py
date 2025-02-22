import pandas as pd
import re


def aggregate_fields_by_label(
    df: pd.DataFrame,
    id_column: str,
    weight_column: str,
    field_columns: list[str],
    label_column: str = None,
    label_regex: dict[str, dict[str, tuple[str, int]]] = None,
) -> pd.DataFrame:
    """Aggregate values in a DataFrame by label, with optional regex matching and multipliers.

    Args:
        df (pd.DataFrame): DataFrame containing the data to aggregate.
        id_column (str): IDs column contains IDs to be associated with groups.
        weight_column (str): Weight column to use for weighted averages.
        field_columns (list[str]): List of field columns to be aggregated as weighted average by group.
        label_column (str): Column containing labels to use for grouping. Defaults to None.
        label_regex (dict): Dictionary of regex patterns and multipliers to use for grouping. Defaults to None.

        label_regex example:
        {
            "group1": {
                "subgroup1": ("regex1", multiplier1),
                "subgroup2": ("regex2", multiplier2),
                # ...additional subgroups...
            },
            "group2": {
                "subgroup1": ("regex3", multiplier3),
                # ...additional subgroups...
            },
            # ...additional groups...
        }

    Returns:
        pd.DataFrame: DataFrame with aggregated values for each group.
    """
    # Check if label_regex and label_column are both provided
    if label_regex and label_column:
        raise ValueError(
            "Only one of 'label_regex' and 'label_column' should be provided."
        )

    if label_regex:
        # Create new columns 'group', 'subgroup', and 'multiplier' based on regex matching
        def match_label(
            label: str,
        ) -> list[tuple[str, str, int]]:
            matches = []
            for group, subgroups in label_regex.items():
                for subgroup, (regex, multiplier) in subgroups.items():
                    if re.match(regex, label):
                        matches.append((group, subgroup, multiplier))
            return matches if matches else [(None, None, None)]

        expanded_rows = []
        for _, row in df.iterrows():
            matches = match_label(row[id_column])
            for match in matches:
                new_row = row.copy()
                new_row["group"], new_row["subgroup"], new_row["multiplier"] = match
                expanded_rows.append(new_row)
        df = pd.DataFrame(expanded_rows)
    elif label_column:
        # Use the label column as the group and set subgroup as 'default' with multiplier 1
        df["group"] = df[label_column]
        df["subgroup"] = "default"
        df["multiplier"] = 1
    else:
        # Use the id column as the group and set subgroup as 'default' with multiplier 1
        df["group"] = df[id_column]
        df["subgroup"] = "default"
        df["multiplier"] = 1

    # Drop rows where 'group' is None (no match found)
    df = df.dropna(subset=["group"])

    # Calculate weighted value for each row and each field column
    for field_column in field_columns:
        df[f"weighted_{field_column}"] = df[field_column] * df[weight_column]

    # Calculate weighted sum for each group and subgroup
    def calculate_weighted_sum(group: pd.DataFrame, field_column: str) -> float:
        return group[f"weighted_{field_column}"].sum() / group[weight_column].sum()

    aggregated_dfs = []
    for field_column in field_columns:
        aggregated_df = (
            df.groupby(["group", "subgroup"])
            .apply(calculate_weighted_sum, field_column, include_groups=False)
            .reset_index()
        )
        aggregated_df.columns = ["group", "subgroup", f"aggregated_{field_column}"]
        aggregated_dfs.append(aggregated_df)

    final_df = aggregated_dfs[0]
    for aggregated_df in aggregated_dfs[1:]:
        final_df = final_df.merge(aggregated_df, on=["group", "subgroup"])

    if label_regex:
        # Create a unique index for the subgroup multipliers
        multiplier_dict = {
            f"{group}_{subgroup}": multiplier
            for group, subgroups in label_regex.items()
            for subgroup, (_, multiplier) in subgroups.items()
        }

        # Apply multipliers to the aggregated values
        for field_column in field_columns:
            final_df[f"aggregated_{field_column}"] *= final_df.apply(
                lambda row: multiplier_dict[f"{row['group']}_{row['subgroup']}"], axis=1
            )

    # Sum the values for each group
    group_sums = final_df.groupby("group").sum().reset_index()
    group_sums = group_sums.drop(columns=["subgroup"])

    return group_sums
