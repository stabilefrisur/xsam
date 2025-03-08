import pandas as pd
import re


def aggregate_fields_by_label(
    df: pd.DataFrame,
    id_column: str,
    weight_column: str = None,
    field_columns: list[str] = None,
    label_column: str = None,
    label_regex: dict[str, dict[str, tuple[str, int]]] = None,
    method: str = "sum",
    preliminary: bool = False,
) -> pd.DataFrame:
    """Aggregate values in a DataFrame by label, with optional regex matching and multipliers.

    Args:
        df (pd.DataFrame): DataFrame containing the data to aggregate.
        id_column (str): IDs column contains IDs to be associated with groups.
        weight_column (str): Weight column to use for weighted averages.
        field_columns (list[str]): List of field columns to be aggregated as weighted average by group.
        label_column (str): Column containing labels to use for grouping. Defaults to None.
        label_regex (dict): Dictionary of regex patterns and multipliers to use for grouping. Defaults to None.
        method (str): Method to use for aggregation. Defaults to "sum".
        preliminary (bool): Whether to return the DataFrame before summing the values for each group. Defaults to False.

        method options:
            "sum": Sum the values for each group. Use for count, market value, already weighted values.
            "wsum": Sum the weighted values for each group. Use to calculate contribution to a total.
            "avg": Average the values for each group. Use for equal weighting.
            "wavg": Average the weighted values for each group. Use to calculate average contribution.

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
        pd.DataFrame: DataFrame with aggregated values for each group. Index is the group column. Columns are the field columns.
    """
    # Assert that the required columns are present
    assert id_column in df.columns, f"Column '{id_column}' not found in DataFrame."
    if weight_column:
        assert weight_column in df.columns, (
            f"Column '{weight_column}' not found in DataFrame."
        )
    if field_columns:
        for field_column in field_columns:
            assert field_column in df.columns, (
                f"Column '{field_column}' not found in DataFrame."
            )
    if label_column:
        assert label_column in df.columns, (
            f"Column '{label_column}' not found in DataFrame."
        )

    # # Assert that either label_regex or label_column but not both are provided
    # assert (label_regex is None) != (label_column is None), (
    #     "Either 'label_regex' or 'label_column' should be provided."
    # )

    # Assert that method is valid
    assert method in ["sum", "wsum", "avg", "wavg"], "Invalid aggregation method."

    # Assert that weight column is provided for weighted methods
    if method in ["wsum", "wavg"]:
        assert weight_column, "Weight column must be provided for weighted methods."

    # If no field columns are provided, use all columns that are numerical
    if not field_columns:
        field_columns = df.select_dtypes(include="number").columns.tolist()

    # Calculate sum for each group and subgroup
    def calculate_sum(group: pd.DataFrame, field_column: str) -> float:
        return group[field_column].sum()

    # Calculate weighted sum for each group and subgroup
    def calculate_wsum(group: pd.DataFrame, field_column: str) -> float:
        return group[f"Weighted_{field_column}"].sum()

    # Calculate average for each group and subgroup
    def calculate_avg(group: pd.DataFrame, field_column: str) -> float:
        return group[field_column].mean()

    # Calculate weighted average for each group and subgroup
    def calculate_wavg(group: pd.DataFrame, field_column: str) -> float:
        return group[f"Weighted_{field_column}"].sum() / group[weight_column].sum()

    method_dict = {
        "sum": calculate_sum,
        "wsum": calculate_wsum,
        "avg": calculate_avg,
        "wavg": calculate_wavg,
    }

    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

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
                new_row["Group"], new_row["Subgroup"], new_row["Multiplier"] = match
                expanded_rows.append(new_row)
        df = pd.DataFrame(expanded_rows)
    elif label_column:
        # Use the label column as the group and set subgroup as 'default' with multiplier 1
        df["Group"] = df[label_column]
        df["Subgroup"] = "Default"
        df["Multiplier"] = 1
    else:
        # Use the id column as the group and set subgroup as 'default' with multiplier 1
        df["Group"] = df[id_column]
        df["Subgroup"] = "Default"
        df["Multiplier"] = 1

    # Drop rows where 'group' is None (no match found)
    df = df.dropna(subset=["Group"])

    # Calculate weighted value for each row and each field column
    if method in ["wsum", "wavg"]:
        for field_column in field_columns:
            df[f"Weighted_{field_column}"] = df[field_column] * df[weight_column]

    aggregated_dfs = []
    for field_column in field_columns:
        aggregated_df = (
            df.groupby(["Group", "Subgroup"], observed=True)
            .apply(method_dict[method], field_column, include_groups=False)
            .reset_index()
        )
        aggregated_df.columns = ["Group", "Subgroup", field_column]
        aggregated_dfs.append(aggregated_df)

    final_df = aggregated_dfs[0]
    for aggregated_df in aggregated_dfs[1:]:
        final_df = final_df.merge(aggregated_df, on=["Group", "Subgroup"])

    if label_regex:
        # Create a unique index for the subgroup multipliers
        multiplier_dict = {
            f"{group}_{subgroup}": multiplier
            for group, subgroups in label_regex.items()
            for subgroup, (_, multiplier) in subgroups.items()
        }

        # Apply multipliers to the aggregated values
        for field_column in field_columns:
            final_df[field_column] *= final_df.apply(
                lambda row: multiplier_dict[f"{row['Group']}_{row['Subgroup']}"], axis=1
            )

    # Make categorical columns for 'Group' ordered by label_regex or label_column or id_column
    if label_regex:
        final_df["Group"] = pd.Categorical(
            final_df["Group"], categories=label_regex.keys(), ordered=True
        )
    elif label_column:
        final_df["Group"] = pd.Categorical(
            final_df["Group"], categories=df[label_column].unique(), ordered=True
        )
    else:
        final_df["Group"] = final_df["Group"].astype(df[id_column].dtype)

    # If preliminary results are requested, return the DataFrame before summing
    if preliminary:
        return final_df.set_index(["Group", "Subgroup"]).sort_index()

    # Sum the values for each group
    group_sums = (
        final_df.groupby("Group", observed=True)
        .sum()
        .reset_index()
        .drop(columns=["Subgroup"])
    )

    return group_sums.set_index("Group").sort_index()
