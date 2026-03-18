Title: Optimize Aggregation Efficiency

Description:
The current aggregation process involves building `final_df` by merging one dataframe at a time using the following code snippet:

```python
aggregated_dfs = []
for field_column in field_columns:
    aggregated_df = (
        df.groupby(["Group", "Subgroup"], observed=True)
        .apply(method_dict[method], field_column, include_groups=False)
        .reset_index()
    )
    aggregated_df.columns = ["Group", "Subgroup", field_column]
    aggregated_dfs.append(aggregated_df)
```

This sequential merging approach could be inefficient. Please investigate ways to optimize this process while ensuring no changes to the final output.