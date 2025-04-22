import pandas as pd
import pytest

from xsam.output import save
from xsam.input import load


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})


def test_load_csv(sample_dataframe, tmp_path):
    save(sample_dataframe, "csv_data", "csv", tmp_path, add_timestamp=False)
    loaded_df = load("csv_data", "csv")
    pd.testing.assert_frame_equal(sample_dataframe, loaded_df)


def test_load_xlsx(sample_dataframe, tmp_path):
    save(sample_dataframe, "xlsx_data", "xlsx", tmp_path, add_timestamp=False)
    loaded_df = load("xlsx_data", "xlsx")
    pd.testing.assert_frame_equal(sample_dataframe, loaded_df)


def test_load_pickle(sample_dataframe, tmp_path):
    save(sample_dataframe, "pickle_data", "pickle", tmp_path, add_timestamp=False)
    loaded_df = load("pickle_data", "pickle")
    pd.testing.assert_frame_equal(sample_dataframe, loaded_df)
