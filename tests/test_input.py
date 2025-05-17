import pandas as pd
import pytest

from xsam.output import export_obj
from xsam.input import import_obj


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})


def test_import_csv(sample_dataframe, tmp_path):
    export_obj(sample_dataframe, "csv_data", file_extension="csv", file_path=tmp_path, add_timestamp=False)
    loaded_df = import_obj(file_name="csv_data", file_extension="csv")
    loaded_df = loaded_df.set_index(loaded_df.columns[0])
    pd.testing.assert_frame_equal(sample_dataframe, loaded_df, check_names=False)


def test_import_xlsx(sample_dataframe, tmp_path):
    export_obj(sample_dataframe, "xlsx_data", file_extension="xlsx", file_path=tmp_path, add_timestamp=False)
    loaded_df = import_obj(file_name="xlsx_data", file_extension="xlsx")
    loaded_df = loaded_df.set_index(loaded_df.columns[0])
    pd.testing.assert_frame_equal(sample_dataframe, loaded_df, check_names=False)


def test_import_pickle(sample_dataframe, tmp_path):
    export_obj(sample_dataframe, "pickle_data", file_extension="p", file_path=tmp_path, add_timestamp=False)
    loaded_df = import_obj(file_name="pickle_data", file_extension="p")
    pd.testing.assert_frame_equal(sample_dataframe, loaded_df)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
