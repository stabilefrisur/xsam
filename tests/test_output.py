import pandas as pd
import pytest

from xsam.output import export_obj


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})


def test_export_csv(sample_dataframe, tmp_path):
    export_obj(sample_dataframe, "data", file_extension="csv", file_path=tmp_path, add_timestamp=False)
    assert (tmp_path / "data.csv").exists()


def test_export_xlsx(sample_dataframe, tmp_path):
    export_obj(sample_dataframe, "data", file_extension="xlsx", file_path=tmp_path, add_timestamp=False)
    assert (tmp_path / "data.xlsx").exists()


def test_export_pickle(sample_dataframe, tmp_path):
    export_obj(sample_dataframe, "data", file_extension="p", file_path=tmp_path, add_timestamp=False)
    assert (tmp_path / "data.p").exists()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
