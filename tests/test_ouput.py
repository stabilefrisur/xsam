import pandas as pd
import pytest

from xsam.output import save


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})


def test_save_csv(sample_dataframe, tmp_path):
    save(sample_dataframe, "data", "csv", tmp_path, add_timestamp=False)
    assert (tmp_path / "data.csv").exists()


def test_save_xlsx(sample_dataframe, tmp_path):
    save(sample_dataframe, "data", "xlsx", tmp_path, add_timestamp=False)
    assert (tmp_path / "data.xlsx").exists()


def test_save_pickle(sample_dataframe, tmp_path):
    save(sample_dataframe, "data", "pickle", tmp_path, add_timestamp=False)
    assert (tmp_path / "data.p").exists()


if __name__ == "__main__":
    pytest.main()
