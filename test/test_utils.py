import pytest
from utils import validate_data_quality
import pandas as pd


def test_clean_data_removes_nulls():
    df = pd.DataFrame({"col": [1, None, 3]})
    cleaned = validate_data_quality(df)
    assert cleaned["col"].isnull().sum() == 0
