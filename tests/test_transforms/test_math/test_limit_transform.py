import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.transforms.math import LimitTransform
from etna.models import ProphetModel
from etna.pipeline import Pipeline



@pytest.fixture()
def example_ts_(example_df_with_nontarget_column) -> TSDataset:
    ts = TSDataset(example_df_with_nontarget_column, freq="D")
    return ts


@pytest.mark.parametrize("lower_bound,upper_bound", [(-1e10, 1e10), (-15, 100)])
def test_fit_transform_target(example_ts_: TSDataset, lower_bound: float, upper_bound: float):
    """Check the value of transform result"""
    preprocess = LimitTransform(in_column="target", lower_bound=lower_bound, upper_bound=upper_bound)
    example_df_ = example_ts_.to_pandas()
    result = preprocess.fit_transform(ts=example_ts_).to_pandas()
    for segment in ["segment_1", "segment_2"]:
        true_array = np.log((example_df_[segment]["target_no_change"] - lower_bound + 1e-10) /
                            (upper_bound + 1e-10 - example_df_[segment]["target_no_change"]))
        np.testing.assert_array_almost_equal(
            result[segment]["target"], true_array
        )


@pytest.mark.parametrize("lower_bound,upper_bound", [(-1e10, 1e10), (-10, 10)])
def test_fit_transform_non_target(example_ts_: TSDataset, lower_bound: float, upper_bound: float):
    """Check the value of transform result"""
    preprocess = LimitTransform(in_column="non_target", lower_bound=lower_bound, upper_bound=upper_bound)
    example_df_ = example_ts_.to_pandas()
    result = preprocess.fit_transform(ts=example_ts_).to_pandas()
    for segment in ["segment_1", "segment_2"]:
        true_array = np.log((example_df_[segment]["non_target_no_change"] - lower_bound + 1e-10) /
                            (upper_bound + 1e-10 - example_df_[segment]["non_target_no_change"]))
        np.testing.assert_array_almost_equal(
            result[segment]["non_target"], true_array
        )


@pytest.mark.parametrize("lower_bound,upper_bound", [(-1e10, 1e10), (-15, 100)])
def test_inverse_transform(example_ts_: TSDataset, lower_bound: float, upper_bound: float):
    """Check that inverse_transform rolls back transform result"""
    preprocess = LimitTransform(in_column="target", lower_bound=lower_bound, upper_bound=upper_bound)
    example_df_ = example_ts_.to_pandas()
    preprocess.fit_transform(ts=example_ts_)
    preprocess.inverse_transform(ts=example_ts_)
    for segment in ["segment_1", "segment_2"]:
        np.testing.assert_array_almost_equal(
            example_ts_.to_pandas()[segment]["target"], example_df_[segment]["target_no_change"], decimal=5
        )


def test_fit_transform_values_out_borders(example_ts_: TSDataset):
    """Check that Exception raises when there are values out of bounds"""
    transform = LimitTransform(in_column="target", lower_bound=0, upper_bound=10)
    with pytest.raises(ValueError):
        transform.fit_transform(example_ts_)


def test_full_pipeline(example_ts_: TSDataset):
    model = ProphetModel()
    transform = LimitTransform(in_column="target", lower_bound=-20, upper_bound=105)
    pipeline = Pipeline(model=model, transforms=[transform], horizon=10)
    pipeline.fit(example_ts_)
    forecast_ts = pipeline.forecast()
    df = forecast_ts.to_pandas()
    features = df.loc[:, pd.IndexSlice[:, 'target']]
    assert (features >= -20).all().all() and (features <= 105).all().all()



