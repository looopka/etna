import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import MSE
from etna.models import ProphetModel
from etna.pipeline import Pipeline
from etna.transforms.math import LimitTransform


@pytest.fixture
def ts_check_pipeline_with_limit_transform(random_seed) -> TSDataset:
    periods = 100
    df1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df1["segment"] = ["segment_1"] * periods
    df1["target"] = np.arange(periods)
    df1.loc[5, "target"] = np.NaN
    df1["target_no_change"] = df1["target"]
    df1["non_target"] = np.random.uniform(0, 10, size=periods)
    df1["non_target_no_change"] = df1["non_target"]

    df2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df2["segment"] = ["segment_2"] * periods
    df2["target"] = np.random.uniform(-15, 5, size=periods)
    df2.loc[10, "target"] = np.NaN
    df2["target_no_change"] = df2["target"]
    df2["non_target"] = np.random.uniform(-10, 0, size=periods)
    df2["non_target_no_change"] = df2["non_target"]

    df = pd.concat((df1, df2))
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="D")
    return tsds


@pytest.fixture
def ts_check_limit_transform(random_seed) -> TSDataset:
    periods = 4
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df["segment"] = ["segment_1"] * periods
    df["target"] = np.array([1, 2, 3, 4])
    df["target_no_change"] = np.array([1, 2, 3, 4])
    df["target_transformed_1"] = np.array(
        [2.000000165280742e-10, 4.000000330161484e-10, 6.000000494642226e-10, 8.000000658722967e-10]
    )
    df["target_transformed_2"] = np.array(
        [-1.3862943610448906, -0.4054651080914978, 0.4054651080914977, 1.3862943610448906]
    )
    df["non_target"] = np.array([5, 6, 7, 8])
    df["non_target_transformed_1"] = np.array(
        [1.000000082240371e-09, 1.200000098568445e-09, 1.3999998928119147e-09, 1.599999909059989e-09]
    )
    df["non_target_transformed_2"] = np.array([-0.6931471805349453, 0.0, 0.6931471805349453, 1.6094379123541003])
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="D")
    return tsds


@pytest.mark.parametrize("lower_bound,upper_bound,n_test", [(-1e10, 1e10, "1"), (0, 5, "2")])
def test_fit_transform_target(ts_check_limit_transform: TSDataset, lower_bound: float, upper_bound: float, n_test: str):
    """Check forward scaled logit transform works correctly on target column"""
    preprocess = LimitTransform(in_column="target", lower_bound=lower_bound, upper_bound=upper_bound)
    result = preprocess.fit_transform(ts=ts_check_limit_transform).to_pandas()
    np.testing.assert_array_almost_equal(
        result["segment_1"]["target"], result["segment_1"]["target_transformed_" + n_test]
    )


@pytest.mark.parametrize("lower_bound,upper_bound,n_test", [(-1e10, 1e10, "1"), (3, 9, "2")])
def test_fit_transform_non_target(
    ts_check_limit_transform: TSDataset, lower_bound: float, upper_bound: float, n_test: str
):
    """Check forward scaled logit transform works correctly on nontarget column"""
    preprocess = LimitTransform(in_column="non_target", lower_bound=lower_bound, upper_bound=upper_bound)
    result = preprocess.fit_transform(ts=ts_check_limit_transform).to_pandas()
    np.testing.assert_array_almost_equal(
        result["segment_1"]["non_target"], result["segment_1"]["non_target_transformed_" + n_test]
    )


@pytest.mark.parametrize("lower_bound,upper_bound", [(-1e10, 1e10), (-15, 100)])
def test_inverse_transform(ts_check_limit_transform: TSDataset, lower_bound: float, upper_bound: float):
    """Check that inverse_transform rolls back transform result"""
    preprocess = LimitTransform(in_column="target", lower_bound=lower_bound, upper_bound=upper_bound)
    preprocess.fit_transform(ts=ts_check_limit_transform)
    preprocess.inverse_transform(ts=ts_check_limit_transform)
    result = ts_check_limit_transform.to_pandas()
    np.testing.assert_array_almost_equal(result["segment_1"]["target"], result["segment_1"]["target_no_change"])


@pytest.mark.parametrize("lower_bound,upper_bound", [(-10, 200), (-30, 50), (-5, 5)])
def test_fit_transform_values_out_of_borders(
    ts_check_pipeline_with_limit_transform: TSDataset, lower_bound: float, upper_bound: float
):
    """Check that Exception raises when there are values out of bounds"""
    transform = LimitTransform(in_column="target", lower_bound=0, upper_bound=10)
    with pytest.raises(ValueError):
        transform.fit_transform(ts_check_pipeline_with_limit_transform)


def test_pipeline_with_limit_transform(ts_check_pipeline_with_limit_transform: TSDataset):
    """Check that forecasted target and quantiles are in given range with LimitTransform"""
    model = ProphetModel()
    transform = LimitTransform(in_column="target", lower_bound=-20, upper_bound=105)
    pipeline = Pipeline(model=model, transforms=[transform], horizon=10)
    pipeline.fit(ts_check_pipeline_with_limit_transform)
    forecast_ts = pipeline.forecast(prediction_interval=True)
    forecast_df = forecast_ts.to_pandas()
    features = forecast_df.loc[:, pd.IndexSlice[:, ["target", "target_0.025", "target_0.975"]]]
    assert (features >= -20).all().all() and (features <= 105).all().all()


def test_pipeline_without_limit_transform(ts_check_pipeline_with_limit_transform: TSDataset):
    """Check that forecasted target is not in given range without LimitTransform"""
    model = ProphetModel()
    pipeline = Pipeline(model=model, horizon=10)
    pipeline.fit(ts_check_pipeline_with_limit_transform)
    forecast_ts = pipeline.forecast()
    forecast_df = forecast_ts.to_pandas()
    assert (forecast_df < -20).any().any() or (forecast_df > 105).any().any()


def test_backtest(ts_check_pipeline_with_limit_transform: TSDataset):
    """Check that backtest function executes without errors"""
    model = ProphetModel()
    transform = LimitTransform(in_column="target", lower_bound=-20, upper_bound=105)
    pipeline = Pipeline(model=model, transforms=[transform], horizon=10)
    pipeline.backtest(ts=ts_check_pipeline_with_limit_transform, metrics=[MSE()])
