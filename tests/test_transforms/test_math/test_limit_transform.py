import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.metrics import MSE
from etna.models import NaiveModel
from etna.pipeline import Pipeline
from etna.transforms.math import LimitTransform


@pytest.fixture
def ts_check_pipeline_with_limit_transform(random_seed) -> TSDataset:
    periods = 100
    df1 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df1["segment"] = ["segment_1"] * periods
    df1["target"] = np.arange(periods)
    df1.loc[5, "target"] = np.NaN

    df2 = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=periods)})
    df2["segment"] = ["segment_2"] * periods
    df2["target"] = np.random.uniform(-15, 5, size=periods)
    df2.loc[10, "target"] = np.NaN

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
    df["target_transformed_1"] = np.array([1, 2, 3, 4])
    df["target_transformed_2"] = np.array(
        [1.3862943611448906, 1.0986122887014431, 0.6931471806099453, 1.000000082690371e-10]
    )
    df["target_transformed_3"] = np.array(
        [0.6931471806099453, 1.0986122887014431, 1.3862943611448906, 1.6094379124541003]
    )
    df["target_transformed_4"] = np.array(
         [-1.3862943610448906, -0.4054651080914978, 0.4054651080914977, 1.3862943610448906]
     )
    df["non_target"] = np.array([5, 6, 7, 8])
    df["non_target_transformed_1"] = np.array([5, 6, 7, 8])
    df["non_target_transformed_2"] = np.array(
        [1.3862943611448906, 1.0986122887014431, 0.6931471806099453, 1.000000082690371e-10]
    )
    df["non_target_transformed_3"] = np.array(
        [1.6094379124541003, 1.7917594692447216, 1.945910149069599, 2.079441541692336]
    )
    df["non_target_transformed_4"] = np.array([-0.6931471805349453, 0.0, 0.6931471805349453, 1.6094379123541003])
    df = TSDataset.to_dataset(df)
    tsds = TSDataset(df, freq="D")
    return tsds


@pytest.mark.parametrize("lower_bound,upper_bound,n_test", [(None, None, "1"), (None, 5, "2"), (-1, None, "3"), (0, 5, "4")])
def test_fit_transform_target(ts_check_limit_transform: TSDataset, lower_bound: float, upper_bound: float, n_test: str):
    """Check forward scaled logit transform works correctly on target column"""
    preprocess = LimitTransform(in_column="target", lower_bound=lower_bound, upper_bound=upper_bound)
    result = preprocess.fit_transform(ts=ts_check_limit_transform).to_pandas()
    np.testing.assert_array_almost_equal(
        result["segment_1"]["target"], result["segment_1"]["target_transformed_" + n_test]
    )


@pytest.mark.parametrize("lower_bound,upper_bound,n_test", [(None, None, "1"), (None, 9, "2"), (0, None, "3"), (3, 9, "4")])
def test_fit_transform_non_target(
    ts_check_limit_transform: TSDataset, lower_bound: float, upper_bound: float, n_test: str
):
    """Check forward scaled logit transform works correctly on nontarget column"""
    preprocess = LimitTransform(in_column="non_target", lower_bound=lower_bound, upper_bound=upper_bound)
    result = preprocess.fit_transform(ts=ts_check_limit_transform).to_pandas()
    np.testing.assert_array_almost_equal(
        result["segment_1"]["non_target"], result["segment_1"]["non_target_transformed_" + n_test]
    )


@pytest.mark.parametrize("lower_bound,upper_bound", [(None, None), (None, 9), (0, None), (0, 9)])
def test_inverse_transform(ts_check_limit_transform: TSDataset, lower_bound: float, upper_bound: float):
    """Check that inverse_transform rolls back transform result"""
    df_copy = ts_check_limit_transform.to_pandas()
    preprocess = LimitTransform(in_column="target", lower_bound=lower_bound, upper_bound=upper_bound)
    preprocess.fit_transform(ts=ts_check_limit_transform)
    preprocess.inverse_transform(ts=ts_check_limit_transform)
    result = ts_check_limit_transform.to_pandas()
    np.testing.assert_array_almost_equal(result["segment_1"]["target"], df_copy["segment_1"]["target"])


@pytest.mark.parametrize("lower_bound,upper_bound,n_test", [(-10, 200, '1'), (-10, None, '2'), (-30, 50, '3'), (None, 50, '4'), (-5, 5, '5')])
def test_fit_transform_values_out_of_borders(
    ts_check_pipeline_with_limit_transform: TSDataset, lower_bound: float, upper_bound: float, n_test: str
):
    """Check that Exception raises when there are values out of bounds"""
    transform = LimitTransform(in_column="target", lower_bound=lower_bound, upper_bound=upper_bound)
    if n_test == '2':
        left_border = lower_bound
        right_border = np.inf
    elif n_test == '4':
        left_border = np.NINF
        right_border = upper_bound
    else:
        left_border = lower_bound
        right_border = upper_bound
    with pytest.raises(ValueError, match=f"Detected values out \[{left_border}, {right_border}\]"):
        transform.fit_transform(ts_check_pipeline_with_limit_transform)


def test_pipeline_with_limit_transform(ts_check_pipeline_with_limit_transform: TSDataset):
    """
    Check that forecasted target and quantiles are in given range with LimitTransform
    Check that forecasted target is not in given range without LimitTransform
    """
    model = NaiveModel()
    transform = LimitTransform(in_column="target", lower_bound=-20, upper_bound=105)
    pipeline_with_limit = Pipeline(model=model, transforms=[transform], horizon=10)
    pipeline_without_limit = Pipeline(model=model, transforms=[], horizon=10)

    pipeline_with_limit.fit(ts_check_pipeline_with_limit_transform)
    pipeline_without_limit.fit(ts_check_pipeline_with_limit_transform)

    forecast_ts_limited = pipeline_with_limit.forecast(prediction_interval=True)
    forecast_ts_unlimited = pipeline_without_limit.forecast(prediction_interval=True)

    forecast_df_limited = forecast_ts_limited.to_pandas()
    forecast_df_unlimited = forecast_ts_unlimited.to_pandas()
    assert (forecast_df_limited >= -20).all().all() and (forecast_df_limited <= 105).all().all()
    assert (forecast_df_unlimited < -20).any().any() or (forecast_df_unlimited > 105).any().any()


def test_backtest(ts_check_pipeline_with_limit_transform: TSDataset):
    """Check that backtest function executes without errors"""
    model = NaiveModel()
    transform = LimitTransform(in_column="target", lower_bound=-20, upper_bound=105)
    pipeline = Pipeline(model=model, transforms=[transform], horizon=10)
    pipeline.backtest(ts=ts_check_pipeline_with_limit_transform, metrics=[MSE()], n_folds=2)
