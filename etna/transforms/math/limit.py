from typing import List
import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.datasets import set_columns_wide
from etna.transforms.base import ReversibleTransform
from etna.transforms.utils import match_target_quantiles


class LimitTransform(ReversibleTransform):
    """LimitTransform limits values of some feature between the borders"""

    def __init__(self, in_column: str, lower_bound: float = -1e10, upper_bound: float = 1e10):
        """
        Init LimitTransform.

        Parameters
        ----------
        in_column:
            column to apply transform
        lower_bound:
            lower bound for the value of the column
        upper_bound:
            upper bound for the value of the column
        """
        super().__init__(required_features=[in_column])
        self.in_column = in_column
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.tol = 1e-10

    def _get_column_name(self) -> str:
        return self.in_column

    def _fit(self, df: pd.DataFrame) -> "LimitTransform":
        """Fit method does nothing and is kept for compatibility.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        result: AddConstTransform
        """
        return self

    def fit(self, ts: TSDataset) -> "LimitTransform":
        """Fit the transform."""
        super().fit(ts)
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaled logit transform to the dataset.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result: pd.Dataframe
            transformed dataframe
        """
        result = df
        features = df.loc[:, pd.IndexSlice[:, self.in_column]]
        if (features < self.lower_bound).any().any() or (features > self.upper_bound).any().any():
            raise ValueError("Detected values out of limit")
        # reference to formula https://datasciencestunt.com/time-series-forecasting-within-limits/
        transformed_features = np.log((features - self.lower_bound + self.tol) /
                                      (self.upper_bound + self.tol - features))
        result = set_columns_wide(
            result, transformed_features, features_left=[self.in_column], features_right=[self.in_column]
        )
        return result

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inverse transformation to the dataset.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result: pd.DataFrame
            transformed series
        """
        result = df
        features = df.loc[:, pd.IndexSlice[:, self.in_column]]

        range_fc = self.upper_bound - self.lower_bound
        exp_fc = np.exp(features)
        transformed_features = (range_fc * exp_fc) / (1 + exp_fc) + self.lower_bound
        result = set_columns_wide(
            result, transformed_features, features_left=[self.in_column], features_right=[self.in_column]
        )
        if self.in_column == "target":
            segment_columns = result.columns.get_level_values("feature").tolist()
            quantiles = match_target_quantiles(set(segment_columns))
            for quantile_column_nm in quantiles:
                features = df.loc[:, pd.IndexSlice[:, quantile_column_nm]]
                exp_fc = np.exp(features)
                transformed_features = (range_fc * exp_fc) / (1 + exp_fc) + self.lower_bound
                result = set_columns_wide(
                    result,
                    transformed_features,
                    features_left=[quantile_column_nm],
                    features_right=[quantile_column_nm],
                )

        return result

    def get_regressors_info(self) -> List[str]:
        return []


__all__ = ["LimitTransform"]
