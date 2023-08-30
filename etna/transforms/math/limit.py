from typing import List

import numpy as np
import pandas as pd

from etna.transforms.base import ReversibleTransform


class LimitTransform(ReversibleTransform):
    """LimitTransform limits values of some feature between the borders.

    For more details visit https://datasciencestunt.com/time-series-forecasting-within-limits/ .
    """

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

    def _fit(self, df: pd.DataFrame):
        """Fit method does nothing and is kept for compatibility.

        Parameters
        ----------
        df:
            dataframe with data.
        """
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaled logit transform to the dataset.

        .. math::
            y = \\log (\\frac{x-a}{b-x}),

        where :math:`x` is feature, :math:`a` is lower bound, :math:`b` is upper bound.
        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        :
            transformed dataframe

        Raises
        ------
        ValueError:
            Some values of ``df`` is less than ``lower_bound`` or greater than ``upper_bound``.
        """
        if (df < self.lower_bound).any().any() or (df > self.upper_bound).any().any():
            raise ValueError(f"Detected values out [{self.lower_bound}, {self.upper_bound}]")

        transformed_features = np.log((df - self.lower_bound + self.tol) / (self.upper_bound + self.tol - df))
        return transformed_features

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaled logit reverse transform to the dataset.

        .. math::
            x = \\frac{(b-a) \\exp{y}}{1 + \\exp{y}} + a,

        where :math:`y` is feature, :math:`a` is lower bound, :math:`b` is upper bound.
        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        :
            transformed series
        """
        exp_df = np.exp(df)
        transformed_features = ((self.upper_bound - self.lower_bound) * exp_df) / (1 + exp_df) + self.lower_bound
        return transformed_features

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return []


__all__ = ["LimitTransform"]
