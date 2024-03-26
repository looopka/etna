"""
Module for AutoML utilities.

Basic usage
-----------

.. code-block::

   import pathlib

   import pandas as pd

   from etna.auto import Auto
   from etna.datasets import TSDataset
   from etna.metrics import SMAPE

   CURRENT_DIR_PATH = pathlib.Path(__file__).parent

   if __name__ == "__main__":
      df = pd.read_csv(CURRENT_DIR_PATH / "data" / "example_dataset.csv")
      ts = TSDataset(df, freq="D")

      # Create Auto object for greedy search
      # All trials will be saved in sqlite database
      # You can use it later for analysis with ``Auto.summary``
      auto = Auto(
         target_metric=SMAPE(),
         horizon=14,
         experiment_folder="auto-example",
      )

      # Get best pipeline
      best_pipeline = auto.fit(ts, catch=(Exception,))
      print(best_pipeline)

      # Get all metrics of greedy search
      print(auto.summary())
"""

from etna import SETTINGS

if SETTINGS.auto_required:
    from etna.auto.auto import Auto
    from etna.auto.auto import Tune
    from etna.auto.pool import Pool
