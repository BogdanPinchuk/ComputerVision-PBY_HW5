from typing import Optional

import numpy as np
import pandas as pd
import apps.reporter as rpt
from IPython.display import display
from IPython.core.display import Markdown
from pandas import Series, DataFrame
from pandas.io.formats.style import Styler
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, mean_absolute_percentage_error)


def calc_metrics(original_data, changed_data, max_value: Optional[float] = None) -> Styler:
    """
    Calc and print metrics
    :param original_data: original target test set
    :param changed_data: predicted set
    :param max_value: the maximum value for PSNR
    :return: DataFrame styler
    """
    rp = rpt.Reporter()
    rp.tolerance = 4

    original_flat = original_data.flatten()
    changed_flat = changed_data.flatten()

    # Note. I used it from a Data Science course
    # Mean Absolute Error
    mae = mean_absolute_error(original_flat, changed_flat)
    rp.add_item("MAE", rp.format_value(mae))  # type: ignore
    # Mean Squared Error
    mse = mean_squared_error(original_flat, changed_flat)
    rp.add_item("MSE", rp.format_value(mse))  # type: ignore
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    rp.add_item("RMSE", rp.format_value(rmse))  # type: ignore
    if max_value is not None:
        # Peak Signal-To-Noise Ratio
        psnr = 20 * np.log10(max_value / rmse)
        rp.add_item("PSNR, dB", rp.format_value(psnr))  # type: ignore
    # R2 - coefficient of determination
    r2 = r2_score(original_flat, changed_flat)
    rp.add_item(f"R²\n(coefficient of the determination)", rp.format_value(r2))
    # # Mean Absolute Percentage Error
    # mape = mean_absolute_percentage_error(original_flat, changed_flat)
    # rp.add_item("MAPE", rp.format_value(mape))

    df = rp.get_pd_report()

    # Print results
    rp.print_pd_report(f"Metrics")

    return df
