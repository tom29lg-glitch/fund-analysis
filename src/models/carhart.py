"""Carhart four-factor regression model."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import (
    RegressionResultsWrapper,
)
from statsmodels.stats.stattools import durbin_watson


def run_carhart_regression(
    dataset: pd.DataFrame,
    hac_lags: int = 3,
) -> RegressionResultsWrapper:
    """
    Estimate the Carhart four-factor model.

    Newey-West HAC standard errors are used to account for
    heteroskedasticity and residual autocorrelation.
    """
    required_columns = {
        "fund_excess",
        "MKT_RF",
        "SMB",
        "HML",
        "MOM",
    }

    missing_columns = required_columns.difference(
        dataset.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing model columns: {sorted(missing_columns)}"
        )

    model_data = dataset[
        [
            "fund_excess",
            "MKT_RF",
            "SMB",
            "HML",
            "MOM",
        ]
    ].dropna()

    if len(model_data) < 24:
        raise ValueError(
            "At least 24 monthly observations are required."
        )

    y = model_data["fund_excess"]

    x = model_data[
        ["MKT_RF", "SMB", "HML", "MOM"]
    ]
    x = sm.add_constant(x)

    model = sm.OLS(y, x)

    results = model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags},
    )

    return results


def create_coefficient_table(
    results: RegressionResultsWrapper,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Create a coefficient and significance table."""
    alpha = 1 - confidence_level
    confidence_intervals = results.conf_int(
        alpha=alpha
    )

    table = pd.DataFrame(
        {
            "coefficient": results.params,
            "standard_error": results.bse,
            "t_statistic": results.tvalues,
            "p_value": results.pvalues,
            "confidence_lower": confidence_intervals[0],
            "confidence_upper": confidence_intervals[1],
        }
    )

    table.index = table.index.map(
        lambda name: "alpha" if name == "const" else name
    )
    table.index.name = "term"

    return table


def create_model_summary(
    results: RegressionResultsWrapper,
) -> pd.Series:
    """Create the main Carhart model diagnostics."""
    monthly_alpha = float(results.params["const"])

    annualized_alpha = (
        (1 + monthly_alpha) ** 12 - 1
        if monthly_alpha > -1
        else np.nan
    )

    return pd.Series(
        {
            "observations": int(results.nobs),
            "r_squared": results.rsquared,
            "adjusted_r_squared": results.rsquared_adj,
            "monthly_alpha": monthly_alpha,
            "annualized_alpha": annualized_alpha,
            "durbin_watson": durbin_watson(results.resid),
        },
        name="carhart_model",
    )