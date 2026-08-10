"""Per-ticker fixed-parameter GARCH(1,1) rolling-origin forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ml.data import DATE_COLUMN, TICKER_COLUMN

try:  # MLflow is optional for importing the forecasting class itself.
    from mlflow.pyfunc import PythonModel
except ImportError:  # pragma: no cover - dependency setup path
    class PythonModel:  # type: ignore[no-redef]
        """Fallback base used when MLflow is not installed."""


class GarchPyfuncModel(PythonModel):
    """MLflow pyfunc adapter for one fitted ticker's rolling forecasts."""

    def __init__(self, forecaster: "GarchForecaster") -> None:
        self.forecaster = forecaster

    def predict(self, context: object, model_input: pd.DataFrame) -> np.ndarray:
        """Return one forecast per input row."""

        return self.forecaster.predict(model_input)["prediction"].to_numpy(dtype=float)

RETURNS_SCALE = 100.0
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class GarchParameters:
    """Fitted parameters and terminal state for one ticker."""

    ticker: str
    omega: float
    alpha: float
    beta: float
    nu: float
    last_variance: float
    last_return: float
    train_end: pd.Timestamp


class GarchForecaster:
    """Fit one Student-t GARCH(1,1) per ticker and forecast 20-day RV."""

    def __init__(
        self,
        *,
        horizon: int = 20,
        min_train_rows: int = 100,
    ) -> None:
        self.horizon = horizon
        self.min_train_rows = min_train_rows
        self.parameters: dict[str, GarchParameters] = {}

    def fit(self, train: pd.DataFrame) -> "GarchForecaster":
        """Fit parameters using only the supplied chronological training frame."""

        try:
            from arch import arch_model
        except ImportError as exc:  # pragma: no cover - setup failure
            raise RuntimeError("Install arch before training GARCH models") from exc

        self.parameters = {}
        for ticker, group in train.groupby(TICKER_COLUMN, sort=True):
            ordered = group.sort_values(DATE_COLUMN, kind="stable")
            returns = ordered["log_return"].to_numpy(dtype=float)
            if len(returns) < self.min_train_rows:
                continue
            if not np.isfinite(returns).all():
                raise ValueError(f"Non-finite log returns found for ticker {ticker}")

            scaled_returns = returns * RETURNS_SCALE
            fitted = arch_model(
                scaled_returns,
                mean="Zero",
                vol="GARCH",
                p=1,
                o=0,
                q=1,
                dist="t",
                rescale=False,
            ).fit(disp="off", show_warning=False)
            params = fitted.params
            omega = max(float(params["omega"]), 1e-12)
            alpha = max(float(params["alpha[1]"]), 0.0)
            beta = max(float(params["beta[1]"]), 0.0)
            total = alpha + beta
            if total >= 0.999:
                beta = max(0.0, 0.998 - alpha)

            variances = self._conditional_variances(
                scaled_returns,
                omega=omega,
                alpha=alpha,
                beta=beta,
            )
            self.parameters[str(ticker)] = GarchParameters(
                ticker=str(ticker),
                omega=omega,
                alpha=alpha,
                beta=beta,
                nu=float(params.get("nu", np.nan)),
                last_variance=float(variances[-1]),
                last_return=float(scaled_returns[-1]),
                train_end=pd.Timestamp(ordered[DATE_COLUMN].iloc[-1]),
            )
        if not self.parameters:
            raise ValueError("No ticker had enough rows to fit a GARCH model")
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        """Forecast each supplied future row without using returns after its date."""

        rows: list[dict[str, object]] = []
        for ticker, group in future.groupby(TICKER_COLUMN, sort=True):
            key = str(ticker)
            if key not in self.parameters:
                continue
            params = self.parameters[key]
            ordered = group.sort_values(DATE_COLUMN, kind="stable")
            state = params.last_variance
            previous_return = params.last_return
            for _, row in ordered.iterrows():
                # Advance the conditional variance to the current origin using
                # the previous observed return only.
                current_variance = self._next_variance(
                    state,
                    previous_return,
                    params.omega,
                    params.alpha,
                    params.beta,
                )
                current_return = float(row["log_return"]) * RETURNS_SCALE
                forecast_variances = []
                next_variance = self._next_variance(
                    current_variance,
                    current_return,
                    params.omega,
                    params.alpha,
                    params.beta,
                )
                forecast_variances.append(next_variance)
                for _ in range(1, self.horizon):
                    # Under a zero-mean model, future shocks have expected
                    # squared value zero in the conditional mean forecast.
                    next_variance = params.omega + params.beta * next_variance
                    forecast_variances.append(next_variance)
                prediction = np.sqrt(
                    TRADING_DAYS_PER_YEAR * np.mean(forecast_variances)
                ) / RETURNS_SCALE
                rows.append({
                    TICKER_COLUMN: key,
                    DATE_COLUMN: pd.Timestamp(row[DATE_COLUMN]),
                    "prediction": float(max(0.0, prediction)),
                })
                state = current_variance
                previous_return = current_return
        return pd.DataFrame(rows, columns=[TICKER_COLUMN, DATE_COLUMN, "prediction"])

    def parameter_table(self) -> pd.DataFrame:
        """Return fitted parameters for MLflow and diagnostics."""

        return pd.DataFrame([
            {
                "ticker": value.ticker,
                "omega": value.omega,
                "alpha": value.alpha,
                "beta": value.beta,
                "nu": value.nu,
                "train_end": value.train_end,
            }
            for value in self.parameters.values()
        ]).sort_values("ticker", kind="stable")

    @staticmethod
    def _conditional_variances(
        returns: np.ndarray,
        *,
        omega: float,
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        """Compute the causal conditional variance recursion."""

        variances = np.empty(len(returns), dtype=float)
        variances[0] = max(float(np.var(returns)), omega / max(1.0 - alpha - beta, 1e-6))
        for index in range(1, len(returns)):
            variances[index] = GarchForecaster._next_variance(
                variances[index - 1],
                returns[index - 1],
                omega,
                alpha,
                beta,
            )
        return variances

    @staticmethod
    def _next_variance(
        variance: float,
        observed_return: float,
        omega: float,
        alpha: float,
        beta: float,
    ) -> float:
        """Advance GARCH variance using one observed scaled return."""

        return max(1e-12, omega + alpha * observed_return**2 + beta * variance)
