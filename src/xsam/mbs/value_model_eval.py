import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from xsam.constants import DATE_FORMAT
from xsam.mbs.mock_data import generate_historical_data
from xsam.mbs.value_model import JointReversionModel
from xsam import save


def run_value_model(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    sigma_r_data: pd.Series,
    nu_r_data: pd.Series,
    enable_spread_cvx: bool,
    enable_rate_vol: bool,
    enable_local_vol: bool,
    enable_mle: bool,
    estimation_freq: str,
    simulation_freq: str,
    simulation_steps: int,
    num_paths: int,
    seed: int,
    model_param_overrides: dict[str, float] = {},
    simulation_param_overrides: dict[str, float] = {},
    verbose: bool = False,
    run_simulation: bool = True,
    run_adf: bool = False,
) -> tuple[dict[str, pd.DataFrame], JointReversionModel]:
    """Run the MBS valuation model. The function performs the following steps:
    1. Resample historical data to the estimation frequency.
    2. Initialise the model.
    3. Perform stationarity tests for OAS and Convexity.
    4. Estimate model parameters using OLS.
    5. Estimate model parameters using MLE if enabled.
    6. Perform Monte Carlo simulation.

    Args:
        oas_data (pd.Series): Historical data for OAS.
        cvx_data (pd.Series): Historical data for Convexity.
        sigma_r_data (pd.Series): Historical data for Rates Volatility.
        nu_r_data (pd.Series): Historical data for Rates Volatility of Volatility.
        enable_spread_cvx (bool): Flag to enable Convexity.
        enable_rate_vol (bool): Flag to enable Volatility.
        enable_local_vol (bool): Flag to enable Local Volatility.
        enable_mle (bool): Flag to enable MLE estimation.
        estimation_freq (str): Frequency for estimation.
        simulation_freq (str): Frequency for Monte Carlo simulation.
        simulation_steps (int): Number of simulation steps.
        num_paths (int): Number of Monte Carlo paths.
        seed (int): Seed for reproducibility.
        model_param_overrides (dict[str, float], optional): Model parameter overrides. Defaults to {}.
        simulation_param_overrides (dict[str, float], optional): Simulation parameter overrides. Defaults to {}.
        verbose (bool, optional): Flag to enable verbose output. Defaults to False.
        run_simulation (bool, optional): Flag to enable Monte Carlo simulation. Defaults to True.
        run_adf (bool, optional): Flag to enable ADF stationarity tests. Defaults to False.
    """
    # Resample historical data to the estimation frequency
    oas_data = oas_data.asfreq("D").ffill().asfreq(estimation_freq)
    cvx_data = cvx_data.asfreq("D").ffill().asfreq(estimation_freq)
    sigma_r_data = sigma_r_data.asfreq("D").ffill().asfreq(estimation_freq)
    nu_r_data = nu_r_data.asfreq("D").ffill().asfreq(estimation_freq)

    # Monte Carlo simulation parameters
    simulation_params = {
        "S_OAS_init": float(oas_data.iloc[-1]),
        "C_init": float(cvx_data.iloc[-1]),
        "S_OAS_inf": float(np.mean(oas_data)),
        "C_inf": float(np.mean(cvx_data)),
        "sigma_r_forward": float(sigma_r_data.iloc[-1]),
        "nu_r_forward": float(nu_r_data.iloc[-1]),
    }

    # Override simulation parameters if specified
    simulation_params.update(simulation_param_overrides)

    # Derive time steps for simulation in both estimation and simulation frequencies
    start_date = oas_data.index[-1]
    simulation_dates = pd.date_range(start=start_date, periods=simulation_steps, freq=simulation_freq)
    simulation_dt = 1 / simulation_steps
    simulation_dates_estimation_freq = pd.date_range(
        start=simulation_dates[0], end=simulation_dates[-1], freq=estimation_freq
    )
    simulation_steps_estimation_freq = len(simulation_dates_estimation_freq)
    simulation_dt_estimation_freq = 1 / simulation_steps_estimation_freq
    if verbose:
        print("\nEstimation config:")
        print(f"Steps: {simulation_steps_estimation_freq}")
        print(f"Frequency: '{estimation_freq}'")
        est_start = oas_data.index[0].strftime(DATE_FORMAT)
        est_end = oas_data.index[-1].strftime(DATE_FORMAT)
        print(f"Range: from {est_start} to {est_end}")
        print("\nSimulation config:")
        print(f"Steps: {simulation_steps}")
        print(f"Frequency: '{simulation_freq}'")
        sim_start = simulation_dates[0].strftime(DATE_FORMAT)
        sim_end = simulation_dates[-1].strftime(DATE_FORMAT)
        print(f"Range: from {sim_start} to {sim_end}")

    # Assert that forward data series have the simulation frequency
    if isinstance(simulation_params["sigma_r_forward"], pd.Series):
        assert simulation_params["sigma_r_forward"].index.freq == simulation_dates.freq
    if isinstance(simulation_params["nu_r_forward"], pd.Series):
        assert simulation_params["nu_r_forward"].index.freq == simulation_dates.freq

    # Initialise the model
    model = JointReversionModel(
        dt=simulation_dt,
        enable_spread_cvx=enable_spread_cvx,
        enable_rate_vol=enable_rate_vol,
        enable_local_vol=enable_local_vol,
    )

    # Stationarity tests
    if run_adf:
        adf_OAS, adf_C = model.stationarity_tests_adf(oas_data, cvx_data, verbose=verbose)

    # Estimate model parameters using OLS
    model.estimate_parameters_ols(
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        simulation_params["S_OAS_inf"],
        simulation_params["C_inf"],
        simulation_steps_estimation_freq,
        verbose=verbose,
    )

    if enable_mle:
        # Estimate model parameters using MLE
        model.estimate_parameters_mle(
            oas_data,
            cvx_data,
            sigma_r_data,
            nu_r_data,
            simulation_params["S_OAS_inf"],
            simulation_params["C_inf"],
            simulation_dt_estimation_freq,
            verbose=verbose,
        )

    # Override model parameters if specified
    model.update_parameters(**model_param_overrides)

    paths = None
    if run_simulation:
        # Perform Monte Carlo simulation
        paths = model.monte_carlo_simulation(
            **simulation_params,
            simulation_dates=simulation_dates,
            num_paths=num_paths,
            seed=seed,
            verbose=verbose,
        )

    return paths, model


def plot_value_model(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    paths: dict[str, pd.DataFrame],
    S_OAS_inf: float,
    C_inf: float,
) -> plt.Figure:
    """Plot historical data and Monte Carlo paths for OAS and Convexity."""
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    axs = axs.flatten()

    def plot_series(ax, data, paths, inf, color, title):
        combined = pd.concat([data, paths], axis=1)
        ax.plot(combined.iloc[:, 0], color=color, label=title)
        ax.plot(combined.iloc[:, 1:], color=color, alpha=0.1)
        ax.plot(
            combined.iloc[:, 1:].mean(axis=1),
            color=color,
            linestyle=":",
            label=f"Projected {title}",
        )
        ax.axhline(y=inf, color=color, linestyle="--", label="Reversion Level")
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        ax.set_title(f"Monte Carlo Simulation of {title}")
        ax.set_xlabel("")
        ax.set_ylabel(title)
        ax.legend()

    plot_series(axs[0], oas_data, paths["oas"], S_OAS_inf, "darkblue", "OAS")
    plot_series(axs[1], oas_data.iloc[-len(paths["oas"]) * 3 :], paths["oas"], S_OAS_inf, "darkblue", "OAS - Zoomed In")
    plot_series(axs[2], cvx_data, paths["cvx"], C_inf, "darkgreen", "Convexity")
    plot_series(
        axs[3], cvx_data.iloc[-len(paths["cvx"]) * 3 :], paths["cvx"], C_inf, "darkgreen", "Convexity - Zoomed In"
    )

    fig.tight_layout()
    return fig


def run_model_vs_actual(
    model: JointReversionModel,
    oas_data: pd.Series,
    cvx_data: pd.Series,
    sigma_r_data: pd.Series,
    nu_r_data: pd.Series,
    simulation_freq: str,
    simulation_steps: int,
    num_paths: int,
    seed: int,
    step_interval: int = 1,
    verbose: bool = False,
) -> dict:
    """Compare model expected vs actual change in OAS and Convexity.

    Args:
        model (JointReversionModel): Model object.
        oas_data (pd.Series): Historical data for OAS.
        cvx_data (pd.Series): Historical data for Convexity.
        sigma_r_data (pd.Series): Historical data for Rates Volatility.
        nu_r_data (pd.Series): Historical data for Rates Volatility of Volatility.
        simulation_freq (str): Frequency for Monte Carlo simulation.
        simulation_steps (int): Number of simulation steps.
        num_paths (int): Number of Monte Carlo paths.
        seed (int): Seed for reproducibility.
        step_interval (int, optional): Interval for simulation steps. Defaults to 1.
        verbose (bool, optional): Flag to enable verbose output. Defaults to False.

    Returns:
        dict: Contains DataFrames for OAS and Convexity expected vs actual change.
    """
    S_OAS_inf = float(np.mean(oas_data))
    C_inf = float(np.mean(cvx_data))

    oas_expected_paths = []
    cvx_expected_paths = []

    oas_expected = []
    cvx_expected = []

    oas_actual_start = []
    cvx_actual_start = []

    oas_actual_end = []
    cvx_actual_end = []

    oas_expected_change = []
    cvx_expected_change = []

    oas_actual_change = []
    cvx_actual_change = []

    # Get the last start date as a location in oas_data.index
    final_simulation_dates = pd.date_range(end=oas_data.index[-1], periods=simulation_steps, freq=simulation_freq)
    final_start = oas_data.index.get_loc(final_simulation_dates[0])

    # Loop through simulation windows
    for start in oas_data.index[:final_start:step_interval]:
        if verbose:
            print(start.strftime(DATE_FORMAT))

        simulation_dates = pd.date_range(start=start, periods=simulation_steps, freq=simulation_freq)
        end = simulation_dates[-1]

        paths = model.monte_carlo_simulation(
            S_OAS_init=oas_data.loc[start],
            C_init=cvx_data.loc[start],
            S_OAS_inf=S_OAS_inf,
            C_inf=C_inf,
            sigma_r_forward=sigma_r_data.loc[start],
            nu_r_forward=nu_r_data.loc[start],
            simulation_dates=simulation_dates,
            num_paths=num_paths,
            seed=seed,
            verbose=False,
        )
        oas_expected_path = paths["oas"].mean(axis=1)
        cvx_expected_path = paths["cvx"].mean(axis=1)

        oas_expected_paths.append(oas_expected_path)
        cvx_expected_paths.append(cvx_expected_path)

        oas_expected.append((end, float(oas_expected_path.iloc[-1])))
        cvx_expected.append((end, float(cvx_expected_path.iloc[-1])))

        oas_actual_start.append((start, float(oas_data.loc[start])))
        cvx_actual_start.append((start, float(cvx_data.loc[start])))

        oas_actual_end.append((end, float(oas_data.loc[end])))
        cvx_actual_end.append((end, float(cvx_data.loc[end])))

        oas_expected_change.append((end, float(oas_expected_path.iloc[-1]) - float(oas_data.loc[start])))
        cvx_expected_change.append((end, float(cvx_expected_path.iloc[-1]) - float(cvx_data.loc[start])))

        oas_actual_change.append((end, float(oas_data.loc[end]) - float(oas_data.loc[start])))
        cvx_actual_change.append((end, float(cvx_data.loc[end]) - float(cvx_data.loc[start])))

    # Collate results into DataFrames
    oas_expected = pd.DataFrame(oas_expected, columns=["Date", "Expected End"]).set_index("Date")
    cvx_expected = pd.DataFrame(cvx_expected, columns=["Date", "Expected End"]).set_index("Date")

    oas_actual_start = pd.DataFrame(oas_actual_start, columns=["Date", "Actual Start"]).set_index("Date")
    cvx_actual_start = pd.DataFrame(cvx_actual_start, columns=["Date", "Actual Start"]).set_index("Date")

    oas_actual_end = pd.DataFrame(oas_actual_end, columns=["Date", "Actual End"]).set_index("Date")
    cvx_actual_end = pd.DataFrame(cvx_actual_end, columns=["Date", "Actual End"]).set_index("Date")

    oas_expected_change = pd.DataFrame(oas_expected_change, columns=["Date", "Expected Change"]).set_index("Date")
    cvx_expected_change = pd.DataFrame(cvx_expected_change, columns=["Date", "Expected Change"]).set_index("Date")

    oas_actual_change = pd.DataFrame(oas_actual_change, columns=["Date", "Actual Change"]).set_index("Date")
    cvx_actual_change = pd.DataFrame(cvx_actual_change, columns=["Date", "Actual Change"]).set_index("Date")

    oas = [
        # oas_actual_start,
        oas_actual_end,
        oas_expected,
        oas_actual_change,
        oas_expected_change,
    ]
    oas = [ser[~ser.index.duplicated(keep="last")] for ser in oas]
    oas = pd.concat(oas, axis=1)

    cvx = [
        # cvx_actual_start,
        cvx_actual_end,
        cvx_expected,
        cvx_actual_change,
        cvx_expected_change,
    ]
    cvx = [ser[~ser.index.duplicated(keep="last")] for ser in cvx]
    cvx = pd.concat(cvx, axis=1)

    output = {
        "OAS": oas,
        "Convexity": cvx,
        "OAS Expected Paths": oas_expected_paths,
        "Convexity Expected Paths": cvx_expected_paths,
        "Model": model,
    }

    return output


def plot_model_vs_actual(oas: pd.DataFrame, cvx: pd.DataFrame, num_predictors) -> plt.Figure:
    """Plot model expected vs actual change in OAS and Convexity.

    Args:
        oas (pd.DataFrame): OAS DataFrame with actual and expected change.
        cvx (pd.DataFrame): Convexity DataFrame with actual and expected change.

    Returns:
        plt.Figure: Figure object with the plot.
    """

    def plot_scatter(ax, data, title):
        actual_change = data["Actual Change"]
        expected_change = data["Expected Change"]
        ax.scatter(actual_change, expected_change)
        X_with_const = sm.add_constant(actual_change)
        model = sm.OLS(expected_change, X_with_const).fit()
        ax.plot(actual_change, model.predict(X_with_const), color="red")
        ax.text(0.05, 0.95, f"y = {model.params[0]:.2f} + {model.params[1]:.2f}x", transform=ax.transAxes, va="top")
        r2 = model.rsquared
        n = len(actual_change)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - num_predictors - 1)
        ax.text(0.05, 0.9, f"Adj R\u00b2 = {adj_r2:.2f}", transform=ax.transAxes, va="top")
        ax.plot(actual_change, actual_change, color="lightgray")
        ax.set_title(title)
        ax.set_xlabel("Actual Change")
        ax.set_ylabel("Expected Change")

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plot_scatter(axs[0], oas.loc[:, ["Actual Change", "Expected Change"]].dropna(), "OAS")
    plot_scatter(axs[1], cvx.loc[:, ["Actual Change", "Expected Change"]].dropna(), "Convexity")
    fig.suptitle("Model Expected vs Actual Change in OAS and Convexity")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def evaluate_frequency(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    sigma_r_data: pd.Series,
    nu_r_data: pd.Series,
    complexity_params: dict,
    simulation_dates: pd.DatetimeIndex,
    num_paths: int,
    seed: int,
    step_interval: int = 1,
    verbose: bool = False,
) -> dict:
    """Evaluate model on different frequencies."""
    if verbose:
        print("\nEvaluating model on different frequencies...")

    frequencies = {"daily": "B", "weekly": "W-FRI", "monthly": "BME"}
    output = {}

    for freq_name, freq in frequencies.items():
        if verbose:
            print(f"\nFrequency: {freq_name}")
        resampled_dates = pd.date_range(start=simulation_dates[0], end=simulation_dates[-1], freq=freq)
        resampled_steps = len(resampled_dates)
        _, model = run_value_model(
            oas_data,
            cvx_data,
            sigma_r_data,
            nu_r_data,
            enable_spread_cvx=complexity_params["enable_spread_cvx"],
            enable_rate_vol=complexity_params["enable_rate_vol"],
            enable_local_vol=complexity_params["enable_local_vol"],
            enable_mle=False,
            estimation_freq=freq,
            simulation_freq=freq,
            simulation_steps=resampled_steps,
            num_paths=num_paths,
            seed=seed,
            verbose=verbose,
            run_simulation=False,
        )

        results = run_model_vs_actual(
            model,
            oas_data,
            cvx_data,
            sigma_r_data,
            nu_r_data,
            freq,
            resampled_steps,
            num_paths,
            seed,
            step_interval=step_interval,
            verbose=verbose,
        )
        output[freq_name] = results

    return output


def evaluate_complexity(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    sigma_r_data: pd.Series,
    nu_r_data: pd.Series,
    simulation_dates: pd.DatetimeIndex,
    num_paths: int,
    seed: int,
    step_interval: int = 1,
    verbose: bool = False,
) -> dict:
    """Evaluate model with different complexities."""
    if verbose:
        print("\nEvaluating model with different complexities...")

    estimation_freq = "B"
    simulation_freq = "W-FRI"
    complexities = {
        "mean reversion only": {
            "enable_spread_cvx": False,
            "enable_rate_vol": False,
            "enable_local_vol": False,
        },
        "spread cvx": {
            "enable_spread_cvx": True,
            "enable_rate_vol": False,
            "enable_local_vol": False,
        },
        "rate vol": {
            "enable_spread_cvx": False,
            "enable_rate_vol": True,
            "enable_local_vol": False,
        },
        "local vol": {
            "enable_spread_cvx": False,
            "enable_rate_vol": False,
            "enable_local_vol": True,
        },
        "spread cvx + rate vol": {
            "enable_spread_cvx": True,
            "enable_rate_vol": True,
            "enable_local_vol": False,
        },
        "spread cvx + local vol": {
            "enable_spread_cvx": True,
            "enable_rate_vol": False,
            "enable_local_vol": True,
        },
        "rate vol + local vol": {
            "enable_spread_cvx": False,
            "enable_rate_vol": True,
            "enable_local_vol": True,
        },
        "full model": {
            "enable_spread_cvx": True,
            "enable_rate_vol": True,
            "enable_local_vol": True,
        },
    }
    output = {}

    resampled_dates = pd.date_range(start=simulation_dates[0], end=simulation_dates[-1], freq=simulation_freq)
    resampled_steps = len(resampled_dates)

    for complexity_name, complexity_params in complexities.items():
        if verbose:
            print(f"\nComplexity: {complexity_name}")
        _, model = run_value_model(
            oas_data,
            cvx_data,
            sigma_r_data,
            nu_r_data,
            enable_spread_cvx=complexity_params["enable_spread_cvx"],
            enable_rate_vol=complexity_params["enable_rate_vol"],
            enable_local_vol=complexity_params["enable_local_vol"],
            enable_mle=False,
            estimation_freq=estimation_freq,
            simulation_freq=simulation_freq,
            simulation_steps=resampled_steps,
            num_paths=num_paths,
            seed=seed,
            verbose=verbose,
            run_simulation=False,
        )

        results = run_model_vs_actual(
            model,
            oas_data,
            cvx_data,
            sigma_r_data,
            nu_r_data,
            simulation_freq,
            resampled_steps,
            num_paths,
            seed,
            step_interval=step_interval,
            verbose=verbose,
        )
        output[complexity_name] = results

    return output


def evaluate_estimation(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    sigma_r_data: pd.Series,
    nu_r_data: pd.Series,
    complexity_params: dict,
    simulation_dates: pd.DatetimeIndex,
    num_paths: int,
    seed: int,
    step_interval: int = 1,
    verbose: bool = False,
) -> dict:
    """Evaluate model using OLS vs OLS + MLE."""
    if verbose:
        print("\nEvaluating model using different estimation methods...")

    estimation_freq = "B"
    simulation_freq = "W-FRI"
    output = {}

    resampled_dates = pd.date_range(start=simulation_dates[0], end=simulation_dates[-1], freq=simulation_freq)
    resampled_steps = len(resampled_dates)

    if verbose:
        print("\nEstimation: OLS")
    _, model = run_value_model(
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        enable_spread_cvx=complexity_params["enable_spread_cvx"],
        enable_rate_vol=complexity_params["enable_rate_vol"],
        enable_local_vol=complexity_params["enable_local_vol"],
        enable_mle=False,
        estimation_freq=estimation_freq,
        simulation_freq=simulation_freq,
        simulation_steps=resampled_steps,
        num_paths=num_paths,
        seed=seed,
        verbose=verbose,
        run_simulation=False,
    )

    results = run_model_vs_actual(
        model,
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        simulation_freq,
        resampled_steps,
        num_paths,
        seed,
        step_interval=step_interval,
        verbose=verbose,
    )
    output["ols"] = results

    if verbose:
        print("\nEstimation: OLS + MLE")
    _, model = run_value_model(
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        enable_spread_cvx=complexity_params["enable_spread_cvx"],
        enable_rate_vol=complexity_params["enable_rate_vol"],
        enable_local_vol=complexity_params["enable_local_vol"],
        enable_mle=True,
        estimation_freq=estimation_freq,
        simulation_freq=simulation_freq,
        simulation_steps=resampled_steps,
        num_paths=num_paths,
        seed=seed,
        verbose=verbose,
        run_simulation=False,
    )

    results = run_model_vs_actual(
        model,
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        simulation_freq,
        resampled_steps,
        num_paths,
        seed,
        step_interval=step_interval,
        verbose=verbose,
    )
    output["mle"] = results

    return output


def evaluation_criteria(expected: pd.Series, actual: pd.Series, model: JointReversionModel) -> dict:
    """Evaluate model performance using different criteria.

    MSE: Mean Squared Error
    RMSE: Root Mean Squared Error
    MAE: Mean Absolute Error
    R2: R-squared
    Adj R2: Adjusted R-squared

    Args:
        expected (pd.Series): Expected data.
        actual (pd.Series): Actual data.
        model (JointReversionModel): Model object.

    Returns:
        dict: Dictionary with evaluation criteria.
    """
    actual = actual.reindex(expected.index)
    try:
        X_with_const = sm.add_constant(actual)
        ols_model = sm.OLS(expected, X_with_const).fit()
        mse = np.mean((expected - actual) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(expected - actual))
        r2 = ols_model.rsquared
        # Adjusted R-squared using the number of predictors from the underlying model
        n = len(actual)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - model.get_num_predictors() - 1)
    except ValueError:
        mse, rmse, mae, r2, adj_r2 = np.nan, np.nan, np.nan, np.nan, np.nan

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "Adj R2": adj_r2}


def summarize_ols(model: JointReversionModel) -> dict:
    """Summarize OLS model results."""
    ols = model.ols.copy()
    ols.pop("model_OAS", None)
    ols.pop("model_C", None)
    ols.pop("model_sigma_O", None)
    return ols


def evaluate_value_model(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    sigma_r_data: pd.Series,
    nu_r_data: pd.Series,
    complexity_params: dict = None,
    simulation_days: int = 252,
    num_paths: int = 100,
    seed: int = 42,
    step_interval: int = 1,
    verbose: bool = False,
):
    """Evaluate the model with different frequencies, complexities, and estimation methods."""
    complexity_params = (
        {
            "enable_spread_cvx": True,
            "enable_rate_vol": True,
            "enable_local_vol": True,
        }
        if complexity_params is None
        else complexity_params
    )

    # Make sure all data series are in B frequency
    oas_data = oas_data.asfreq("D").ffill().asfreq("B")
    cvx_data = cvx_data.asfreq("D").ffill().asfreq("B")
    sigma_r_data = sigma_r_data.asfreq("D").ffill().asfreq("B")
    nu_r_data = nu_r_data.asfreq("D").ffill().asfreq("B")

    # Generate reference range of dates for simulation
    simulation_dates = pd.date_range(end=oas_data.index[-1], periods=simulation_days, freq="B")

    frequency_results = evaluate_frequency(
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        complexity_params,
        simulation_dates,
        num_paths,
        seed,
        step_interval,
        verbose,
    )

    complexity_results = evaluate_complexity(
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        simulation_dates,
        num_paths,
        seed,
        step_interval,
        verbose,
    )

    estimation_results = evaluate_estimation(
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        complexity_params,
        simulation_dates,
        num_paths,
        seed,
        step_interval,
        verbose,
    )

    # Evaluate criteria for each result set
    cols = ["Actual Change", "Expected Change"]

    frequency_oas = {k: (v["OAS"].loc[:, cols].dropna(), v["Model"]) for k, v in frequency_results.items()}
    complexity_oas = {k: (v["OAS"].loc[:, cols].dropna(), v["Model"]) for k, v in complexity_results.items()}
    estimation_oas = {k: (v["OAS"].loc[:, cols].dropna(), v["Model"]) for k, v in estimation_results.items()}
    oas_evaluation = {
        "frequency": {
            k: evaluation_criteria(v["Expected Change"], v["Actual Change"], m) for k, (v, m) in frequency_oas.items()
        },
        "complexity": {
            k: evaluation_criteria(v["Expected Change"], v["Actual Change"], m) for k, (v, m) in complexity_oas.items()
        },
        "estimation": {
            k: evaluation_criteria(v["Expected Change"], v["Actual Change"], m) for k, (v, m) in estimation_oas.items()
        },
    }

    frequency_cvx = {k: (v["Convexity"].loc[:, cols].dropna(), v["Model"]) for k, v in frequency_results.items()}
    complexity_cvx = {k: (v["Convexity"].loc[:, cols].dropna(), v["Model"]) for k, v in complexity_results.items()}
    estimation_cvx = {k: (v["Convexity"].loc[:, cols].dropna(), v["Model"]) for k, v in estimation_results.items()}
    cvx_evaluation = {
        "frequency": {
            k: evaluation_criteria(v["Expected Change"], v["Actual Change"], m) for k, (v, m) in frequency_cvx.items()
        },
        "complexity": {
            k: evaluation_criteria(v["Expected Change"], v["Actual Change"], m) for k, (v, m) in complexity_cvx.items()
        },
        "estimation": {
            k: evaluation_criteria(v["Expected Change"], v["Actual Change"], m) for k, (v, m) in estimation_cvx.items()
        },
    }

    oas_ols_summary = {
        "frequency": {k: summarize_ols(v["Model"]) for k, v in frequency_results.items()},
        "complexity": {k: summarize_ols(v["Model"]) for k, v in complexity_results.items()},
        "estimation": {k: summarize_ols(v["Model"]) for k, v in estimation_results.items()},
    }

    cvx_ols_summary = {
        "frequency": {k: summarize_ols(v["Model"]) for k, v in frequency_results.items()},
        "complexity": {k: summarize_ols(v["Model"]) for k, v in complexity_results.items()},
        "estimation": {k: summarize_ols(v["Model"]) for k, v in estimation_results.items()},
    }

    def flatten_evaluation(evaluation_dict):
        """Flatten the evaluation dictionary into a DataFrame."""
        flattened_data = []
        for topic, cases in evaluation_dict.items():
            for case, metrics in cases.items():
                flattened_entry = {"Topic": topic, "Case": case}
                flattened_entry.update(metrics)
                flattened_data.append(flattened_entry)
        return pd.DataFrame(flattened_data)

    return {
        "OAS Eval": flatten_evaluation(oas_evaluation),
        "Convexity Eval": flatten_evaluation(cvx_evaluation),
        "OAS OLS": flatten_evaluation(oas_ols_summary),
        "Convexity OLS": flatten_evaluation(cvx_ols_summary),
        "Frequency Results": frequency_results,
        "Complexity Results": complexity_results,
        "Estimation Results": estimation_results,
    }


if __name__ == "__main__":
    # Set seed for reproducibility
    seed = 42

    # Define training data parameters
    zv_params = {"mu": 50, "theta": 0.01, "sigma": 20, "X0": 80}
    oas_params = {"mu": 30, "theta": 0.02, "sigma": 10, "X0": 20}
    zv_oas_rho = 0.8
    sigma_r_params = {"mu": 20, "theta": 0.01, "sigma": 10, "X0": 20}
    nu_r_params = {"mu": 10, "theta": 0.01, "sigma": 10, "X0": 10}

    # Mock data simulation parameters
    train_start_date = "2013-01-01"
    train_end_date = (pd.Timestamp.today() - pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    historical_freq = "B"
    historical_dates = pd.date_range(start=train_start_date, end=train_end_date, freq=historical_freq)

    # Generate training data
    zv_hist, oas_hist, sigma_r_hist, nu_r_hist = generate_historical_data(
        zv_params,
        oas_params,
        zv_oas_rho,
        sigma_r_params,
        nu_r_params,
        historical_dates,
        seed,
    )
    cvx_hist = zv_hist - oas_hist

    # Monte Carlo simulation parameters
    oas_data = oas_hist.copy()
    cvx_data = cvx_hist.copy()
    sigma_r_data = sigma_r_hist.copy()
    nu_r_data = nu_r_hist.copy()

    # Perform model evaluation
    results = evaluate_value_model(
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        seed=seed,
        step_interval=252,
        verbose=True,
    )

    print("\nModel evaluation results:")
    print(f"OAS evaluation:\n{results['OAS Eval']}")
    print(f"CVX evaluation:\n{results['Convexity Eval']}")
    print("\nModel evaluation complete.")
