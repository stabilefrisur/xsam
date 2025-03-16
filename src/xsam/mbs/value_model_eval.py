import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xsam.constants import DATE_FORMAT
from xsam.mbs.mock_data import generate_historical_data
from xsam.mbs.value_model import JointReversionModel


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
    simulation_dates = pd.date_range(
        start=start_date, periods=simulation_steps, freq=simulation_freq
    )
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
        adf_OAS, adf_C = model.stationarity_tests_adf(
            oas_data, cvx_data, verbose=verbose
        )

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
    """Plot historical data and Monte Carlo paths for OAS and Convexity.

    Args:
        oas_data (pd.Series): Historical data for OAS.
        cvx_data (pd.Series): Historical data for Convexity.
        paths (dict[str, pd.DataFrame]): Monte Carlo paths for OAS and Convexity.
        S_OAS_inf (float): Reversion level for OAS.
        C_inf (float): Reversion level for Convexity.

    Returns:
        plt.Figure: Figure object with the plot.
    """
    # Plot historical data and Monte Carlo paths
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    axs = axs.flatten()

    # Plot OAS
    paths_OAS = paths["oas"]
    OAS = pd.concat([oas_data, paths_OAS], axis=1)
    axs[0].plot(OAS.iloc[:, 0], color="darkblue", label="OAS")
    axs[0].plot(OAS.iloc[:, 1:], color="lightblue", alpha=0.1)
    axs[0].plot(
        OAS.iloc[:, 1:].mean(axis=1),
        color="darkblue",
        linestyle=":",
        label="Projected OAS",
        alpha=1.0,
    )
    axs[0].axhline(
        y=S_OAS_inf, color="darkblue", linestyle="--", label="Reversion Level"
    )
    axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    axs[0].set_title("Monte Carlo Simulation of OAS")
    axs[0].set_xlabel("")
    axs[0].set_ylabel("OAS")
    axs[0].legend()

    # Plot OAS trailing 3x simulation period
    OAS_trail = OAS.iloc[-len(paths_OAS) * 3 :]
    axs[1].plot(OAS_trail.iloc[:, 0], color="darkblue", label="OAS")
    axs[1].plot(OAS_trail.iloc[:, 1:], color="lightblue", alpha=0.1)
    axs[1].plot(
        OAS_trail.iloc[:, 1:].mean(axis=1),
        color="darkblue",
        linestyle=":",
        label="Projected OAS",
        alpha=1.0,
    )
    axs[1].axhline(
        y=S_OAS_inf, color="darkblue", linestyle="--", label="Reversion Level"
    )
    axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    axs[1].set_title("Monte Carlo Simulation of OAS - Zoomed In")
    axs[1].set_xlabel("")
    axs[1].set_ylabel("OAS")
    axs[1].legend()

    # Plot Convexity
    paths_C = paths["cvx"]
    C = pd.concat([cvx_data, paths_C], axis=1)
    axs[2].plot(C.iloc[:, 0], color="darkgreen", label="Convexity")
    axs[2].plot(C.iloc[:, 1:], color="lightgreen", alpha=0.1)
    axs[2].plot(
        C.iloc[:, 1:].mean(axis=1),
        color="darkgreen",
        linestyle=":",
        label="Projected Convexity",
    )
    axs[2].axhline(y=C_inf, color="darkgreen", linestyle="--", label="Reversion Level")
    axs[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    axs[2].set_title("Monte Carlo Simulation of Convexity")
    axs[2].set_xlabel("")
    axs[2].set_ylabel("Convexity")
    axs[2].legend()

    # Plot Convexity trailing 3x simulation period
    C_trail = C.iloc[-len(paths_C) * 3 :]
    axs[3].plot(C_trail.iloc[:, 0], color="darkgreen", label="Convexity")
    axs[3].plot(C_trail.iloc[:, 1:], color="lightgreen", alpha=0.1)
    axs[3].plot(
        C_trail.iloc[:, 1:].mean(axis=1),
        color="darkgreen",
        linestyle=":",
        label="Projected Convexity",
    )
    axs[3].axhline(y=C_inf, color="darkgreen", linestyle="--", label="Reversion Level")
    axs[3].xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    axs[3].set_title("Monte Carlo Simulation of Convexity - Zoomed In")
    axs[3].set_xlabel("")
    axs[3].set_ylabel("Convexity")
    axs[3].legend()

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
    final_simulation_dates = pd.date_range(
        end=oas_data.index[-1], periods=simulation_steps, freq=simulation_freq
    )
    final_start = oas_data.index.get_loc(final_simulation_dates[0])

    # Loop through simulation windows
    for start in oas_data.index[:final_start:step_interval]:
        if verbose:
            print(start.strftime(DATE_FORMAT))

        simulation_dates = pd.date_range(
            start=start, periods=simulation_steps, freq=simulation_freq
        )
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

        oas_expected_change.append(
            (end, float(oas_expected_path.iloc[-1]) - float(oas_data.loc[start]))
        )
        cvx_expected_change.append(
            (end, float(cvx_expected_path.iloc[-1]) - float(cvx_data.loc[start]))
        )

        oas_actual_change.append(
            (end, float(oas_data.loc[end]) - float(oas_data.loc[start]))
        )
        cvx_actual_change.append(
            (end, float(cvx_data.loc[end]) - float(cvx_data.loc[start]))
        )

    # Collate results into DataFrames
    oas_expected = pd.DataFrame(
        oas_expected, columns=["Date", "Expected End"]
    ).set_index("Date")
    cvx_expected = pd.DataFrame(
        cvx_expected, columns=["Date", "Expected End"]
    ).set_index("Date")

    oas_actual_start = pd.DataFrame(
        oas_actual_start, columns=["Date", "Actual Start"]
    ).set_index("Date")
    cvx_actual_start = pd.DataFrame(
        cvx_actual_start, columns=["Date", "Actual Start"]
    ).set_index("Date")

    oas_actual_end = pd.DataFrame(
        oas_actual_end, columns=["Date", "Actual End"]
    ).set_index("Date")
    cvx_actual_end = pd.DataFrame(
        cvx_actual_end, columns=["Date", "Actual End"]
    ).set_index("Date")

    oas_expected_change = pd.DataFrame(
        oas_expected_change, columns=["Date", "Expected Change"]
    ).set_index("Date")
    cvx_expected_change = pd.DataFrame(
        cvx_expected_change, columns=["Date", "Expected Change"]
    ).set_index("Date")

    oas_actual_change = pd.DataFrame(
        oas_actual_change, columns=["Date", "Actual Change"]
    ).set_index("Date")
    cvx_actual_change = pd.DataFrame(
        cvx_actual_change, columns=["Date", "Actual Change"]
    ).set_index("Date")

    oas = pd.concat(
        [
            oas_actual_start,
            oas_actual_end,
            oas_expected,
            oas_actual_change,
            oas_expected_change,
        ],
        axis=1,
    )
    cvx = pd.concat(
        [
            cvx_actual_start,
            cvx_actual_end,
            cvx_expected,
            cvx_actual_change,
            cvx_expected_change,
        ],
        axis=1,
    )

    output = {
        "OAS": oas,
        "Convexity": cvx,
        "OAS Expected Paths": oas_expected_paths,
        "Convexity Expected Paths": cvx_expected_paths,
        "Model": model,
    }

    return output


def plot_model_vs_actual(
    oas: pd.DataFrame,
    cvx: pd.DataFrame,
    num_predictors,
) -> plt.Figure:
    """Plot model expected vs actual change in OAS and Convexity.

    Args:
        oas (pd.DataFrame): OAS DataFrame with actual and expected change.
        cvx (pd.DataFrame): Convexity DataFrame with actual and expected change.

    Returns:
        plt.Figure: Figure object with the plot.
    """

    # Plot OAS and Convexity expected vs actual change in scatter plot with regression line
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    axs = axs.flatten()

    oas_plot = oas.loc[:, ["Actual Change", "Expected Change"]].dropna()
    axs[0].scatter(oas_plot["Actual Change"], oas_plot["Expected Change"])
    # Add regression line
    x = oas_plot["Actual Change"]
    y = oas_plot["Expected Change"]
    m, b = np.polyfit(x, y, 1)
    axs[0].plot(x, m * x + b, color="red")
    # Print regression line equation
    axs[0].text(
        0.05, 0.95, f"y = {b:.2f} + {m:.2f}x", transform=axs[0].transAxes, va="top"
    )
    # Add R^2 to plot
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    n = len(x)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - num_predictors - 1)
    axs[0].text(
        0.05, 0.9, f"Adj R\u00b2 = {adj_r2:.2f}", transform=axs[0].transAxes, va="top"
    )
    axs[0].set_title("OAS Expected vs Actual Change")
    axs[0].set_xlabel("Actual Change")
    axs[0].set_ylabel("Expected Change")

    cvx_plot = cvx.loc[:, ["Actual Change", "Expected Change"]].dropna()
    axs[1].scatter(cvx_plot["Actual Change"], cvx_plot["Expected Change"])
    # Add regression line
    x = cvx_plot["Actual Change"]
    y = cvx_plot["Expected Change"]
    m, b = np.polyfit(x, y, 1)
    axs[1].plot(x, m * x + b, color="red")
    # Print regression line equation in the top left corner
    axs[1].text(
        0.05, 0.95, f"y = {b:.2f} + {m:.2f}x", transform=axs[1].transAxes, va="top"
    )
    # Add R^2 to plot
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    n = len(x)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - num_predictors - 1)
    axs[1].text(
        0.05, 0.9, f"Adj R\u00b2 = {adj_r2:.2f}", transform=axs[1].transAxes, va="top"
    )
    axs[1].set_title("Convexity Expected vs Actual Change")
    axs[1].set_xlabel("Actual Change")
    axs[1].set_ylabel("Expected Change")

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
        resampled_dates = pd.date_range(
            start=simulation_dates[0], end=simulation_dates[-1], freq=freq
        )
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
        "mean_reversion_only": {
            "enable_spread_cvx": False,
            "enable_rate_vol": False,
            "enable_local_vol": False,
        },
        "include_spread_cvx": {
            "enable_spread_cvx": True,
            "enable_rate_vol": False,
            "enable_local_vol": False,
        },
        "include_rate_vol": {
            "enable_spread_cvx": False,
            "enable_rate_vol": True,
            "enable_local_vol": False,
        },
        "include_local_vol": {
            "enable_spread_cvx": False,
            "enable_rate_vol": False,
            "enable_local_vol": True,
        },
        "full_model": {
            "enable_spread_cvx": True,
            "enable_rate_vol": True,
            "enable_local_vol": True,
        },
    }
    output = {}

    resampled_dates = pd.date_range(
        start=simulation_dates[0], end=simulation_dates[-1], freq=simulation_freq
    )
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

    resampled_dates = pd.date_range(
        start=simulation_dates[0], end=simulation_dates[-1], freq=simulation_freq
    )
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
        estimation_freq="B",
        simulation_freq="B",
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


def evaluation_criteria(
    simulated_path: pd.Series, actual_data: pd.Series, model: JointReversionModel
) -> dict:
    """Evaluate model performance using different criteria.

    Args:
        simulated_path (pd.Series): Simulated path.
        actual_data (pd.Series): Actual data.
        model (JointReversionModel): Model object.

    Returns:
        dict: Dictionary with evaluation criteria.

    R-squared (R2) is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. The R-squared value is a number between 0 and 1, where 0 indicates that the model does not fit the data and 1 indicates that the model fits perfectly.
    Adjusted R-squared (Adj R2) is the coefficient of determination adjusted for the number of predictors in the model. It is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model, adjusted for the number of predictors in the model.
    Mean Squared Error (MSE) is the average of the squared differences between the predicted and actual values in a regression model. It is a measure of the quality of an estimator—it is always non-negative, and values closer to zero are better.
    Root Mean Squared Error (RMSE) is the square root of the average of the squared differences between the predicted and actual values in a regression model. It is a measure of the quality of an estimator—it is always non-negative, and values closer to zero are better.
    Mean Absolute Error (MAE) is the average of the absolute differences between the predicted and actual values in a regression model. It is a measure of the quality of an estimator—it is always non-negative, and values closer to zero are better.

    See: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    """
    actual_path = actual_data.reindex(simulated_path.index)
    try:
        r2 = r2_score(actual_path, simulated_path)
        adj_r2 = 1 - (1 - r2) * (len(actual_path) - 1) / (
            len(actual_path) - model.get_num_predictors() - 1
        )
        mse = mean_squared_error(actual_path, simulated_path)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_path, simulated_path)
    except ValueError:
        mse, rmse, mae, r2, adj_r2 = np.nan, np.nan, np.nan, np.nan, np.nan

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "Adj R2": adj_r2}


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
    simulation_dates = pd.date_range(
        end=oas_data.index[-1], periods=simulation_days, freq="B"
    )

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

    frequency_oas = {
        k: (v["OAS"].loc[:, cols].dropna(), v["Model"])
        for k, v in frequency_results.items()
    }
    complexity_oas = {
        k: (v["OAS"].loc[:, cols].dropna(), v["Model"])
        for k, v in complexity_results.items()
    }
    ols_vs_mle_oas = {
        k: (v["OAS"].loc[:, cols].dropna(), v["Model"])
        for k, v in estimation_results.items()
    }
    oas_evaluation = {
        "frequency": {
            k: evaluation_criteria(v["Expected Change"], v["Actual Change"], m)
            for k, (v, m) in frequency_oas.items()
        },
        "complexity": {
            k: evaluation_criteria(v["Expected Change"], v["Actual Change"], m)
            for k, (v, m) in complexity_oas.items()
        },
        "ols_vs_mle": {
            k: evaluation_criteria(v["Expected Change"], v["Actual Change"], m)
            for k, (v, m) in ols_vs_mle_oas.items()
        },
    }

    frequency_cvx = {
        k: (v["Convexity"].loc[:, cols].dropna(), v["Model"])
        for k, v in frequency_results.items()
    }
    complexity_cvx = {
        k: (v["Convexity"].loc[:, cols].dropna(), v["Model"])
        for k, v in complexity_results.items()
    }
    ols_vs_mle_cvx = {
        k: (v["Convexity"].loc[:, cols].dropna(), v["Model"])
        for k, v in estimation_results.items()
    }
    cvx_evaluation = {
        "frequency": {
            k: evaluation_criteria(v["Expected Change"], v["Actual Change"], m)
            for k, (v, m) in frequency_cvx.items()
        },
        "complexity": {
            k: evaluation_criteria(v["Expected Change"], v["Actual Change"], m)
            for k, (v, m) in complexity_cvx.items()
        },
        "ols_vs_mle": {
            k: evaluation_criteria(v["Expected Change"], v["Actual Change"], m)
            for k, (v, m) in ols_vs_mle_cvx.items()
        },
    }

    def flatten_evaluation(evaluation_dict):
        """Flatten the evaluation dictionary into a DataFrame."""
        flattened_data = []
        for topic, cases in evaluation_dict.items():
            for case, metrics in cases.items():
                flattened_entry = {"topic": topic, "case": case}
                flattened_entry.update(metrics)
                flattened_data.append(flattened_entry)
        return pd.DataFrame(flattened_data)

    oas_evaluation_df = flatten_evaluation(oas_evaluation)
    cvx_evaluation_df = flatten_evaluation(cvx_evaluation)

    return {
        "OAS Eval": oas_evaluation_df,
        "Convexity Eval": cvx_evaluation_df,
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
    historical_dates = pd.date_range(
        start=train_start_date, end=train_end_date, freq=historical_freq
    )

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
        step_interval=3 * 21,
        verbose=True,
    )

    print("\nModel evaluation results:")
    print(f"OAS evaluation:\n{results['oas_evaluation']}")
    print(f"CVX evaluation:\n{results['cvx_evaluation']}")
    print("\nModel evaluation complete.")
