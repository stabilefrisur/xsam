import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xsam.mbs.mock_data import generate_historical_data
from xsam.mbs.value_model_main import run_value_model


def evaluate_frequency(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    sigma_r_data: pd.Series,
    nu_r_data: pd.Series,
    complexity_params: dict,
    simulation_dates: pd.DatetimeIndex,
    num_paths: int,
    seed: int,
    verbose: bool = False,
) -> dict:
    """Evaluate model at different frequencies."""
    if verbose:
        print("\nEvaluating model at different frequencies...")

    frequencies = {"daily": "B", "weekly": "W-FRI", "monthly": "BME"}
    results = {}

    for freq_name, freq in frequencies.items():
        resampled_dates = pd.date_range(
            start=simulation_dates[0], end=simulation_dates[-1], freq=freq
        )
        resampled_steps = len(resampled_dates)
        paths, _ = run_value_model(
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
        )
        results[freq_name] = paths

    return results


def evaluate_complexity(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    sigma_r_data: pd.Series,
    nu_r_data: pd.Series,
    simulation_dates: pd.DatetimeIndex,
    num_paths: int,
    seed: int,
    verbose: bool = False,
) -> dict:
    """Evaluate model with different complexities."""
    if verbose:
        print("\nEvaluating model with different complexities...")

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
    results = {}

    resampled_dates = pd.date_range(
        start=simulation_dates[0], end=simulation_dates[-1], freq="B"
    )
    resampled_steps = len(resampled_dates)

    for complexity_name, complexity_params in complexities.items():
        paths, _ = run_value_model(
            oas_data,
            cvx_data,
            sigma_r_data,
            nu_r_data,
            enable_spread_cvx=complexity_params["enable_spread_cvx"],
            enable_rate_vol=complexity_params["enable_rate_vol"],
            enable_local_vol=complexity_params["enable_local_vol"],
            enable_mle=False,
            estimation_freq="B",
            simulation_freq="B",
            simulation_steps=resampled_steps,
            num_paths=num_paths,
            seed=seed,
            verbose=verbose,
        )
        results[complexity_name] = paths

    return results


def evaluate_ols_vs_mle(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    sigma_r_data: pd.Series,
    nu_r_data: pd.Series,
    complexity_params: dict,
    simulation_dates: pd.DatetimeIndex,
    num_paths: int,
    seed: int,
    verbose: bool = False,
) -> dict:
    """Evaluate model using OLS vs OLS + MLE."""
    if verbose:
        print("\nEvaluating model using OLS vs OLS + MLE...")

    results = {}

    resampled_dates = pd.date_range(
        start=simulation_dates[0], end=simulation_dates[-1], freq="B"
    )
    resampled_steps = len(resampled_dates)

    paths_ols, _ = run_value_model(
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        enable_spread_cvx=complexity_params["enable_spread_cvx"],
        enable_rate_vol=complexity_params["enable_rate_vol"],
        enable_local_vol=complexity_params["enable_local_vol"],
        enable_mle=False,
        estimation_freq="B",
        simulation_freq="B",
        simulation_steps=resampled_steps,
        num_paths=num_paths,
        seed=seed,
        verbose=verbose,
    )
    results["ols"] = paths_ols

    paths_mle, _ = run_value_model(
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
    )
    results["mle"] = paths_mle

    return results


def evaluate_criteria(simulated_path, actual_data):
    """Evaluate the model using various criteria."""
    actual_path = actual_data.reindex(simulated_path.index)
    try:
        mse = mean_squared_error(actual_path, simulated_path)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_path, simulated_path)
        r2 = r2_score(actual_path, simulated_path)
    except ValueError:
        mse, rmse, mae, r2 = np.nan, np.nan, np.nan, np.nan

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def evaluate_model(
    oas_data: pd.Series,
    cvx_data: pd.Series,
    sigma_r_data: pd.Series,
    nu_r_data: pd.Series,
    num_paths: int,
    seed: int,
    verbose: bool = False,
):
    """Evaluate the model with different frequencies, complexities, and estimation methods."""
    # Make sure all data series are in B frequency
    oas_data = oas_data.asfreq("D").ffill().asfreq("B")
    cvx_data = cvx_data.asfreq("D").ffill().asfreq("B")
    sigma_r_data = sigma_r_data.asfreq("D").ffill().asfreq("B")
    nu_r_data = nu_r_data.asfreq("D").ffill().asfreq("B")

    # Reserve forward steps for out-of-sample evaluation
    simulation_steps = 252
    in_oas_data = oas_data[:-simulation_steps]
    in_cvx_data = cvx_data[:-simulation_steps]
    in_sigma_r_data = sigma_r_data[:-simulation_steps]
    in_nu_r_data = nu_r_data[:-simulation_steps]
    simulation_dates = pd.date_range(
        end=oas_data.index[-1], periods=simulation_steps, freq="B"
    )

    complexity_params = {
        "enable_spread_cvx": True,
        "enable_rate_vol": True,
        "enable_local_vol": True,
    }

    frequency_results = evaluate_frequency(
        in_oas_data,
        in_cvx_data,
        in_sigma_r_data,
        in_nu_r_data,
        complexity_params,
        simulation_dates,
        num_paths,
        seed,
        verbose,
    )

    complexity_results = evaluate_complexity(
        in_oas_data,
        in_cvx_data,
        in_sigma_r_data,
        in_nu_r_data,
        simulation_dates,
        num_paths,
        seed,
        verbose,
    )

    ols_vs_mle_results = evaluate_ols_vs_mle(
        in_oas_data,
        in_cvx_data,
        in_sigma_r_data,
        in_nu_r_data,
        complexity_params,
        simulation_dates,
        num_paths,
        seed,
        verbose,
    )

    # Evaluate criteria for each result set
    frequency_oas = {k: v["oas"].mean(axis=1) for k, v in frequency_results.items()}
    complexity_oas = {k: v["oas"].mean(axis=1) for k, v in complexity_results.items()}
    ols_vs_mle_oas = {k: v["oas"].mean(axis=1) for k, v in ols_vs_mle_results.items()}
    oas_evaluation = {
        "frequency": {k: evaluate_criteria(v, oas_data) for k, v in frequency_oas.items()},
        "complexity": {k: evaluate_criteria(v, oas_data) for k, v in complexity_oas.items()},
        "ols_vs_mle": {k: evaluate_criteria(v, oas_data) for k, v in ols_vs_mle_oas.items()},
    }

    frequency_cvx = {k: v["cvx"].mean(axis=1) for k, v in frequency_results.items()}
    complexity_cvx = {k: v["cvx"].mean(axis=1) for k, v in complexity_results.items()}
    ols_vs_mle_cvx = {k: v["cvx"].mean(axis=1) for k, v in ols_vs_mle_results.items()}
    cvx_evaluation = {
        "frequency": {k: evaluate_criteria(v, cvx_data) for k, v in frequency_cvx.items()},
        "complexity": {k: evaluate_criteria(v, cvx_data) for k, v in complexity_cvx.items()},
        "ols_vs_mle": {k: evaluate_criteria(v, cvx_data) for k, v in ols_vs_mle_cvx.items()},
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
        "frequency_results": frequency_results,
        "complexity_results": complexity_results,
        "ols_vs_mle_results": ols_vs_mle_results,
        "oas_evaluation": oas_evaluation_df,
        "cvx_evaluation": cvx_evaluation_df,
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
    num_paths = 100  # Number of Monte Carlo paths

    # Perform model evaluation
    results = evaluate_model(
        oas_data,
        cvx_data,
        sigma_r_data,
        nu_r_data,
        num_paths,
        seed,
        verbose=True,
    )

    print("\nModel evaluation results:")
    print(f"OAS evaluation:\n{results['oas_evaluation']}")
    print(f"CVX evaluation:\n{results['cvx_evaluation']}")
    print("\nModel evaluation complete.")
