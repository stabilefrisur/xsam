import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller


class JointReversionModel:
    """Joint Reversion Model for OAS and Convexity.

    The model is defined by the following stochastic differential equations:
    dS_OAS = kappa * (S_OAS_inf - S_OAS) * dt + (gamma_0 * C + gamma_1 * sigma_r + gamma_2 * nu_r) * dt + sigma_O * dW_O
    dC = lambda * (C_inf - C) * dt + (beta_0 * S_OAS + beta_1 * sigma_r + beta_2 * nu_r) * dt + sigma_C * dW_C
    where:
    - S_OAS is the OAS spread.
    - C is the spread Convexity.
    - kappa and lambda are the reversion speeds for OAS and Convexity, respectively.
    - S_OAS_inf and C_inf are the long-term means of OAS and Convexity, respectively.
    - gamma and beta are the interest rate interaction coefficients for OAS and Convexity, respectively.
    - sigma_r is the volatility of interest rates.
    - nu_r is the volatility of volatility of interest rates.
    - sigma_O is the volatility of OAS.
    - sigma_C is the volatility of Convexity.
    - dt is the time step as a fraction of the investment horizon.
    - dW_O and dW_C are Wiener processes for OAS and Convexity, respectively.

    The model is simulated using the Euler-Maruyama method with the following parameters:
    - S_OAS_init: Initial value of OAS spread.
    - C_init: Initial value of Convexity.
    - sigma_r_forward: Volatility of interest rates over the investment horizon.
    - nu_r_forward: Volatility of volatility of interest rates over the investment horizon.
    - steps: Number of time steps of the investment horizon.

    For educational and illustrative purposes only. Not intended for trading or investment purposes.
    Use at your own risk. No warranty or guarantee of accuracy or reliability.
    """
    def __init__(
        self,
        dt: float,
        kappa: float = 0.0,
        gamma: list[float] = [0.0, 0.0, 0.0],
        sigma_O_0: float = 0.0,
        delta: float = 0.0,
        lambda_: float = 0.0,
        beta: list[float] = [0.0, 0.0, 0.0],
        sigma_C: float = 0.0,
        enable_spread_cvx: bool = True,
        enable_rate_vol: bool = True,
        enable_local_vol: bool = True,
    ):
        """Initialize the Joint Reversion Model.

        Args:
            dt (float): Simulation time step size. Usually 1/steps.
            kappa (float): Reversion speed for OAS.
            gamma (list[float]): Interest rate interaction coefficients for OAS.
            sigma_O_0 (float): Initial volatility of OAS.
            delta (float): Convexity volatility coefficient.
            lambda_ (float): Reversion speed for Convexity.
            beta (list[float]): Interest rate interaction coefficients for Convexity.
            sigma_C (float): Volatility of Convexity.
            enable_spread_cvx (bool, optional): Enable convexity interaction. Defaults to True.
            enable_rate_vol (bool, optional): Enable interest rate interaction. Defaults to True.
            enable_local_vol (bool, optional): Enable local volatility. Defaults to True.
            ols_OAS (sm.regression.linear_model.RegressionResults, optional): OLS regression results for OAS. Defaults to None.
            ols_C (sm.regression.linear_model.RegressionResults, optional): OLS regression results for Convexity. Defaults to None.
            ols_sigma_O (sm.regression.linear_model.RegressionResults, optional): OLS regression results for volatility of OAS. Defaults to None.
        """
        self.dt = dt
        self.kappa = kappa
        self.lambda_ = lambda_
        self.gamma = gamma
        self.beta = beta
        self.sigma_O_0 = sigma_O_0
        self.delta = delta
        self.sigma_C = sigma_C
        self.enable_spread_cvx = enable_spread_cvx
        self.enable_rate_vol = enable_rate_vol
        self.enable_local_vol = enable_local_vol

    def get_parameters(self) -> dict[str, float | list[float]]:
        """Get the parameters of the Joint Reversion Model.

        Returns:
            dict[str, float | list[float]]: Model parameters.
        """
        return {
            "dt": self.dt,
            "kappa": self.kappa,
            "gamma": self.gamma,
            "sigma_O_0": self.sigma_O_0,
            "delta": self.delta,
            "lambda": self.lambda_,
            "beta": self.beta,
            "sigma_C": self.sigma_C,
            "enable_spread_cvx": self.enable_spread_cvx,
            "enable_rate_vol": self.enable_rate_vol,
            "enable_local_vol": self.enable_local_vol,
        }
    
    def update_parameters(
        self,
        dt: float = None,
        kappa: float = None,
        gamma: list[float] = None,
        sigma_O_0: float = None,
        delta: float = None,
        lambda_: float = None,
        beta: list[float] = None,
        sigma_C: float = None,
        enable_spread_cvx: bool = None,
        enable_rate_vol: bool = None,
        enable_local_vol: bool = None,
    ) -> None:
        """Update the parameters of the Joint Reversion Model.

        Args:
            dt (float): Simulation time step size. Usually 1/steps.
            kappa (float): Reversion speed for OAS.
            gamma (list[float]): Interest rate interaction coefficients for OAS.
            sigma_O_0 (float): Initial volatility of OAS.
            delta (float): Convexity volatility coefficient.
            lambda_ (float): Reversion speed for Convexity.
            beta (list[float]): Interest rate interaction coefficients for Convexity.
            sigma_C (float): Volatility of Convexity.
            enable_spread_cvx (bool): Enable convexity interaction.
            enable_rate_vol (bool): Enable interest rate interaction. 
        """
        if dt is not None:
            self.dt = dt
        if kappa is not None:
            self.kappa = kappa
        if gamma is not None:
            self.gamma = gamma
        if sigma_O_0 is not None:
            self.sigma_O_0 = sigma_O_0
        if delta is not None:
            self.delta = delta
        if lambda_ is not None:
            self.lambda_ = lambda_
        if beta is not None:
            self.beta = beta
        if sigma_C is not None:
            self.sigma_C = sigma_C
        if enable_spread_cvx is not None:
            self.enable_spread_cvx = enable_spread_cvx
        if enable_rate_vol is not None:
            self.enable_rate_vol = enable_rate_vol
        if enable_local_vol is not None:
            self.enable_local_vol = enable_local_vol

    def estimate_parameters_ols(
        self,
        S_OAS: pd.Series,
        C: pd.Series,
        sigma_r: pd.Series,
        nu_r: pd.Series,
        S_OAS_inf: float,
        C_inf: float,
        steps: int,
        verbose: bool = True,
    ) -> None:
        """Estimate the parameters of the joint reversion model using OLS.

        Args:
            S_OAS (pd.Series): OAS spread time series.
            C (pd.Series): Convexity time series.
            sigma_r (pd.Series): Volatility of interest rates time series.
            nu_r (pd.Series): Volatility of volatility of interest rates time series.
            S_OAS_inf (float): Long-term mean of OAS spread.
            C_inf (float): Long-term mean of Convexity.
            steps (int): Number of time steps of the investment horizon.
            verbose (bool, optional): Print estimated parameters. Defaults to True.
        """
        # Convert pd.Series to np.ndarray for better performance
        S_OAS = S_OAS.values
        C = C.values
        sigma_r = sigma_r.values
        nu_r = nu_r.values

        # Mean Reversion Parameter Estimation using OLS
        X_OAS = [S_OAS_inf - S_OAS[:-steps]]
        X_C = [C_inf - C[:-steps]]

        if self.enable_spread_cvx:
            X_OAS.append(C[:-steps])
            X_C.append(S_OAS[:-steps])
        if self.enable_rate_vol:
            X_OAS.extend([sigma_r[:-steps], nu_r[:-steps]])
            X_C.extend([sigma_r[:-steps], nu_r[:-steps]])

        X_OAS = np.vstack(X_OAS).T
        y_OAS = S_OAS[steps:] - S_OAS[:-steps]
        X_C = np.vstack(X_C).T
        y_C = C[steps:] - C[:-steps]

        # OLS regression for OAS
        X_OAS = sm.add_constant(X_OAS)
        ols_OAS = sm.OLS(y_OAS, X_OAS).fit()
        kappa = ols_OAS.params[0]
        kappa_p = ols_OAS.pvalues[0]
        if not self.enable_spread_cvx and not self.enable_rate_vol:
            gamma = [0.0, 0.0, 0.0]
            gamma_p = [None, None, None]
        elif self.enable_spread_cvx and not self.enable_rate_vol:
            gamma = [ols_OAS.params[1], 0.0, 0.0]
            gamma_p = [ols_OAS.pvalues[1], None, None]
        elif not self.enable_spread_cvx and self.enable_rate_vol:
            gamma = [0.0] + list(ols_OAS.params[1:])
            gamma_p = [None] + list(ols_OAS.pvalues[1:])
        else:
            gamma = list(ols_OAS.params[1:])
            gamma_p = list(ols_OAS.pvalues[1:])

        # OLS regression for Convexity
        X_C = sm.add_constant(X_C)
        ols_C = sm.OLS(y_C, X_C).fit()
        lambda_ = ols_C.params[0]
        lambda_p = ols_C.pvalues[0]
        if not self.enable_spread_cvx and not self.enable_rate_vol:
            beta = [0.0, 0.0, 0.0]
            beta_p = [None, None, None]
        elif self.enable_spread_cvx and not self.enable_rate_vol:
            beta = [ols_C.params[1], 0.0, 0.0]
            beta_p = [ols_C.pvalues[1], None, None]
        elif not self.enable_spread_cvx and self.enable_rate_vol:
            beta = [0.0] + list(ols_C.params[1:])
            beta_p = [None] + list(ols_C.pvalues[1:])
        else:
            beta = list(ols_C.params[1:])
            beta_p = list(ols_C.pvalues[1:])

        # Residuals for variance calibration
        residuals_OAS = y_OAS - X_OAS @ ols_OAS.params
        residuals_C = y_C - X_C @ ols_C.params

        # Residual Variance Calibration
        ols_sigma_O = None
        if self.enable_local_vol:
            ols_sigma_O = sm.OLS(residuals_OAS**2, sm.add_constant(C[:-steps]**2)).fit()
            sigma_O_0 = np.sqrt(ols_sigma_O.params[0])
            sigma_O_0_p = ols_sigma_O.pvalues[0]
            delta = ols_sigma_O.params[1]
            delta_p = ols_sigma_O.pvalues[1]
        else:
            sigma_O_0 = np.std(residuals_OAS)
            sigma_O_0_p = None
            delta = 0.0
            delta_p = None

        sigma_C = np.std(residuals_C)

        ols = {
            "kappa": kappa,
            "gamma_0": gamma[0],
            "gamma_1": gamma[1],
            "gamma_2": gamma[2],
            "kappa_p": kappa_p,
            "gamma_0_p": gamma_p[0],
            "gamma_1_p": gamma_p[1],
            "gamma_2_p": gamma_p[2],
            "r2_OAS": ols_OAS.rsquared,
            "r2_adj_OAS": ols_OAS.rsquared_adj,
            "model_OAS": ols_OAS,
            "lambda": lambda_,
            "beta_0": beta[0],
            "beta_1": beta[1],
            "beta_2": beta[2],
            "lambda_p": lambda_p,
            "beta_0_p": beta_p[0],
            "beta_1_p": beta_p[1],
            "beta_2_p": beta_p[2],
            "r2_C": ols_C.rsquared,
            "r2_adj_C": ols_C.rsquared_adj,
            "model_C": ols_C,
            "sigma_O_0": sigma_O_0,
            "delta": delta,
            "sigma_O_0_p": sigma_O_0_p,
            "delta_p": delta_p,
            "r2_sigma_O": ols_sigma_O.rsquared if ols_sigma_O is not None else None,
            "r2_adj_sigma_O": ols_sigma_O.rsquared_adj if ols_sigma_O is not None else None,
            "model_sigma_O": ols_sigma_O,
            "sigma_C": sigma_C,
        }

        if verbose:
            print('\nEstimated Parameters (OLS):')
            print(f'kappa: {kappa:.4f}')
            print(f'gamma: {", ".join(f"{g:.4f}" for g in gamma)}')
            print(f'sigma_O_0: {sigma_O_0:.4f}')
            print(f'delta: {delta:.4f}')
            print(f'lambda: {lambda_:.4f}')
            print(f'beta: {", ".join(f"{b:.4f}" for b in beta)}')
            print(f'sigma_C: {sigma_C:.4f}')

        self.kappa = kappa
        self.lambda_ = lambda_
        self.gamma = gamma
        self.beta = beta
        self.sigma_O_0 = sigma_O_0
        self.delta = delta
        self.sigma_C = sigma_C
        self.ols = ols

    def estimate_parameters_mle(
        self,
        S_OAS: pd.Series,
        C: pd.Series,
        sigma_r: pd.Series,
        nu_r: pd.Series,
        S_OAS_inf: float,
        C_inf: float,
        dt: float,
        verbose: bool = True,
    ) -> None:
        """Estimate the parameters of the joint reversion model using MLE.

        Args:
            S_OAS (pd.Series): OAS spread time series.
            C (pd.Series): Convexity time series.
            sigma_r (pd.Series): Volatility of interest rates time series.
            nu_r (pd.Series): Volatility of volatility of interest rates time series.
            S_OAS_inf (float): Long-term mean of OAS spread.
            C_inf (float): Long-term mean of Convexity.
            dt (float): Time fraction of one step of the investment horizon.
            verbose (bool, optional): Print estimated parameters. Defaults to True.
        """
        S_OAS = S_OAS.values
        C = C.values
        sigma_r = sigma_r.values
        nu_r = nu_r.values

        def log_likelihood(params):
            kappa = params[0]
            gamma = params[1:4]
            sigma_O_0 = params[4]
            delta = params[5]
            lambda_ = params[6]
            beta = params[7:10]
            sigma_C = params[10]

            n = len(S_OAS)
            log_likelihood = 0

            for t in range(1, n):
                S_OAS_prev = S_OAS[t - 1]
                C_prev = C[t - 1]
                sigma_r_prev = sigma_r[t - 1]
                nu_r_prev = nu_r[t - 1]

                S_OAS_mean = S_OAS_prev + kappa * (S_OAS_inf - S_OAS_prev) * dt
                if self.enable_spread_cvx:
                    S_OAS_mean += gamma[0] * C_prev * dt
                if self.enable_rate_vol:
                    S_OAS_mean += (gamma[1] * sigma_r_prev + gamma[2] * nu_r_prev) * dt

                sigma_O = (
                    np.sqrt(sigma_O_0**2 + delta * C_prev**2)
                    if self.enable_local_vol
                    else sigma_O_0
                )

                C_mean = C_prev + lambda_ * (C_inf - C_prev) * dt
                if self.enable_spread_cvx:
                    C_mean += beta[0] * S_OAS_prev * dt
                if self.enable_rate_vol:
                    C_mean += (beta[1] * sigma_r_prev + beta[2] * nu_r_prev) * dt

                log_likelihood += -0.5 * np.log(2 * np.pi * sigma_O**2 * dt) - 0.5 * (
                    (S_OAS[t] - S_OAS_mean) ** 2 / (sigma_O**2 * dt)
                )
                log_likelihood += -0.5 * np.log(2 * np.pi * sigma_C**2 * dt) - 0.5 * (
                    (C[t] - C_mean) ** 2 / (sigma_C**2 * dt)
                )

            return -log_likelihood

        # Optimize the log-likelihood function
        initial_guess = (
            self.kappa,
            *self.gamma,
            self.sigma_O_0,
            self.delta,
            self.lambda_,
            *self.beta,
            self.sigma_C,
        )
        result = minimize(log_likelihood, initial_guess, method='L-BFGS-B')

        # Extract the estimated parameters
        kappa = result.x[0]
        gamma = result.x[1:4]
        sigma_O_0 = result.x[4]
        delta = result.x[5]
        lambda_ = result.x[6]
        beta = result.x[7:10]
        sigma_C = result.x[10]

        if verbose:
            print('\nEstimated Parameters (MLE):')
            print(f'kappa: {kappa:.4f}')
            print(f'gamma: {", ".join(f"{g:.4f}" for g in gamma)}')
            print(f'sigma_O_0: {sigma_O_0:.4f}')
            print(f'delta: {delta:.4f}')
            print(f'lambda: {lambda_:.4f}')
            print(f'beta: {", ".join(f"{b:.4f}" for b in beta)}')
            print(f'sigma_C: {sigma_C:.4f}')

        self.kappa = kappa
        self.gamma = gamma
        self.sigma_O_0 = sigma_O_0
        self.delta = delta
        self.lambda_ = lambda_
        self.beta = beta
        self.sigma_C = sigma_C

    # function to get the number of predictors for the model based on the enabled features
    def get_num_predictors(self) -> int:
        """Get the number of predictors for the model.

        Returns:
            int: Number of predictors.
        """
        # Check how many paramters are estimated to be non-zero
        num_predictors = 0
        num_predictors += 1 if self.kappa != 0 else 0
        num_predictors += sum(1 for g in self.gamma if g != 0)
        num_predictors += 1 if self.sigma_O_0 != 0 else 0
        num_predictors += 1 if self.delta != 0 else 0
        num_predictors += 1 if self.lambda_ != 0 else 0
        num_predictors += sum(1 for b in self.beta if b != 0)
        num_predictors += 1 if self.sigma_C != 0 else 0
        return num_predictors

    @staticmethod
    def stationarity_tests(
        S_OAS: pd.Series,
        C: pd.Series,
        verbose: bool = True,
    ) -> tuple[
        tuple[float, float, int, int, dict, float],
        tuple[float, float, int, int, dict, float],
    ]:
        """Test the stationarity of the time series using the Augmented Dickey-Fuller test.

        Args:
            S_OAS (pd.Series): OAS spread time series.
            C (pd.Series): Convexity time series.

        Returns:
            tuple[ tuple[float, float, int, int, dict, float], tuple[float, float, int, int, dict, float], ]: ADF test results for OAS and Convexity.

        ADF test results:
        - Test statistic.
        - P-value.
        - Number of lags used.
        - Number of observations used.
        - Critical values.
        - Maximum information criterion.

        ADF test null hypothesis:
        - H0: The time series is non-stationary.

        ADF test alternative hypothesis:
        - H1: The time series is stationary.

        If the p-value is less than the significance level (e.g., 0.05), we reject the null hypothesis and conclude that the time series is stationary.

        Reference:
        - https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
        """
        adf_OAS = adfuller(S_OAS)
        adf_C = adfuller(C)

        if verbose:
            # Print ADF test results
            print('\nAugmented Dickey-Fuller Test for OAS:')
            print(f'Test Statistic: {adf_OAS[0]:.4f}')
            print(f'P-Value: {adf_OAS[1]:.4f}')
            print(f'Lags Used: {adf_OAS[2]:.0f}')
            print(f'Observations Used: {adf_OAS[3]:.0f}')
            print('Critical Values:')
            for key, value in adf_OAS[4].items():
                print(f'  {key}: {value:.4f}')
            print(f'Max Information Criteria: {adf_OAS[5]:.4f}')

            print('\nAugmented Dickey-Fuller Test for Convexity:')
            print(f'Test Statistic: {adf_C[0]:.4f}')
            print(f'P-Value: {adf_C[1]:.4f}')
            print(f'Lags Used: {adf_C[2]:.0f}')
            print(f'Observations Used: {adf_C[3]:.0f}')
            print('Critical Values:')
            for key, value in adf_C[4].items():
                print(f'  {key}: {value:.4f}')
            print(f'Max Information Criteria: {adf_C[5]:.4f}')

        return adf_OAS, adf_C

    def simulate(
        self,
        S_OAS_init: float,
        C_init: float,
        S_OAS_inf: float,
        C_inf: float,
        sigma_r: pd.Series,
        nu_r: pd.Series,
        simulation_dates: pd.DatetimeIndex,
        rng: np.random.Generator = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Simulate the joint reversion model for OAS and Convexity (and Volatility of OAS).

        Args:
            S_OAS_init (float): Initial value of OAS spread.
            C_init (float): Initial value of Convexity.
            S_OAS_inf (float): Long-term mean of OAS spread.
            C_inf (float): Long-term mean of Convexity.
            sigma_r (pd.Series): Volatility of interest rates.
            nu_r (pd.Series): Volatility of volatility of interest rates.
            simulation_dates (pd.DatetimeIndex): Dates for the simulation.
            rng (np.random.Generator, optional): Random number generator. Defaults to None.

        Returns:
            tuple[pd.Series, pd.Series, pd.Series]: Simulated paths of OAS, Convexity, and Volatility of OAS.
        """
        if rng is None:
            rng = np.random.default_rng()

        steps = len(simulation_dates)
        S_OAS = pd.Series(index=simulation_dates)
        C = pd.Series(index=simulation_dates)
        sigma_O = pd.Series(index=simulation_dates)
        S_OAS.iloc[0] = S_OAS_init
        C.iloc[0] = C_init
        sigma_O.iloc[0] = np.sqrt(self.sigma_O_0**2 + self.delta * C_init**2)

        for t in range(1, steps):
            Z_O = rng.normal()
            Z_C = rng.normal()

            if self.enable_spread_cvx:
                gamma_term = self.gamma[0] * float(C.iloc[t - 1])
                beta_term = self.beta[0] * float(S_OAS.iloc[t - 1])
            else:
                gamma_term = 0
                beta_term = 0

            if self.enable_rate_vol:
                gamma_term += (
                    self.gamma[1] * float(sigma_r.iloc[t - 1]) 
                    + self.gamma[2] * float(nu_r.iloc[t - 1])
                )
                beta_term += (
                    self.beta[1] * float(sigma_r.iloc[t - 1]) 
                    + self.beta[2] * float(nu_r.iloc[t - 1])
                )

            S_OAS.iloc[t] = (
                float(S_OAS.iloc[t - 1])
                + self.kappa * (S_OAS_inf - float(S_OAS.iloc[t - 1])) * self.dt
                + gamma_term * self.dt
                + sigma_O.iloc[t - 1] * np.sqrt(self.dt) * Z_O
            )

            C.iloc[t] = (
                float(C.iloc[t - 1])
                + self.lambda_ * (C_inf - float(C.iloc[t - 1])) * self.dt
                + beta_term * self.dt
                + self.sigma_C * np.sqrt(self.dt) * Z_C
            )

            variance_O = self.sigma_O_0**2 + self.delta * C.iloc[t]**2
            if variance_O >= 0:
                sigma_O.iloc[t] = np.sqrt(variance_O)
            else:
                sigma_O.iloc[t] = np.nan

        return S_OAS, C, sigma_O


    def monte_carlo_simulation(
        self,
        S_OAS_init: float,
        C_init: float,
        S_OAS_inf: float,
        C_inf: float,
        sigma_r_forward: float | pd.Series,
        nu_r_forward: float | pd.Series,
        simulation_dates: pd.DatetimeIndex,
        num_paths: int = 1000,
        seed: int = None,
        verbose: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Perform Monte Carlo simulation of the Joint Reversion Model.

        Args:
            S_OAS_init (float): Initial value of OAS spread.
            C_init (float): Initial value of Convexity.
            S_OAS_inf (float): Long-term mean of OAS spread.
            C_inf (float): Long-term mean of Convexity.
            sigma_r_forward (float | pd.Series): Volatility of interest rates.
            nu_r_forward (float | pd.Series): Volatility of volatility of interest rates.
            simulation_dates (pd.DatetimeIndex): Dates for the simulation.
            num_paths (int): Number of Monte Carlo paths.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            verbose (bool, optional): Print simulation results. Defaults to True.

        Returns:
            dict[str, pd.DataFrame]: Simulated paths of OAS, Convexity, and Volatility of OAS.

        The Monte Carlo simulation generates multiple paths of the OAS spread, Convexity, and Volatility of OAS using the Joint Reversion Model.
        The expected value of OAS, Convexity, and Volatility of OAS can be calculated as the average of the simulated paths.

        Reference:
        - https://en.wikipedia.org/wiki/Monte_Carlo_method

        Note:
        - The Monte Carlo simulation is a stochastic process that generates random paths.
        - The results may vary each time the simulation is run.
        - The results are based on the model assumptions and parameters.
        - The results are for illustrative and educational purposes only.
        - Use at your own risk. No warranty or guarantee of accuracy or reliability.
        """
        paths_OAS = []
        paths_C = []
        paths_sigma_O = []

        steps = len(simulation_dates)

        if isinstance(sigma_r_forward, float):
            sigma_r_series = pd.Series(np.full(steps, sigma_r_forward), index=simulation_dates)
        else:
            sigma_r_series = sigma_r_forward

        if isinstance(nu_r_forward, float):
            nu_r_series = pd.Series(np.full(steps, nu_r_forward), index=simulation_dates)
        else:
            nu_r_series = nu_r_forward

        rng = np.random.default_rng(seed)

        for i in range(num_paths):
            S_OAS, C, sigma_O = self.simulate(
                S_OAS_init,
                C_init,
                S_OAS_inf,
                C_inf,
                sigma_r_series,
                nu_r_series,
                simulation_dates,
                rng,
            )
            paths_OAS.append(S_OAS)
            paths_C.append(C)
            paths_sigma_O.append(sigma_O)

        if verbose:
            final_oas = np.mean([path.iloc[-1] for path in paths_OAS])
            final_cvx = np.mean([path.iloc[-1] for path in paths_C])
            final_sigma_o = np.mean([path.iloc[-1] for path in paths_sigma_O])
            print(f"\nMonte Carlo Simulation ({num_paths} paths):")
            print(f"Final expected value of OAS: {final_oas:.0f} bps")
            print(f"Final expected value of Convexity: {final_cvx:.0f} bps")
            print(f"Final expected value of Sigma_O: {final_sigma_o:.0f} bps")

        return {
            "oas": pd.DataFrame(paths_OAS).T,
            "cvx": pd.DataFrame(paths_C).T,
            "sigma_o": pd.DataFrame(paths_sigma_O).T,
        }
