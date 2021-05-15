from datetime import datetime

import pandas as pd
import sklearn.metrics as learn
from matplotlib import pyplot as plt
from pandas import DataFrame
from statsmodels.tsa.seasonal import seasonal_decompose


class Simulator:
    def __init__(self, trading_data: DataFrame = None, currency_pair: str = None,
                 model_name: str = None):
        self.model_name = model_name
        self.trading_data = trading_data
        self.training_data = None
        self.validation_data = None
        self.currency_pair = currency_pair
        self.forecasts = []
        self.forecasts_upper = []
        self.forecasts_lower = []
        self.errors = []
        self.metrics = None
        self.forecasts_raw = []
        self.decomposition = None

    def forecast(self, forecast_horizon: int = 100):
        training_data_size = self.trading_data.size - forecast_horizon
        self.training_data = self.trading_data.head(training_data_size)
        self.validation_data = self.trading_data.tail(forecast_horizon)

        time_series = self.training_data[["close"]]
        raw_decomposition = seasonal_decompose(time_series, model='additive', period=1)
        decomposition = DataFrame.from_records(
            [{"trend": trend, "residual": residual, "seasonality": seasonality} for trend, residual, seasonality in
             zip(raw_decomposition.trend, raw_decomposition.resid, raw_decomposition.seasonal)])
        self.decomposition = decomposition

        print("\n\ndata_frame preview:")
        print(self.training_data.head())
        print(self.training_data.tail())

        print("\n\ndata_frame shape:")
        print(self.training_data.shape)

        print("\n\ndata_frame summary:")
        print(self.training_data.describe())
        print(self.training_data.info())

        print("\n\ndata_frame trend/residual/seasonality:")
        print(self.decomposition)
        decomposition.to_csv(f"output/{self.currency_pair}__{self.model_name.lower()}__trend_seasonality.csv")
        print("\n\n")

    def evaluate_forecast(self):
        n = min(len(self.validation_data), len(self.forecasts))
        y_forecast = self.forecasts[:n]
        y_actual = self.validation_data.tail(n)["close"]

        mean_abs_err = learn.mean_absolute_error(y_actual, y_forecast)
        mean_sq_err = learn.mean_squared_error(y_actual, y_forecast)
        mean_sq_lg_err = learn.mean_squared_log_error(y_actual, y_forecast)
        mean_abs_percent_err = learn.mean_absolute_percentage_error(y_actual, y_forecast)
        median_abs_err = learn.median_absolute_error(y_actual, y_forecast)
        mean_gamma_dev = learn.mean_gamma_deviance(y_actual, y_forecast)
        mean_poisson_dev = learn.mean_poisson_deviance(y_actual, y_forecast)
        mean_tweedie_dev = learn.mean_tweedie_deviance(y_actual, y_forecast)
        explained_variance = learn.explained_variance_score(y_actual, y_forecast)
        max_residual = learn.max_error(y_actual, y_forecast)
        coeff_determination = learn.r2_score(y_actual, y_forecast)

        metrics = {
            "Mean Squared Error (MSE)": mean_sq_err,
            "Mean Absolute Error (MAE)": mean_abs_err,
            "Mean Squared Logarithmic Error (MSLE)": mean_sq_lg_err,
            "Mean Absolute Percentage Error (MAPE)": mean_abs_percent_err,
            "Median Absolute Error (MedAE)": median_abs_err,
            "Mean Gamma Deviance": mean_gamma_dev,
            "Mean Poisson Deviance": mean_poisson_dev,
            "Mean Tweedie Deviance Error": mean_tweedie_dev,
            "Explained Variance Regression Score": explained_variance,
            "Max Residual Error": max_residual,
            "Coefficient of Determination": coeff_determination
        }
        self.metrics = metrics

    def plot_decomposition(self):
        trend = self.decomposition.trend
        seasonal = self.decomposition.seasonal
        residual = self.decomposition.resid

    def plot_trading_data(self):
        x = self.training_data["date"]
        y = self.training_data["close"]

        plt.figure(figsize=(20, 10))
        plt.subplot(221)
        plt.plot(self.training_data)

    def plot_source_dataset(self):
        sub_set = self.validation_data
        forecast_horizon = len(self.forecasts)
        x = pd.date_range(end=datetime.strptime("2021-05-14", "%Y-%m-%d"), periods=forecast_horizon, freq='B').tolist()
        y_real = sub_set["close"]
        y_forecast = self.forecasts[:forecast_horizon]
        y_forecast_upper = self.forecasts_upper[:forecast_horizon]
        y_forecast_lower = self.forecasts_lower[:forecast_horizon]

        fig, ax = plt.subplots()
        ax.plot(x, y_real, c='blue', linewidth=2, label="actual price")
        ax.plot(x, y_forecast, c='green', linewidth=2, label="forecast price")
        ax.fill_between(x, y_forecast_lower, y_forecast_upper, color='k', alpha=0.2,
                        label="95% confidence interval")

        ax.set(xlabel="Date", ylabel="Closing Price",
               title="{}: {} Model - Actual vs Forecast Closing Price".format(self.currency_pair.upper(),
                                                                              self.model_name))
        ax.grid()
        ax.legend(loc='best')
        # fig.savefig("{}_{}_real_forecast.png".format(self.model_name, self.currency_pair))
        plt.show()
