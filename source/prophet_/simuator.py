from pandas import DataFrame
from prophet import Prophet

from source.common.simulator import Simulator


class ProphetSimulator(Simulator):

    def __init__(self, trading_data: DataFrame = None, currency_pair: str = None):
        super().__init__(trading_data, currency_pair, "Prophet")

    def forecast(self, forecast_horizon: int = 96):
        super().forecast(forecast_horizon)

        print("Running Prophet forecast for Currency-pair: {} using forecast horizon: {}", self.currency_pair.upper(),
              forecast_horizon)
        print("Dataset: ", self.currency_pair.upper())
        print(self.training_data.head(5))
        print(".....\t.........\t...")
        print(self.training_data.tail(5))

        # model = Prophet(interval_width=0.99, mcmc_samples=60)
        model = Prophet(interval_width=0.99)
        model.fit(self.training_data)

        future = model.make_future_dataframe(periods=forecast_horizon)
        future.tail()

        _forecast = model.predict(future)
        last_n = _forecast.tail(forecast_horizon)
        last_n.to_csv(f"output/{self.currency_pair}__{self.model_name.lower()}__forecasts.csv")

        # last_n = _forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(n)
        print(last_n)
        self.forecasts = last_n["yhat"]
        self.forecasts_upper = last_n["yhat_upper"]
        self.forecasts_lower = last_n["yhat_lower"]
        self.errors = [(abs(upper - lower) / 2) for upper, lower in zip(self.forecasts_upper, self.forecasts_lower)]
        self.forecasts_raw = last_n
