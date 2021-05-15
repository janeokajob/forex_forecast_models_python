import pickle

import pmdarima as pm
from pandas import DataFrame
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer

from source.common.simulator import Simulator


class ArimaSimulator(Simulator):
    def __init__(self, trading_data: DataFrame = None, currency_pair: str = None):
        super().__init__(trading_data, currency_pair, "ARIMA")

    def forecast(self, forecast_horizon: int = 96):
        super().forecast(forecast_horizon)

        print("Running ARIMA forecast for Currency-pair: {} using forecast horizon: {}", self.currency_pair.upper(),
              forecast_horizon)
        print("Dataset: ", self.currency_pair.upper())
        print(self.training_data.head(5))
        print(".....\t.........\t...")
        print(self.training_data.tail(5))

        # define and fit the pipeline/model
        pipeline = Pipeline([
            ('boxcox', BoxCoxEndogTransformer(lmbda2=1e-6)),
            ('arima', pm.AutoARIMA(start_p=1, start_q=1, max_p=3, max_q=3, d=1, D=1, start_P=0, error_action='ignore',
                                   suppress_warnings=True, stepwise=True, seasonal=True, m=12, trace=True))
        ])
        pipeline.fit(self.training_data['close'])
        # model = pm.auto_arima(self.training_data["close"], seasonal=True, m=12)

        # serialize model
        model_file = f"intermediates/arima_{self.currency_pair}.pkl"
        with open(model_file, "wb") as file:
            pickle.dump(pipeline, file)

        # load model and make predictions seamlessly
        with open(model_file, "rb") as file:
            model = pickle.load(file)

        # make the forecasts
        predictions = model.predict(n_periods=forecast_horizon, return_conf_int=True)
        print("ARIMA forecast ... complete")
        collated_results = DataFrame.from_records(
            [{"forecast": value, "error": abs(bounds[0] - bounds[1]) / 2, "forecast_lower": bounds[0],
              "forecast_upper": bounds[1]} for value, bounds in zip(predictions[0], predictions[1])])

        self.forecasts = collated_results["forecast"]
        self.errors = collated_results["error"]
        self.forecasts_lower = collated_results["forecast_lower"]
        self.forecasts_upper = collated_results["forecast_upper"]
        self.forecasts_raw = collated_results

        collated_results.to_csv(f"output/{self.currency_pair}__{self.model_name.lower()}__forecasts.csv")
        print(collated_results)
