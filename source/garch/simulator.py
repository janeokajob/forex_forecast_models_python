from armagarch import ARMA, empModel, normalDist, garch
from numpy import sqrt
from pandas import DataFrame

from source.common.simulator import Simulator
from source.garch.base import ModelParams


class GarchSimulator(Simulator):

    def __init__(self, trading_data: DataFrame = None, currency_pair: str = None,
                 params: ModelParams = ModelParams()):
        super().__init__(trading_data, currency_pair, "GARCH")
        self.params = params

    def forecast(self, forecast_horizon: int = 96):
        super().forecast(forecast_horizon)
        print("Running GARCH forecast for Currency-pair: {} using forecast horizon: {}", self.currency_pair.upper(), forecast_horizon)
        print("Dataset: ", self.currency_pair.upper())
        print(self.training_data.head(5))
        print(".....\t.........\t...")
        print(self.training_data.tail(5))

        # define mean, vol and distribution
        mean = ARMA(order={'AR': self.params.AR, 'MA': self.params.MA})
        vol = garch(order={'p': self.params.p, 'q': self.params.q})
        distribution = normalDist()

        # create a model
        closing_prices = self.training_data['close'].to_frame()
        model = empModel(closing_prices, mean, vol, distribution)
        # fit model
        model.fit()

        # get the conditional mean
        conditional_mean = model.Ey
        print("conditional mean:", conditional_mean)

        # get conditional variance
        ht = model.ht
        conditional_variance = sqrt(ht)
        print("conditional variance:", conditional_variance)

        # get standardized residuals
        standardized_residuals = model.stres
        print("standardized residuals:", standardized_residuals)

        # make a prediction of mean and variance over next 100 days.
        # results is a list of two-arrays with first array being prediction of mean
        # and second array being prediction of variance
        results = model.predict(nsteps=forecast_horizon)
        predictions = results[0]
        errors = results[1]

        print("GARCH forecast ... complete")
        collated_results = DataFrame.from_records(
            [{"forecast": value, "error": error, "forecast_lower": value - error, "forecast_upper": value + error} for
             value, error in zip(predictions, errors)])

        self.forecasts = collated_results["forecast"]
        self.errors = collated_results["error"]
        self.forecasts_lower = collated_results["forecast_lower"]
        self.forecasts_upper = collated_results["forecast_upper"]
        self.forecasts_raw = collated_results

        collated_results.to_csv(f"output/{self.currency_pair}__{self.model_name.lower()}__{forecast_horizon}__forecasts.csv")
        print(collated_results)
