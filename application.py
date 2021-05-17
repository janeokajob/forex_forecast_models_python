import multiprocessing
from enum import unique, Enum

from pandas import DataFrame
from tabulate import tabulate

from source.arima.simulator import ArimaSimulator
from source.common.io import DatasetLoader, DataWriter
from source.common.simulator import Simulator
from source.garch.base import ModelParams
from source.garch.simulator import GarchSimulator
from source.prophet_.simuator import ProphetSimulator

multiprocessing.set_start_method("fork")

garch_arma_parameters = { 
    "eur_cad": ModelParams(ar=2, ma=1),  
    "eur_gbp": ModelParams(ar=3, ma=2), 
    "eur_jpy": ModelParams(ar=1, ma=2), 
    "eur_usd": ModelParams(ar=1, ma=2)  
}


@unique
class ForecastModel(Enum):
    ARIMA = 1
    GARCH = 2
    PROPHET = 3


def forecast_trading_data(file_name: str = None, currencies: str = None, model: ForecastModel = ForecastModel.PROPHET,
                          forecast_horizon: int = 96):
    try:
        raw_data = DatasetLoader.load(file_name)
        trading_data = DataFrame.from_records([item.to_dict() for item in raw_data])

        # run forecast model
        simulator: Simulator
        if ForecastModel.PROPHET == model:
            simulator = ProphetSimulator(trading_data=trading_data, currency_pair=currencies)
        elif ForecastModel.GARCH == model:
            simulator = GarchSimulator(trading_data=trading_data,
                                       currency_pair=currencies,
                                       params=garch_arma_parameters[
                                           currencies])

        else:
            simulator = ArimaSimulator(trading_data=trading_data, currency_pair=currencies)

        simulator.forecast(forecast_horizon)

        print("forecasts:")
        print(simulator.forecasts)

        simulator.evaluate_forecast()
        # simulator.plot_source_dataset()

        print("\n\nForecast Evaluation:")
        print("Regression Metrics: {} Forecasts for {}".format(simulator.model_name, simulator.currency_pair.upper()))
        DataWriter.write(
            f"output/{simulator.currency_pair}__{simulator.model_name.lower()}__{forecast_horizon}__evaluation.txt",
            "\n".join([f"{key}\t{value:0.6f}" for key, value in simulator.metrics.items()]))
        print(tabulate([[_key, f"{value:0.6f}"] for _key, value in simulator.metrics.items()]))
    except Exception as e:
        print(
            f"Error running forecast for model {model} and currency-pair: {currencies} with forecast-horizon: {forecast_horizon}")


if __name__ == "__main__":
    # load raw input
    # https://uk.investing.com/currencies/eur-usd-historical-data
    trading_data_files = {
        "eur_cad": "input/eur_cad_trading_data.csv",
        "eur_gbp": "input/eur_gbp_trading_data.csv",
        "eur_jpy": "input/eur_jpy_trading_data.csv"
        "eur_usd": "input/eur_usd_trading_data.csv"
    }

    models = [ForecastModel.PROPHET, ForecastModel.GARCH, ForecastModel.ARIMA]

    for model in models:
        for key in trading_data_files.keys():
            for time_horizon in [100, 200, 500]:
                # load input frames
                data_file = trading_data_files[key]
                currency_pair = key
                forecast_trading_data(file_name=data_file, currencies=currency_pair, model=model,
                                      forecast_horizon=time_horizon)
