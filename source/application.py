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


# garch_arma_parameters = {
#     "eur_cad": ModelParams(ar=5, ma=8),
#     "eur_gbp": ModelParams(ar=6, ma=4),
#     "eur_jpy": ModelParams(ar=1, ma=6),
#     "eur_usd": ModelParams(ar=5, ma=6)
# }

garch_arma_parameters = {  # ARIMA(p, d, q)(P, D, Q)
    "eur_cad": ModelParams(ar=2, ma=1),  # ARIMA(0,1,0)(2,1,0)[12]
    "eur_gbp": ModelParams(ar=3, ma=2),  # ARIMA(0,1,0)(0,1,2)[12]
    "eur_jpy": ModelParams(ar=1, ma=2),  # ARIMA(0,1,0)(0,1,1)[12]
    "eur_usd": ModelParams(ar=1, ma=2)  # ARIMA(0,1,1)(0,1,2)[12]
}


@unique
class ForecastModel(Enum):
    ARIMA = 1
    GARCH = 2
    PROPHET = 3


def forecast_trading_data(file_name: str = None, currencies: str = None,
                          model: ForecastModel = ForecastModel.PROPHET):
    raw_data = DatasetLoader.load(file_name)
    trading_data = DataFrame.from_records([item.to_dict() for item in raw_data])
    forecast_horizon = 96

    # run forecast model
    simulator: Simulator
    if ForecastModel.PROPHET == model:
        simulator = ProphetSimulator(trading_data=trading_data, currency_pair=currencies)
    elif ForecastModel.GARCH == model:
        simulator = GarchSimulator(trading_data=trading_data,
                                   currency_pair=currencies,
                                   params=garch_arma_parameters[currencies])  # params=garch_arma_parameters[currencies]

    else:
        simulator = ArimaSimulator(trading_data=trading_data, currency_pair=currencies)

    simulator.forecast(forecast_horizon)

    print("forecasts:")
    print(simulator.forecasts)

    simulator.evaluate_forecast()
    # simulator.plot_source_dataset()

    print("\n\nForecast Evaluation:")
    print("Regression Metrics: {} Forecasts for {}".format(simulator.model_name, simulator.currency_pair.upper()))
    DataWriter.write(f"output/{simulator.currency_pair}__{simulator.model_name.lower()}__evaluation.txt",
                     "\n".join([f"{key}\t{value:0.6f}" for key, value in simulator.metrics.items()]))
    print(tabulate([[_key, f"{value:0.6f}"] for _key, value in simulator.metrics.items()]))


if __name__ == "__main__":
    # load raw input
    # https://uk.investing.com/currencies/eur-usd-historical-data
    trading_data_files = {"eur_cad": "input/eur_cad_trading_data.csv",
                          "eur_gbp": "input/eur_gbp_trading_data.csv",
                          "eur_jpy": "input/eur_jpy_trading_data.csv",
                          "eur_usd": "input/eur_usd_trading_data.csv"}

    # trading_data_files = {"eur_cad": "input/eur_cad_trading_data.csv"}

    # models = [ForecastModel.PROPHET, ForecastModel.GARCH, ForecastModel.ARIMA]
    models = [ForecastModel.GARCH]

    for model in models:
        for key in trading_data_files.keys():
            # load input frames
            data_file = trading_data_files[key]
            currency_pair = key
            forecast_trading_data(file_name=data_file, currencies=currency_pair, model=model)
