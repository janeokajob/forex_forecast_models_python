from logging.config import dictConfig
from statistics import variance
from warnings import filterwarnings

from meanwhile import Job
from pandas import DataFrame

from source.garch.base import ModelParams
from source.garch.simulator import GarchSimulator

filterwarnings("ignore")

dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})


class GarchParameterEstimator:
    analysis_results = []
    data_frame: DataFrame = None

    @staticmethod
    def initialize(analysis_results: list = None, data_frame: DataFrame = None):
        GarchParameterEstimator.analysis_results = analysis_results
        GarchParameterEstimator.data_frame = data_frame

    @staticmethod
    def simulate(param: ModelParams = ModelParams()):
        print("* * *   simulation using parameter: {}    * * *".format(param))
        try:
            model = GarchSimulator(trading_data=GarchParameterEstimator.data_frame, params=param)
            _results = model.forecast(100)
            GarchParameterEstimator.analyse_results(_results, param)
        except Exception as e:
            print("Simulation error:", e)

    @staticmethod
    def generate_param_set(max_range: int = 2):
        _param_set = []
        for ar in range(0, max_range):
            for ma in range(0, max_range):
                _params = ModelParams(ar=ar, ma=ma, p=1, q=1)
                _param_set.append(_params)
        _param_set.sort(key=lambda item: (item.to_dict()["order_pq"], item.to_dict()["order_arma"]))
        return _param_set

    @staticmethod
    def analyse_results(results: list = None, param: ModelParams = ModelParams()):
        if len(results) == 2:
            predictions = results[0]
            errors = results[0]
            prediction_variance = variance(predictions)
            error_range = max(errors) - min(errors)
            GarchParameterEstimator.analysis_results.append({
                "params": param,
                "prediction_variance": prediction_variance,
                "error_range": error_range
            })
            GarchParameterEstimator.print_results()

    @staticmethod
    def print_results():
        sorted_results = sorted(GarchParameterEstimator.analysis_results,
                                key=lambda item: (item["error_range"], -item["prediction_variance"]))
        print("\n\n============================================\n")
        print("ideal params:", "\n".join([str(it) for it in sorted_results]))
        print("\n============================================\n\n")

    @staticmethod
    def find_optimal_parameters(max_range: int = 1, thread_count: int = 12):
        # create the param set
        param_set = GarchParameterEstimator.generate_param_set(max_range)

        # concurrent threads
        job = Job(GarchParameterEstimator.simulate, thread_count)
        job.add_many(param_set)
        job.wait()
        job.get_results()

        GarchParameterEstimator.print_results()

    @staticmethod
    def verify_parameters(params: list):
        for _param in params:
            GarchParameterEstimator.simulate(_param)


def estimate_garch_parameters(df: DataFrame):
    # find optimal parameters
    max_range: int = 1
    thread_count: int = 6
    GarchParameterEstimator.initialize(analysis_results=[], data_frame=df)
    GarchParameterEstimator.find_optimal_parameters(max_range=max_range, thread_count=thread_count)

    # verify optimal parameters
    optimal_params = []
    GarchParameterEstimator.verify_parameters(optimal_params)
