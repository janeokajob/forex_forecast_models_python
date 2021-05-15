from reader import Reader

from source.common.data import ForexData


class DataReader:
    @staticmethod
    def read_file(filename: str):
        dataset = []
        with Reader.openWithName(filename) as file:
            for line in file:
                if "OPEN" in line or "Open" in line:
                    continue
                try:
                    trading_data = ForexData.from_list(line)
                    dataset.append(trading_data)
                except Exception as e:
                    print("Error parsing data entry", e)
        return dataset

    @staticmethod
    def read_string(_text: str):
        dataset = []
        trading_data = ForexData.from_string(_text)
        dataset.append(trading_data)
        return dataset


class DataWriter:
    @staticmethod
    def write(filename: str, _text: str):
        f = open(filename, "w")
        f.write(_text)
        f.close()


class DatasetLoader:
    @staticmethod
    def load(data_file: str = None):
        raw_data = DataReader.read_file(data_file)
        print("raw input loaded ...")
        return raw_data
