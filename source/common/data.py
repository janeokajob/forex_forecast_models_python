from datetime import datetime


class ForexData:
    def __init__(self, date: str = None, close: str = None, _open: str = None, high: str = None, low: str = None,
                 percent_change: str = None):
        date_format = '%Y-%m-%d'  # "%b %d, %Y"
        self.date = datetime.strptime(date, date_format).date()
        self.ds = self.date
        self.open = float(_open)
        self.high = float(high)
        self.low = float(low)
        self.close = float(close)
        self.percent_change = percent_change
        self.y = self.close

    @staticmethod
    def from_string(_data: str):
        tokens = _data.split(",")
        assert len(tokens) == 6
        return ForexData(*tokens)

    @staticmethod
    def from_list(_data: list):
        assert len(_data) == 6
        return ForexData(*_data)

    def to_dict(self):
        return {
            "date": self.date,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "percent_change": self.percent_change,
            "ds": self.date,
            "y": self.y
        }

    def __repr__(self):
        return str(", ".join([f"{key}: {value}" for key, value in self.to_dict().items()]))

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.date == other.date and self.close == other.close
