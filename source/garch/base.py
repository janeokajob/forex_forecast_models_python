class ModelParams:
    def __init__(self, ar: int = 2, ma: int = 1, p: int = 1, q: int = 1):
        self.AR = ar
        self.MA = ma
        self.p = p
        self.q = q
        self.order_pq = p + q
        self.order_arma = ar + ma

    def to_dict(self):
        return {
            "AR": self.AR,
            "MA": self.MA,
            "p": self.p,
            "q": self.q,
            "order_pq": self.order_pq,
            "order_arma": self.order_arma
        }

    def __repr__(self):
        values = self.to_dict()
        del values["order_pq"]
        del values["order_arma"]
        return str(values)
