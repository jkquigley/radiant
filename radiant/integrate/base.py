class BaseIntegrator:
    def __init__(self, a, b, accuracy):
        self.a = a
        self.b = b
        self.accuracy = accuracy

    def __call__(self, func):
        raise NotImplementedError
