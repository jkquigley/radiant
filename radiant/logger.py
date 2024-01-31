def id(*args, **kwargs):
    pass


class Logger:
    def __init__(self, callback=None):
        if callback is None:
            callback = id

        self.callback = callback

    def tick(self, *args, **kwargs):
        self.callback(*args, **kwargs)
