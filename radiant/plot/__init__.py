__all__ = [
    "ml_animate",
    "many",
]


from matplotlib.pylab import rcParams
from .animate import ml_animate
from .plot import many


rcParams['axes.xmargin'] = 0
