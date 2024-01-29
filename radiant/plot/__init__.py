__all__ = [
    "ml_animate",
    "overlay",
]


from matplotlib.pylab import rcParams
from .animate import ml_animate
from .plot import overlay


rcParams['axes.xmargin'] = 0
