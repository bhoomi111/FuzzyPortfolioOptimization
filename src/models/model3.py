from src.moments.mean import credibilistic_mean
from src.moments.cvar import cvar
from src.moments.skewness import skewness
from src.moments.semikurtosis import semikurtosis


def model3_objectives(fuzzy):
    e = credibilistic_mean(fuzzy)

    return [
        -e,
        cvar(fuzzy),
        -skewness(fuzzy, e),
        semikurtosis(fuzzy, e)
    ]