from src.moments.mean import credibilistic_mean
from src.moments.cvar import cvar
from src.moments.skewness import skewness
from src.moments.semikurtosis import semikurtosis
from src.config import CVAR_CONFIDENCE


def model3_objectives(fuzzy):
    e = credibilistic_mean(fuzzy)

    return [
        -e,
        cvar(fuzzy, CVAR_CONFIDENCE),
        -skewness(fuzzy, e),
        semikurtosis(fuzzy, e)
    ]
