from src.moments.mean import credibilistic_mean
from src.moments.masd import masd
from src.moments.skewness import skewness
from src.moments.semikurtosis import semikurtosis


def model2_objectives(fuzzy):
    e = credibilistic_mean(fuzzy)

    return [
        -e,
        masd(fuzzy),
        -skewness(fuzzy, e),
        semikurtosis(fuzzy, e)
    ]