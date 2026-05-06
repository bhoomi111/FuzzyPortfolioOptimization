from src.moments.mean import credibilistic_mean
from src.moments.semivariance import semivariance
from src.moments.skewness import skewness
from src.moments.semikurtosis import semikurtosis


def model1_objectives(fuzzy):
    e = credibilistic_mean(fuzzy)

    return [
        -e,
        semivariance(fuzzy, e),
        -skewness(fuzzy, e),
        semikurtosis(fuzzy, e)
    ]
