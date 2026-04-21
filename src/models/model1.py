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
def fitness_vector(obj):
    """
    Convert to minimization problem
    """
    return [
        -obj["mean"],       # maximize
        obj["risk"],        # minimize
        -obj["skewness"],   # maximize
        obj["kurtosis"],    # minimize
    ]