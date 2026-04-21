from src.moments.mean import credibilistic_mean
from src.moments.semivariance import semivariance
from src.moments.skewness import skewness
from src.moments.semikurtosis import semikurtosis


def compute_objectives(fuzzy):
    e = credibilistic_mean(fuzzy)

    return {
        "mean": e,
        "risk": semivariance(fuzzy, e),
        "skewness": skewness(fuzzy, e),
        "kurtosis": semikurtosis(fuzzy, e),
    }


def fitness_vector(obj):
    return [
        -obj["mean"],        # maximize
        obj["risk"],         # minimize
        -obj["skewness"],    # maximize
        obj["kurtosis"],     # minimize
    ]