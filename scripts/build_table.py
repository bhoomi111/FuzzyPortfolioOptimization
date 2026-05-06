from src.moments.mean import credibilistic_mean
from src.moments.semivariance import semivariance
from src.moments.masd import masd
from src.moments.cvar import cvar
from src.moments.skewness import skewness
from src.moments.semikurtosis import semikurtosis

import pandas as pd

TABLE_COLUMNS = [
    "Rep",
    "Center position",
    "cp",
    "mp",
    "run",
    "Total solutions",
    "b1",
    "b2",
    "b3",
    "k",
    "Mean",
    "Risk",
    "Skewness",
    "Semikurtosis",
]


def build_table(results, model_type):
    if not results:
        return pd.DataFrame(columns=TABLE_COLUMNS)

    rows = []

    for i, r in enumerate(results):
        fuzzy = r["fuzzy"]

        e = credibilistic_mean(fuzzy)

        if model_type == "model1":
            risk = semivariance(fuzzy, e)
        elif model_type == "model2":
            risk = masd(fuzzy)
        elif model_type == "model3":
            risk = cvar(fuzzy)
        else:
            raise ValueError("Invalid model type")

        row = {
            "Rep": f"R{i+1}",
            "Center position": r.get("center_position"),
            "cp": r["cp"],
            "mp": r["mp"],
            "run": r.get("run", 1),
            "Total solutions": r.get("total_pareto_solutions"),
            "b1": fuzzy.b1,
            "b2": fuzzy.b2,
            "b3": fuzzy.b3,
            "k": fuzzy.k,
            "Mean": e,
            "Risk": risk,
            "Skewness": skewness(fuzzy, e),
            "Semikurtosis": semikurtosis(fuzzy, e),
        }

        rows.append(row)

    return pd.DataFrame(rows, columns=TABLE_COLUMNS)
