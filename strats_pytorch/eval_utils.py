from sklearn.metrics import (
    accuracy_score,
    d2_tweedie_score,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)

from strats_pytorch.utils import reg_to_classif


def get_metrics(y_true, y_pred):
    """mae, medae, r2, gamma d2, acc, f1"""
    y_pred_classif = reg_to_classif(y_pred)
    y_true_classif = reg_to_classif(y_true)

    try:
        d2 = d2_tweedie_score(y_true, y_pred, power=2.0)
    except ValueError:
        d2 = 0.0

    return (
        mean_absolute_error(y_true, y_pred),
        median_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred),
        d2,
        accuracy_score(y_true_classif, y_pred_classif),
        f1_score(y_true_classif, y_pred_classif, average="weighted"),
    )
