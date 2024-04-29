def denorm(val, mean, std):
    if std != 0:
        return (val * std) + mean
    return val + mean


def denorm_list(vals, mean, std):
    return [denorm(val, mean, std) for val in vals]


def norm(val, mean, std):
    if std != 0:
        return (val - mean) / std
    return val - mean


def descale(val, min, max):
    if min != max:
        return (val * (max - min)) + min
    return val + min


def scale(val, min, max):
    if min != max:
        return (val - min) / (max - min)
    return val - min


def value_to_index(vals, cast_from_numpy=False, return_key=False):
    d = {}
    indexes = []
    free_index = 0  # the lowest unused index
    for id in vals:
        if cast_from_numpy:
            id = id.item()
        if id not in d.keys():
            # if the id is new
            # allocate to the id the free index
            d[id] = free_index
            free_index += 1
        indexes.append(d[id])
    if return_key:
        return indexes, d
    return indexes


def creat_to_4_stages(value: float) -> int:
    """Converts creatinine values to renal risk/injury/failure stages
    according to the KDIGO criteria

    Parameters
    ----------
    value : float
        Creatinine (serum) value, in mg/dL

    Returns
    -------
    int
        0: Normal; 1: Risk; 2: Injury; 3: Failure
    """
    if value < 1.35:
        return 0
    elif value < 2.68:
        return 1
    elif value < 4.16:
        return 2
    return 3


def reg_to_classif(reg_vals):
    return [creat_to_4_stages(val) for val in reg_vals]
