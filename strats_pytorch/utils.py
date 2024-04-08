def denorm(val, mean, std):
    """Denormalize val using the provided mean and standard deviation"""
    if std != 0:
        return (val * std) + mean
    return val + mean


def norm(val, mean, std):
    """Normalize val using the provided mean and standard deviation"""
    if std != 0:
        return (val - mean) / std
    return val - mean


def value_to_index(vals, cast_from_numpy=False, return_key=False):
    """Associate every different value in vals to an index,
    starting from zero, in order of appearance"""
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
