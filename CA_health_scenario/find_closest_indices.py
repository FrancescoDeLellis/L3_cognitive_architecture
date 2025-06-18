def find_closest_indices(time_array, targets):
    """
    Find the indices in time_array closest to each value in targets.
    Assumes time_array is monotonic (increasing or decreasing).
    """
    closest_indices = []

    for tgt in targets:
        closest_idx = 0
        min_diff = abs(time_array[0] - tgt)
        for i in range(1, len(time_array)):
            diff = abs(time_array[i] - tgt)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
            else:  break   # time_array is monotonic, so once diff starts increasing, we passed closest point
        closest_indices.append(closest_idx)

    return closest_indices