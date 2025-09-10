import numpy as np


def top_n_rows(arr, n=1, col_idx=4, keepdims=False):
    """
    Return the top-n rows of `arr` sorted descending by column `col_idx`.

    Parameters
    ----------
    arr : np.ndarray
        2D array to select from.
    n : int
        Number of rows to select (default 1).
    col_idx : int
        Column index to sort by (default 4).
    keepdims : bool
        If False and n==1, return a 1D row. If True, always return a 2D array.
    """
    if arr.size == 0:
        return None

    n = min(n, arr.shape[0])
    col = arr[:, col_idx]
    col_cmp = np.where(np.isnan(col), -np.inf, col)

    idx = np.argpartition(col_cmp, -n)[-n:]
    idx = idx[np.argsort(col_cmp[idx])[::-1]]
    result = arr[idx]

    if n == 1 and not keepdims:
        return result[0]
    return result
