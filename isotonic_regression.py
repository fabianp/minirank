import numpy as np

# Author : Fabian Pedregosa <fabian@fseoane.net>


def isotonic_regression(w, y, x_min=None, x_max=None):
    """
    Solve the isotonic regression with complete ordering model:

        min_x Sum_i{ w_i (y_i - x_i) ** 2 }

        subject to x_min = x_1 <= x_2 ... <= x_n = x_max

    where each w_i is strictly positive and each y_i is an arbitrary
    real number.

    Parameters
    ----------
    w : iterable of floating-point values
    y : iterable of floating-point values

    Returns
    -------
    x : list of floating-point values
    """

    if x_min is not None or x_max is not None:
        y = np.copy(y)
        w = np.copy(w)
        C = np.dot(w, y * y) * 10 # upper bound on the cost function
        if x_min is not None:
            y[0] = x_min
            w[0] = C
        if x_max is not None:
            y[-1] = x_max
            w[-1] = C

    J = [(w[i] * y[i], w[i], [i,]) for i in range(len(y))]
    cur = 0

    while cur < len(J) - 1:
        v0, v1, v2 = 0, 0, np.inf
        w0, w1, w2 = 1, 1, 1
        while v0 * w1 <= v1 * w0 and cur < len(J) - 1:
            v0, w0, idx0 = J[cur]
            v1, w1, idx1 = J[cur + 1]
            if v0 * w1 <= v1 * w0:
                cur +=1

        if cur == len(J) - 1:
            break

        # merge two groups
        v0, w0, idx0 = J.pop(cur)
        v1, w1, idx1 = J.pop(cur)
        J.insert(cur, (v0 + v1, w0 + w1, idx0 + idx1))
        while v2 * w0 > v0 * w2 and cur > 0:
            v0, w0, idx0 = J[cur]
            v2, w2, idx2 = J[cur - 1]
            if w0 * v2 >= w2 * v0:
                J.pop(cur)
                J[cur - 1] = (v0 + v2, w0 + w2, idx0 + idx2)
                cur -= 1

    sol = np.empty(len(y))
    for v, w, idx in J:
        sol[idx] = v / w
    return sol


if __name__ == '__main__':
    dat = np.arange(10).astype(np.float)
    dat += 2 * np.random.randn(10)  # add noise
    w = np.ones(dat.shape)
    dat_hat = isotonic_regression(w, dat)

    import pylab as pl
    pl.close('all')
    pl.plot(dat, 'rx')
    pl.plot(dat_hat, 'b')
    pl.show()
