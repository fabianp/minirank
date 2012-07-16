import numpy as np

# Author : Alexandre Gramfort
# license : BSD


def pav(y):
    """
    PAV uses the pair adjacent violators method to produce a monotonic
    smoothing of y

    translated from matlab by Sean Collins (2006) as part of the EMAP toolbox
    """
    y = np.asarray(y)
    assert y.ndim == 1
    n_samples = len(y)
    v = y.copy()
    lvls = np.arange(n_samples)
    lvlsets = np.c_[lvls, lvls]
    flag = 1
    while flag:
        deriv = np.diff(v)
        if np.all(deriv >= 0):
            break

        viol = np.where(deriv < 0)[0]
        start = lvlsets[viol[0], 0]
        last = lvlsets[viol[0] + 1, 1]
        s = 0
        n = last - start + 1
        for i in range(start, last + 1):
            s += v[i]

        val = s / n
        for i in range(start, last + 1):
            v[i] = val
            lvlsets[i, 0] = start
            lvlsets[i, 1] = last
    return v


if __name__ == '__main__':
    dat = np.arange(10).astype(np.float)
    dat += 2 * np.random.randn(10)  # add noise

    dat_hat = pav(dat)

    import pylab as pl
    pl.close('all')
    pl.plot(dat, 'rx')
    pl.plot(dat_hat, 'b')
    pl.show()
