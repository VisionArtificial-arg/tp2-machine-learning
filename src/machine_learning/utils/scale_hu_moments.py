import math


def scale_hu_moments(hu):
    eps = 1e-12
    for i in range(7):
        v = hu[i][0]
        if abs(v) < eps:
            hu[i][0] = 0.0
        else:
            hu[i][0] = -1 * math.copysign(1.0, v) * math.log10(abs(v))
    return hu