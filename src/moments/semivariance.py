def semivariance(fuzzy, e):
    b1, b2, b3, k = fuzzy.b1, fuzzy.b2, fuzzy.b3, fuzzy.k

    alpha = b2 - b1
    beta = b3 - b2

    rho1 = e - b1
    rho2 = b2 - e
    rho3 = b3 - e

    c1 = 1 / (k + 1)
    c2 = 1 / ((k + 1) * (k + 2))
    c2p = 1 / ((1 + k) * (1 + 2 * k))

    if b1 <= e <= b2:
        return (k**2 * rho1 ** (1 / k + 2) * c2p) / (alpha ** (1 / k))
    else:
        L1 = (
            -k * alpha * rho2 * c1
            + k**2 * alpha**2 * c2p
            + rho2**2
            + rho2 * beta * c1
            + beta**2 * c2
        )
        return L1 - (rho3 ** (k + 2) * c2) / (beta**k)
