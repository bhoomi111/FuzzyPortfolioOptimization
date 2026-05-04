def semikurtosis(fuzzy, e):
    b1, b2, b3, k = fuzzy.b1, fuzzy.b2, fuzzy.b3, fuzzy.k

    alpha = b2 - b1
    beta = b3 - b2

    rho1 = e - b1
    rho2 = b2 - e
    rho3 = b3 - e

    c1 = 1 / (k + 1)
    c2 = 1 / ((k + 1) * (k + 2))
    c3 = 1 / ((k + 1) * (k + 2) * (k + 3))
    c4 = 1 / ((k + 1) * (k + 2) * (k + 3) * (k + 4))

    c2p = 1 / ((1 + k) * (1 + 2 * k))
    c3p = 1 / ((1 + k) * (1 + 2 * k) * (1 + 3 * k))
    c4p = 1 / ((1 + k) * (1 + 2 * k) * (1 + 3 * k) * (1 + 4 * k))

    if b1 <= e <= b2:
        return (12 * k**4 * rho1 ** (1 / k + 4) * c4p) / (alpha ** (1 / k))
    else:
        N1 = rho2**3 * c1 + 3 * beta * rho2**2 * c2 + 6 * beta**2 * rho2 * c3 + 6 * beta**3 * c4
        N2 = rho2**3 * c1 - 3 * k * alpha * rho2**2 * c2p + 6 * k**2 * alpha**2 * rho2 * c3p - 6 * k**3 * alpha**3 * c4p

        return 2 * beta * N1 - 2 * k * alpha * N2 + rho2**4 - (12 * rho3 ** (k + 4) * c4) / (beta**k)
