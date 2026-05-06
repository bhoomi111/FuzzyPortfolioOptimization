def skewness(fuzzy, e):
    b1, b2, b3, k = fuzzy.b1, fuzzy.b2, fuzzy.b3, fuzzy.k

    alpha = b2 - b1
    beta = b3 - b2

    rho1 = e - b1
    rho2 = b2 - e

    c1 = 1 / (k + 1)
    c2 = 1 / ((k + 1) * (k + 2))
    c3 = 1 / ((k + 1) * (k + 2) * (k + 3))

    c1p = c1
    c2p = 1 / ((1 + k) * (1 + 2 * k))
    c3p = 1 / ((1 + k) * (1 + 2 * k) * (1 + 3 * k))

    M1 = rho2**2 * c1 + 2 * beta * rho2 * c2 + 2 * beta**2 * c3
    M2 = rho2**2 * c1p - 2 * k * alpha * rho2 * c2p + 2 * k**2 * alpha**2 * c3p

    return 1.5 * (beta * M1 - k * alpha * M2 + (2 * rho2**3) / 3)
