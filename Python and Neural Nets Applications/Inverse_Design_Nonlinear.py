
"""
Inverse_Design_Nonlinear.py
Direct Python translation of Inverse_Design_Nonlinear.m

Requires:
    numpy
    scipy

Run:
    python Inverse_Design_Nonlinear.py
"""

import numpy as np
from scipy.optimize import fsolve


def make_grid(N=600, r2=1.0):
    r = np.linspace(0.0, r2, N)
    dr = r[1] - r[0]
    w = np.ones(N) * dr
    w[0] = dr / 2.0
    w[-1] = dr / 2.0
    return r, w


def F_nonlinear(z, r, w, r1, eta1, eta2, beta1, beta2):
    N = len(r)

    u = z[:N]
    lam = z[N]

    F = np.zeros(N + 1)

    u_r = w * u * r
    u_r2 = w * u * r**2

    u3 = u**3
    u3_r = w * u3 * r
    u3_r2 = w * u3 * r**2

    F[0] = u[0] - u[1]

    for i in range(1, N):

        ri = r[i]

        if ri <= r1:

            idx0 = r <= ri
            idx1 = (r > ri) & (r <= r1)
            idx2 = r > r1

            term_in = (
                eta1 * (
                    (1.0 / ri) * np.sum(u_r2[idx0])
                    + np.sum(u_r[idx1])
                )
                +
                beta1 * (
                    (1.0 / ri) * np.sum(u3_r2[idx0])
                    + np.sum(u3_r[idx1])
                )
            )

            term_out = (
                eta2 * np.sum(u_r[idx2])
                +
                beta2 * np.sum(u3_r[idx2])
            )

        else:

            idx0 = r <= r1
            idx1 = (r > r1) & (r <= ri)
            idx2 = r > ri

            term_in = (
                eta1 * (1.0 / ri) * np.sum(u_r2[idx0])
                +
                beta1 * (1.0 / ri) * np.sum(u3_r2[idx0])
            )

            term_out = (
                eta2 * (
                    (1.0 / ri) * np.sum(u_r2[idx1])
                    + np.sum(u_r[idx2])
                )
                +
                beta2 * (
                    (1.0 / ri) * np.sum(u3_r2[idx1])
                    + np.sum(u3_r[idx2])
                )
            )

        rhs = lam * (term_in + term_out)

        F[i] = u[i] - rhs

    F[N] = np.sum(u_r2) - 1.0 / (4.0 * np.pi)

    return F


def solve_mode(
    eta1,
    eta2,
    beta1,
    beta2,
    r,
    w,
    r1,
    z0=None,
):
    N = len(r)

    if z0 is None:
        z0 = np.concatenate([2.0 * np.ones(N), [2.5]])

    z, info, ier, msg = fsolve(
        F_nonlinear,
        z0,
        args=(r, w, r1, eta1, eta2, beta1, beta2),
        full_output=True,
        xtol=1e-10,
    )

    return z[:N], z[N], z, ier


def effective_response(
    u,
    r,
    w,
    r1,
    eta1,
    eta2,
    beta1,
    beta2,
):

    norm_val = np.sqrt(
        4.0 * np.pi * np.sum(w * (u**2) * (r**2))
    )

    u = u / norm_val

    idx_in = r <= r1
    idx_out = r > r1

    U1 = 4.0 * np.pi * np.sum(
        w[idx_in] * u[idx_in] * r[idx_in]**2
    )

    U2 = 4.0 * np.pi * np.sum(
        w[idx_out] * u[idx_out] * r[idx_out]**2
    )

    Ub1 = 4.0 * np.pi * np.sum(
        w[idx_in] * (u[idx_in]**3) * r[idx_in]**2
    )

    Ub2 = 4.0 * np.pi * np.sum(
        w[idx_out] * (u[idx_out]**3) * r[idx_out]**2
    )

    chi_eff = (eta1 * U1 + eta2 * U2) / (U1 + U2)

    beta_eff = (beta1 * Ub1 + beta2 * Ub2) / (Ub1 + Ub2)

    return {
        "U1": U1,
        "U2": U2,
        "Ub1": Ub1,
        "Ub2": Ub2,
        "chi_eff": chi_eff,
        "beta_eff": beta_eff,
    }


def inverse_design(
    target_chi,
    target_beta,
    r1,
    eta2=1.0,
    beta2=0.5,
    eta1_0=2.0,
    beta1_0=2.0,
    N=300,
    r2=1.0,
    max_iter=50,
    tol=1e-6,
    relax=0.5,
):

    r, w = make_grid(N, r2)

    eta1 = eta1_0
    beta1 = beta1_0

    z = None

    for k in range(max_iter):

        u, lam, z, ier = solve_mode(
            eta1,
            eta2,
            beta1,
            beta2,
            r,
            w,
            r1,
            z,
        )

        eff = effective_response(
            u,
            r,
            w,
            r1,
            eta1,
            eta2,
            beta1,
            beta2,
        )

        eta1_new = (
            target_chi * (eff["U1"] + eff["U2"])
            - eta2 * eff["U2"]
        ) / eff["U1"]

        beta1_new = (
            target_beta * (eff["Ub1"] + eff["Ub2"])
            - beta2 * eff["Ub2"]
        ) / eff["Ub1"]

        err = max(
            abs(eta1_new - eta1),
            abs(beta1_new - beta1),
        )

        print(
            f"{k+1:3d} "
            f"eta1={eta1:.10f} "
            f"beta1={beta1:.10f} "
            f"err={err:.3e}"
        )

        if err < tol:
            return {
                "eta1": eta1_new,
                "beta1": beta1_new,
                "lambda": lam,
                "iters": k + 1,
                "converged": True,
            }

        eta1 = (1.0 - relax) * eta1 + relax * eta1_new
        beta1 = (1.0 - relax) * beta1 + relax * beta1_new

    return {
        "eta1": eta1,
        "beta1": beta1,
        "lambda": lam,
        "iters": max_iter,
        "converged": False,
    }


if __name__ == "__main__":

    r1 = 0.5
    r2 = 1.0

    target_eta_eff = 1.5
    target_beta_eff = 1.0

    result = inverse_design(
        target_eta_eff,
        target_beta_eff,
        r1=r1,
        r2=r2,
        eta2=1.0,
        beta2=0.5,
        eta1_0=2.0,
        beta1_0=2.0,
        N=300,
    )

    print()
    print(result)
