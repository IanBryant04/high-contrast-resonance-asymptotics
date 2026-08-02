# dataset_generator.py
# Generates training data for AI inverse design
# Output:
#     inverse_design_dataset.csv
# Columns:
#     eta1
#     beta1
#     r1
#     eta2
#     beta2
#     r2
#     chi_eff
#     beta_eff
#     lambda0
#     solver_success

import numpy as np
import pandas as pd
from tqdm import tqdm

from Inverse_Design_Nonlinear import (
    make_grid,
    solve_mode,
    effective_response,
)



N_SAMPLES = 1000

GRID_POINTS = 300

ETA1_MIN = 0.5
ETA1_MAX = 5.0

BETA1_MIN = 0.0
BETA1_MAX = 5.0

R1_MIN = 0.10
R1_MAX = 0.90

ETA2 = 1.0
BETA2 = 0.5
R2 = 1.0

OUTPUT_FILE = "inverse_design_dataset.csv"

SEED = 12345



def generate_sample(rng):

    eta1 = rng.uniform(ETA1_MIN, ETA1_MAX)

    beta1 = rng.uniform(BETA1_MIN, BETA1_MAX)

    r1 = rng.uniform(R1_MIN, R1_MAX)

    r, w = make_grid(GRID_POINTS, R2)

    try:

        u, lam, z, ier = solve_mode(
            eta1,
            ETA2,
            beta1,
            BETA2,
            r,
            w,
            r1,
            z0=None,
        )

        success = int(ier == 1)

        eff = effective_response(
            u,
            r,
            w,
            r1,
            eta1,
            ETA2,
            beta1,
            BETA2,
        )

        row = {
            "eta1": eta1,
            "beta1": beta1,
            "r1": r1,

            "eta2": ETA2,
            "beta2": BETA2,
            "r2": R2,

            "chi_eff": eff["chi_eff"],
            "beta_eff": eff["beta_eff"],

            "lambda0": lam,

            "solver_success": success,
        }

    except Exception as e:

        row = {
            "eta1": eta1,
            "beta1": beta1,
            "r1": r1,

            "eta2": ETA2,
            "beta2": BETA2,
            "r2": R2,

            "chi_eff": np.nan,
            "beta_eff": np.nan,

            "lambda0": np.nan,

            "solver_success": 0,
        }

    return row


def main():

    rng = np.random.default_rng(SEED)

    rows = []

    print()
    print("======================================")
    print("GENERATING INVERSE DESIGN DATASET")
    print("======================================")
    print()

    print(f"Samples      : {N_SAMPLES}")
    print(f"Grid Points  : {GRID_POINTS}")
    print(f"Output File  : {OUTPUT_FILE}")
    print()

    for _ in tqdm(range(N_SAMPLES)):

        row = generate_sample(rng)

        rows.append(row)

    df = pd.DataFrame(rows)

    success_rate = (
        100.0
        * df["solver_success"].sum()
        / len(df)
    )

    print()
    print("======================================")
    print("DATASET SUMMARY")
    print("======================================")
    print()

    print("Total Samples :", len(df))
    print("Success Rate  : %.2f%%" % success_rate)

    print()

    print("Saving CSV...")

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("Done.")
    print()
    print("Saved:", OUTPUT_FILE)
    print()

    print(df.head())


if __name__ == "__main__":
    main()