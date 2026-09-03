import pandas as pd


RESULT_PATH = "results/design_space_results.csv"
PARETO_PATH = "results/pareto_results.csv"


def find_pareto_optimal(df):
    """
    Find configurations that are not dominated by another
    configuration.

    Objectives:
        - maximize accuracy
        - minimize transmitted bits
    """

    pareto = []

    for i, row in df.iterrows():

        dominated = False

        for j, other in df.iterrows():

            if i == j:
                continue

            better_or_equal_accuracy = (
                other["accuracy"] >= row["accuracy"]
            )

            lower_or_equal_bits = (
                other["transmitted_bits"] <= row["transmitted_bits"]
            )

            strictly_better = (
                other["accuracy"] > row["accuracy"]
                or
                other["transmitted_bits"] < row["transmitted_bits"]
            )

            if (
                better_or_equal_accuracy
                and lower_or_equal_bits
                and strictly_better
            ):
                dominated = True
                break

        if not dominated:
            pareto.append(row)

    return pd.DataFrame(pareto)


# ============================================================
# Load results
# ============================================================

df = pd.read_csv(
    RESULT_PATH
)

print("=" * 70)
print("PARETO ANALYSIS")
print("=" * 70)

print()
print("Total configurations:", len(df))


# ============================================================
# Find Pareto-optimal configurations
# ============================================================

pareto_df = find_pareto_optimal(df)

pareto_df = pareto_df.sort_values(
    by=[
        "transmitted_bits",
        "accuracy"
    ],
    ascending=[
        True,
        False
    ]
)


# ============================================================
# Save
# ============================================================

pareto_df.to_csv(
    PARETO_PATH,
    index=False
)


# ============================================================
# Display
# ============================================================

print()
print(
    "Pareto-optimal configurations:",
    len(pareto_df)
)

print()
print(
    pareto_df[
        [
            "time_steps",
            "beta",
            "split_point",
            "top_k",
            "quantization_bits",
            "accuracy",
            "transmitted_bits",
            "feature_reduction_percent",
            "bit_reduction_percent"
        ]
    ].to_string(index=False)
)

print()
print(
    "Saved to:",
    PARETO_PATH
)