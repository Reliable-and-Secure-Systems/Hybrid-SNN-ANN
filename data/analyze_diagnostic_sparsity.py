import numpy as np

from data.atm_dataset import load_atm_diagnostic_dataset


# ---------------------------------------------------------
# LOAD DIAGNOSTIC EEG DATA
# ---------------------------------------------------------

X, y, subjects, files = load_atm_diagnostic_dataset()


# ---------------------------------------------------------
# OVERALL SPARSITY
# ---------------------------------------------------------

zero_percentage = np.mean(X == 0) * 100

nonzero_counts = np.sum(
    X != 0,
    axis=(1, 2)
)

print()
print("=" * 70)
print("DIAGNOSTIC EEG SPARSITY")
print("=" * 70)

print(
    f"Dataset shape: {X.shape}"
)

print(
    f"Overall zero percentage: "
    f"{zero_percentage:.2f}%"
)

print(
    f"Nonzero values per sample - "
    f"mean: {np.mean(nonzero_counts):.2f}"
)

print(
    f"Nonzero values per sample - "
    f"median: {np.median(nonzero_counts):.2f}"
)

print(
    f"Nonzero values per sample - "
    f"max: {np.max(nonzero_counts)}"
)


# ---------------------------------------------------------
# SPARSITY BY GROUP
# ---------------------------------------------------------

print()
print("=" * 70)
print("SPARSITY BY DIAGNOSTIC GROUP")
print("=" * 70)

for label, group in [
    (0, "HC"),
    (1, "MCI"),
    (2, "AD")
]:

    group_data = X[y == label]

    group_zero_percentage = (
        np.mean(group_data == 0) * 100
    )

    group_nonzero_counts = np.sum(
        group_data != 0,
        axis=(1, 2)
    )

    print()
    print(f"{group}:")

    print(
        f"  Samples: "
        f"{len(group_data)}"
    )

    print(
        f"  Zero percentage: "
        f"{group_zero_percentage:.2f}%"
    )

    print(
        f"  Mean nonzero values: "
        f"{np.mean(group_nonzero_counts):.2f}"
    )

    print(
        f"  Median nonzero values: "
        f"{np.median(group_nonzero_counts):.2f}"
    )

    print(
        f"  Max nonzero values: "
        f"{np.max(group_nonzero_counts)}"
    )


# ---------------------------------------------------------
# NONZERO DISTRIBUTION
# ---------------------------------------------------------

print()
print("=" * 70)
print("NONZERO DISTRIBUTION")
print("=" * 70)

print(
    f"Samples with zero nonzero values: "
    f"{np.sum(nonzero_counts == 0)}"
)

print(
    f"Samples with <= 10 nonzero values: "
    f"{np.sum(nonzero_counts <= 10)}"
)

print(
    f"Samples with > 10 nonzero values: "
    f"{np.sum(nonzero_counts > 10)}"
)

print("=" * 70)