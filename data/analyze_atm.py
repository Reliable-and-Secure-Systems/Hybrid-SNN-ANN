import numpy as np

from data.atm_dataset import load_atm_dataset


# Load ATM dataset
X, y = load_atm_dataset()

print("=" * 60)
print("ATM DATA ANALYSIS")
print("=" * 60)

print("Dataset shape:", X.shape)
print("Labels shape :", y.shape)

# Take first trial
sample = X[0]

print("\nFIRST TRIAL")
print("Shape:", sample.shape)
print("Label:", "LEFT" if y[0] == 0 else "RIGHT")

# Basic statistics
print("\nSTATISTICS")
print("Minimum:", sample.min())
print("Maximum:", sample.max())
print("Mean   :", sample.mean())
print("Std    :", sample.std())

# Sparsity
nonzero = np.count_nonzero(sample)
total = sample.size

print("\nSPARSITY")
print("Total values    :", total)
print("Non-zero values :", nonzero)
print("Zero values     :", total - nonzero)
print("Non-zero ratio  :", nonzero / total)
print("Zero ratio      :", 1 - (nonzero / total))

# Unique values
unique_values = np.unique(sample)

print("\nUNIQUE VALUES")
print("Number of unique values:", len(unique_values))
print("First values:", unique_values[:20])

# Row and column activity
row_activity = np.count_nonzero(sample, axis=1)
column_activity = np.count_nonzero(sample, axis=0)

print("\nROW ACTIVITY")
print("Minimum:", row_activity.min())
print("Maximum:", row_activity.max())
print("Mean   :", row_activity.mean())

print("\nCOLUMN ACTIVITY")
print("Minimum:", column_activity.min())
print("Maximum:", column_activity.max())
print("Mean   :", column_activity.mean())

# Symmetry check
symmetry_difference = np.abs(sample - sample.T).mean()

print("\nMATRIX STRUCTURE")
print("Mean |A - A.T|:", symmetry_difference)

print("=" * 60)