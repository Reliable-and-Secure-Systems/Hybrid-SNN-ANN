import numpy as np


def encode_atm_latency(sample, time_steps=20):
    """
    Latency encoding for a 68 x 68 ATM EEG matrix.

    Zero values produce no spike.
    Non-zero values produce one spike.
    Larger values spike earlier.
    Smaller values spike later.

    Output:
        spikes: time_steps x 4624
    """

    sample = np.clip(sample, 0.0, 1.0)

    features = sample.reshape(-1)

    spikes = np.zeros(
        (time_steps, features.shape[0]),
        dtype=np.float32
    )

    nonzero_indices = np.flatnonzero(features)
    nonzero_values = features[nonzero_indices]

    # Larger value -> earlier timestep
    spike_times = (
        (1.0 - nonzero_values) * (time_steps - 1)
    ).astype(int)

    spikes[
        spike_times,
        nonzero_indices
    ] = 1.0

    return spikes


if __name__ == "__main__":

    sample = np.zeros((68, 68), dtype=np.float32)

    sample[10, 20] = 0.5
    sample[20, 30] = 1.0
    sample[40, 40] = 0.25

    spikes = encode_atm_latency(
        sample,
        time_steps=20
    )

    print("=" * 50)
    print("ATM LATENCY ENCODER TEST")
    print("=" * 50)

    print("Original shape :", sample.shape)
    print("Spike shape    :", spikes.shape)

    print(
        "Original non-zero values:",
        np.count_nonzero(sample)
    )

    print(
        "Generated spikes:",
        int(spikes.sum())
    )

    print(
        "Spikes per timestep:",
        spikes.sum(axis=1).astype(int)
    )

    print("=" * 50)
