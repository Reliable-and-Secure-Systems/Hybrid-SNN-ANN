import numpy as np


def encode_atm_sample(sample, time_steps=20, seed=42):
    """
    Convert one 68 x 68 ATM representation into
    a temporal spike representation.

    Input:
        sample: 68 x 68 matrix with values between 0 and 1

    Output:
        spikes: time_steps x 4624
    """

    rng = np.random.default_rng(seed)

    # Make sure values are between 0 and 1
    sample = np.clip(sample, 0.0, 1.0)

    # Flatten the 68 x 68 brain-region representation
    features = sample.reshape(-1)

    # Repeat the representation across time and
    # generate spikes according to feature magnitude
    random_values = rng.random(
        (time_steps, features.shape[0])
    )

    spikes = random_values < features

    return spikes.astype(np.float32)


if __name__ == "__main__":

    # Test with a sparse example
    sample = np.zeros((68, 68), dtype=np.float32)

    sample[10, 20] = 0.5
    sample[20, 30] = 1.0
    sample[40, 40] = 0.25

    spikes = encode_atm_sample(
        sample,
        time_steps=20
    )

    print("=" * 50)
    print("ATM SPIKE ENCODER TEST")
    print("=" * 50)

    print("Original shape :", sample.shape)
    print("Spike shape    :", spikes.shape)

    print("Original non-zero values:",
          np.count_nonzero(sample))

    print("Generated spikes:",
          int(spikes.sum()))

    print("Spikes per timestep:",
          spikes.sum(axis=1).astype(int))

    print("=" * 50)