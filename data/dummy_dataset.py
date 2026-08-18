import numpy as np

def load_dummy_dataset():

    num_samples = 100
    num_channels = 32
    time_steps = 500
    num_classes = 2

    eeg = np.random.randn(
        num_samples,
        num_channels,
        time_steps
    )

    labels = np.random.randint(
        0,
        num_classes,
        size=num_samples
    )

    return eeg, labels