import numpy as np

def rate_encode(signal):

    signal = (signal - signal.min()) / (
        signal.max() - signal.min() + 1e-8
    )

    spikes = np.random.rand(*signal.shape) < signal

    return spikes.astype(np.float32)