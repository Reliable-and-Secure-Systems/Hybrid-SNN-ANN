import torch


def quantize_tensor(values, bits):
    """
    Symmetric uniform quantisation.

    Returns:
        quantized_values: integer representation
        scale: scale factor required for reconstruction
    """

    if bits not in (4, 8, 16):
        raise ValueError("bits must be one of: 4, 8, 16")

    max_value = values.abs().max()

    if max_value == 0:
        return torch.zeros_like(values, dtype=torch.int32), torch.tensor(1.0)

    max_int = (2 ** (bits - 1)) - 1

    scale = max_value / max_int

    quantized = torch.round(values / scale).clamp(
        -max_int,
        max_int
    )

    return quantized.to(torch.int32), scale


def dequantize_tensor(quantized_values, scale):
    """
    Reconstruct the approximate floating-point values.
    """

    return quantized_values.float() * scale