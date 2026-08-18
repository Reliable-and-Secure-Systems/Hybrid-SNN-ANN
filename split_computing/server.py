import torch

from models.atm_hybrid_model import ATMHybridSNNANN


INPUT_SIZE = 128


def reconstruct_features(
    selected_values,
    selected_indices,
    input_size=INPUT_SIZE
):
    """
    Reconstruct the original SNN feature vector
    from the transmitted selected values and indices.
    """

    reconstructed = torch.zeros(
        selected_values.size(0),
        input_size
    )

    reconstructed.scatter_(
        1,
        selected_indices,
        selected_values
    )

    return reconstructed


def run_server(
    selected_values,
    selected_indices
):

    # --------------------------------------------------
    # Reconstruct compressed SNN representation
    # --------------------------------------------------

    features = reconstruct_features(
        selected_values,
        selected_indices
    )

    # --------------------------------------------------
    # Load trained hybrid model
    # --------------------------------------------------

    model = ATMHybridSNNANN()

    model.load_state_dict(
        torch.load(
            "models/atm_hybrid_trained.pt",
            map_location="cpu"
        )
    )

    model.eval()

    # --------------------------------------------------
    # ANN classification
    # --------------------------------------------------

    with torch.no_grad():

        output = model.ann(
            features
        )

        prediction = output.argmax(
            dim=1
        )

    return prediction, output


if __name__ == "__main__":

    # Example transmitted data

    selected_values = torch.tensor(
        [[0.5, 0.3, 0.2, 0.1]]
    )

    selected_indices = torch.tensor(
        [[3, 20, 50, 100]]
    )

    prediction, output = run_server(
        selected_values,
        selected_indices
    )

    print("=" * 60)
    print("SERVER / ANN")
    print("=" * 60)

    print(
        "Received values:",
        selected_values.shape
    )

    print(
        "Received indices:",
        selected_indices.shape
    )

    print(
        "Reconstructed representation:",
        INPUT_SIZE
    )

    print(
        "ANN output:",
        output
    )

    print(
        "Prediction:",
        "LEFT" if prediction.item() == 0 else "RIGHT"
    )

    print("=" * 60)