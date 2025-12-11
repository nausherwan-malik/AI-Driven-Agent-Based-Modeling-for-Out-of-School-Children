import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn


# LOAD MODELS (same as training)
class Model1(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4)
        )
    def forward(self, x): return self.net(x)


class Model2Strong(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
    def forward(self, x): return self.net(x)


class Model3(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 80), nn.ReLU(),
            nn.Linear(80, 40), nn.ReLU(),
            nn.Linear(40, 4)
        )
    def forward(self, x): return self.net(x)


class Model4(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48), nn.ReLU(),
            nn.Linear(48, 24), nn.ReLU(),
            nn.Linear(24, 4)
        )
    def forward(self, x): return self.net(x)


class Model5(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 70), nn.ReLU(),
            nn.Linear(70, 35), nn.ReLU(),
            nn.Linear(35, 4)
        )
    def forward(self, x): return self.net(x)


P1_MODELS = {
    1: Model1,
    2: Model2Strong,
    3: Model3,
    4: Model4,
    5: Model5
}


# RUN INFERENCE
def run_inference(xfile, weights_file, output_file):
    df = pd.read_csv(xfile)
    hh_ids = df["hh_id"]
    X = df.drop(columns=["hh_id"]).values.astype(np.float32)
    input_dim = X.shape[1]

    tensor_x = torch.tensor(X, dtype=torch.float32)

    # load all 5 models (same as training)
    state = torch.load(weights_file, map_location="cpu")

    preds = {}

    for out_id in range(1, 6):
        model = P1_MODELS[out_id](input_dim)
        model.load_state_dict(state[out_id])
        model.eval()
        with torch.no_grad():
            logits = model(tensor_x)
            pred = logits.argmax(dim=1).numpy()
        preds[f"pred_{out_id}"] = pred

    # save
    out_df = pd.DataFrame({"hh_id": hh_ids})
    for k, v in preds.items():
        out_df[k] = v

    out_df.to_csv(output_file, index=False)
    print(f"Saved Phase 1 inference: {output_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["rural","urban"], required=True)
    p.add_argument("--xfile", required=True)
    p.add_argument("--weights", required=True)   # w_p1/rural.pt or w_p1/urban.pt
    p.add_argument("--output", required=True)
    args = p.parse_args()

    run_inference(args.xfile, args.weights, args.output)


if __name__ == "__main__":
    main()

