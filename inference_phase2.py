import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn


class Phase2Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.start = nn.Sequential(nn.Linear(input_dim, 128), nn.GELU())
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(128,128), nn.GELU(), nn.Linear(128,128)
        ) for _ in range(3)])
        self.out = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64,1), nn.Sigmoid())

    def forward(self, x):
        x = self.start(x)
        for blk in self.blocks:
            x = x + blk(x)
        return self.out(x)


def run_inference(xfile, latentfile, weights, output):
    X = pd.read_csv(xfile)
    Z = pd.read_csv(latentfile)

    df = pd.concat([X, Z[["z1","z2","z3"]]], axis=1)
    data = df.drop(columns=["hh_id"]).values.astype(np.float32)
    hh_ids = df["hh_id"]

    model = Phase2Net(input_dim=data.shape[1])
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(data)).numpy().flatten()

    out = pd.DataFrame({"hh_id": hh_ids, "prob": pred})
    out.to_csv(output, index=False)
    print(f"Saved Phase 2 inference: {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xfile", required=True)
    p.add_argument("--latent", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    run_inference(args.xfile, args.latent, args.weights, args.output)


if __name__ == "__main__":
    main()

