"""Train an MLP on XOR until loss converges near zero."""
from __future__ import annotations

import numpy as np

from nanograd import Tensor, nn, optim
from nanograd.nn import functional as F


def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    model = nn.Sequential(
        nn.Linear(2, 8, seed=0),
        nn.Tanh(),
        nn.Linear(8, 1, seed=1),
    )
    opt = optim.Adam(model.parameters(), lr=0.1)

    for step in range(1, 2001):
        pred = model(Tensor(X))
        loss = F.bce_with_logits_loss(pred, Tensor(y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            probs = F.sigmoid(model(Tensor(X))).data.reshape(-1)
            print(f"step {step:4d}  loss={loss.item():.6f}  probs={probs}")

    final_loss = loss.item()
    probs = F.sigmoid(model(Tensor(X))).data.reshape(-1)
    correct = np.all((probs > 0.5) == y.reshape(-1).astype(bool))
    print(f"\nfinal loss = {final_loss:.6f}  correct = {correct}")
    assert final_loss < 0.01, f"XOR did not converge: loss={final_loss}"
    assert correct, f"XOR predictions wrong: {probs}"


if __name__ == "__main__":
    main()
