import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def scaled_dot_product(key, value, query) -> torch.Tensor:
    """
    Computes the scaled dot product attention.
    """
    out = (query @ key.T)/np.sqrt(key.size()[1])
    out = F.softmax(out, dim=-1)
    out = out @ value
    return out


class Transformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.emb = nn.Embedding(10, 3)
        self.q = nn.Linear(3, 4)
        self.k = nn.Linear(3, 4)
        self.v = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor):
        emb = self.emb(x)
        q = self.q(emb)
        k = self.k(emb)
        v = self.v(emb)

        out = scaled_dot_product(key=k, value=v, query=q)

        return out

if __name__ == "__main__":
    ids = torch.randint(0, 10, (3,))

    model = Transformer()
    y_hat = model(ids)

    print("done")
