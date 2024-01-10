import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

        out = (q @ k.T)/np.sqrt(k.size()[1]) # is this correct?
        out = F.softmax(out, dim=-1) # √ 
        out = out @ v # not so sure about the final interpretation of this

        return out

if __name__ == "__main__":
    ids = torch.randint(0, 10, (3,))

    model = Transformer()
    y_hat = model(ids)

    print("done")
