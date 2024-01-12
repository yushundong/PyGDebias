from pygdebias.debiasing import FairGNN
from pygdebias.datasets import Bail

import numpy as np
from collections import defaultdict
import torch
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


setup_seed(10)

bail = Bail()
adj, feats, idx_train, idx_val, idx_test, labels, sens = (
    bail.adj(),
    bail.features(),
    bail.idx_train(),
    bail.idx_val(),
    bail.idx_test(),
    bail.labels(),
    bail.sens(),
)


params = {
    "sim_coeff": 0.4,
    "n_order": 15,
    "subgraph_size": 20,
    "acc": 0.84,
    "epoch": 500,
}

# Initiate the model (with searched parameters).
model = FairGNN(
    feats.shape[-1],
    sim_coeff=params["sim_coeff"],
    acc=params["acc"],
    n_order=params["n_order"],
    subgraph_size=params["subgraph_size"],
    epoch=params["epoch"],
).cuda()
model.fit(adj, feats, labels, idx_train, idx_val, idx_test, sens, idx_train)


# Evaluate the model.

(
    ACC,
    AUCROC,
    F1,
    ACC_sens0,
    AUCROC_sens0,
    F1_sens0,
    ACC_sens1,
    AUCROC_sens1,
    F1_sens1,
    SP,
    EO,
) = model.predict(idx_test)

print("ACC:", ACC)
print("AUCROC: ", AUCROC)
print("F1: ", F1)
print("ACC_sens0:", ACC_sens0)
print("AUCROC_sens0: ", AUCROC_sens0)
print("F1_sens0: ", F1_sens0)
print("ACC_sens1: ", ACC_sens1)
print("AUCROC_sens1: ", AUCROC_sens1)
print("F1_sens1: ", F1_sens1)
print("SP: ", SP)
print("EO:", EO)
