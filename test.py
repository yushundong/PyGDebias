from pygdebias.debiasing import FairGNN
from pygdebias.datasets import Bail

# Available choices: 'Credit', 'German', 'Facebook', 'Pokec_z', 'Pokec_n', 'Nba', 'Twitter', 'Google', 'LCC', 'LCC_small', 'Cora', 'Citeseer', 'Amazon', 'Yelp', 'Epinion', 'Ciao', 'Dblp', 'Filmtrust', 'Lastfm', 'Ml-100k', 'Ml-1m', 'Ml-20m', 'Oklahoma', 'UNC', 'Bail'.

bail = Bail()
adj, features, idx_train, idx_val, idx_test, labels, sens = (
    bail.adj(),
    bail.features(),
    bail.idx_train(),
    bail.idx_val(),
    bail.idx_test(),
    bail.labels(),
    bail.sens(),
)

# Initiate the model (with default parameters).
model = FairGNN(features.shape[1])

# Train the model.
model.fit(adj, features, labels, idx_train, idx_val, idx_test, sens, idx_train)

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
