from pygdebias.debiasing import (
    CrossWalk,
    EDITS,
    FairEdit,
    FairGNN,
    FairVGNN,
    FairWalk,
    GEAR,
    GNN,
    GUIDE,
    InFoRM_GNN,
    NIFTY,
    RawlsGCN,
    REDRESS,
    UGE,
)
from pygdebias.datasets import Bail
import torch

# Available choices: 'Credit', 'German', 'Facebook', 'Pokec_z', 'Pokec_n', 'Nba', 'Twitter', 'Google', 'LCC', 'LCC_small', 'Cora', 'Citeseer', 'Amazon', 'Yelp', 'Epinion', 'Ciao', 'Dblp', 'Filmtrust', 'Lastfm', 'Ml-100k', 'Ml-1m', 'Ml-20m', 'Oklahoma', 'UNC', 'Bail'.
# Here we use Bail dataset for illustration.

bail = Bail()
adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx = (
    bail.adj(),
    bail.features(),
    bail.idx_train(),
    bail.idx_val(),
    bail.idx_test(),
    bail.labels(),
    bail.sens(),
    bail.sens_idx(),
)

print("FairGNN")
model = FairGNN(
    nfeat=features.shape[1],
    sim_coeff=0.6,
    n_order=10,
    subgraph_size=30,
    acc=0.69,
    epoch=2000,
)
model.fit(
    g=adj,
    features=features,
    labels=labels,
    idx_train=idx_train,
    idx_val=idx_val,
    idx_test=idx_test,
    sens=sens,
    idx_sens_train=idx_train,
    device="cuda",
)

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


print("CrossWalk")
model = CrossWalk()
model.fit(
    adj_matrix=adj,
    feats=features,
    labels=labels,
    idx_train=idx_train,
    sens=sens,
    number_walks=5,
    representation_size=64,
    seed=0,
    walk_length=20,
    window_size=5,
    workers=5,
    pmodified=1.0,
)

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
) = model.predict(idx_test=idx_test, idx_val=idx_val)

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


print("EDITS")
model = EDITS(
    feat=features,
    lr=0.003,
    weight_decay=1e-7,
    nclass=2,
    adj_lambda=1e-1,
    layer_threshold=2,
    dropout=0.1,
)
model.fit(
    adj=adj,
    features=features,
    idx_train=idx_train,
    idx_val=idx_val,
    sens=sens,
    epochs=100,
    normalize=True,
    lr=0.003,
    k=-1,
    device="cuda",
    half=True,
    truncation=4,
)

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
) = model.predict(
    adj_ori=adj,
    labels=labels,
    sens=sens,
    idx_train=idx_train,
    idx_val=idx_val,
    idx_test=idx_test,
    epochs=100,
    lr=0.003,
    nhid=50,
    dropout=0.2,
    weight_decay=1e-7,
    model="GCN",
    device="cuda",
    threshold_proportion=0.015,
)

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


print("FairEdit")
model = FairEdit()
model.fit(
    adj=adj,
    features=features,
    labels=labels,
    idx_train=idx_train,
    idx_val=idx_val,
    idx_test=idx_test,
    sens=sens,
    sens_idx=sens_idx,
    model_name="gcn",
    epochs=100,
    lr=1e-3,
    weight_decay=5e-4,
    hidden=16,
    dropout=0.5,
    edit_num=10,
)

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
) = model.predict()

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


print("FairVGNN")
model = FairVGNN()
model.fit(
    adj=adj,
    feats=features,
    labels=labels,
    idx_train=idx_train,
    idx_val=idx_val,
    idx_test=idx_test,
    sens=sens,
    sens_idx=sens_idx,
    runs=1,
    epochs=200,
    d_epochs=5,
    g_epochs=5,
    c_epochs=5,
    g_lr=0.001,
    g_wd=0,
    d_lr=0.001,
    d_wd=0,
    c_lr=0.001,
    c_wd=0,
    e_lr=0.001,
    e_wd=0,
    early_stopping=0,
    prop="spmm",
    dropout=0.5,
    hidden=16,
    seed=1,
    encoder="GCN",
    K=10,
    top_k=10,
    clip_e=1,
    f_mask="yes",
    weight_clip="yes",
    ratio=1,
    alpha=1,
)

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
) = model.predict()

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


print("FairWalk")
model = FairWalk()
model.fit(
    adj=adj,
    labels=labels,
    idx_train=idx_train,
    sens=sens,
    dimensions=64,
    walk_length=20,
    num_walks=5,
    p=1,
    q=1,
    weight_key="weight",
    workers=1,
    sampling_strategy=None,
    quiet=False,
    temp_folder=None,
)

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
) = model.predict(idx_test=idx_test, idx_val=idx_val)

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


print("GEAR")
model = GEAR(
    adj=adj,
    features=features,
    labels=labels,
    idx_train=idx_train,
    idx_val=idx_val,
    idx_test=idx_test,
    sens=sens,
    sens_idx=sens_idx,
    hidden_size=1024,
    proj_hidden=16,
    num_class=1,
    encoder_hidden_size=1024,
    encoder_base_model="gcn",
    experiment_type="train",
)
model.fit(
    epochs=500,
    lr=0.001,
    batch_size=100,
    weight_decay=1e-5,
    sim_coeff=0.6,
    encoder_name="None",
    dataset_name="None",
    device="cuda",
)
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
) = model.predict(
    encoder_name="None", dataset_name="None", batch_size=100, sim_coeff=0.6
)

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


print("GNN")
model = GNN(
    adj=adj,
    features=features,
    labels=labels,
    idx_train=idx_train,
    idx_val=idx_val,
    idx_test=idx_test,
    sens=sens,
    sens_idx=sens_idx,
    num_hidden=16,
    num_proj_hidden=16,
    lr=0.001,
    weight_decay=1e-5,
    drop_edge_rate_1=0.1,
    drop_edge_rate_2=0.1,
    drop_feature_rate_1=0.1,
    drop_feature_rate_2=0.1,
    encoder="gcn",
    sim_coeff=0.5,
    nclass=1,
    device="cuda",
)
model.fit(epochs=300)
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
) = model.predict()

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


print("GUIDE")
model = GUIDE(
    num_layers=1,
    nfeat=16,
    nhid=16,
    nclass=1,
    heads=1,
    negative_slope=0.2,
    concat=False,
    dropout=0,
    path="./",
)
model.fit(
    adj=adj,
    features=features,
    idx_train=idx_train,
    idx_val=idx_val,
    idx_test=idx_test,
    labels=labels,
    sens=sens,
    hidden_num=16,
    dropout=0,
    lr=0.001,
    weight_decay=1e-5,
    initialize_training_epochs=1000,
    epochs=1000,
    alpha=5e-6,
    beta=1,
    gnn_name="gcn",
    device="cuda",
)

(F1, ACC, AUCROC, individual_unfairness, GDIF) = model.predict()

print("ACC:", ACC)
print("AUCROC: ", AUCROC)
print("F1: ", F1)
print("individual_unfairness: ", individual_unfairness)
print("GDIF: ", GDIF)


print("InFoRM_GNN")
model = InFoRM_GNN(
    adj=adj,
    features=features,
    idx_train=idx_train,
    idx_val=idx_val,
    idx_test=idx_test,
    labels=labels,
    sens=sens,
    gnn_name="gcn",
    lr=0.001,
    hidden=16,
    dropout=0,
    weight_decay=1e-5,
    device="cuda",
    path="./",
)
model.fit(epochs=3000, alpha=5e-6, opt_if=1)

(F1, ACC, AUCROC, individual_unfairness, GDIF) = model.predict()

print("ACC:", ACC)
print("AUCROC: ", AUCROC)
print("F1: ", F1)
print("individual_unfairness: ", individual_unfairness)
print("GDIF: ", GDIF)


print("NIFTY")
model = NIFTY(
    adj=adj,
    features=features,
    labels=labels,
    idx_train=idx_train,
    idx_val=idx_val,
    idx_test=idx_test,
    sens=sens,
    sens_idx=sens_idx,
    num_hidden=16,
    num_proj_hidden=16,
    lr=0.001,
    weight_decay=1e-5,
    drop_edge_rate_1=0.1,
    drop_edge_rate_2=0.1,
    drop_feature_rate_1=0.1,
    drop_feature_rate_2=0.1,
    encoder="gcn",
    sim_coeff=0.5,
    nclass=1,
    device="cuda",
)
model.fit(epochs=300)

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


print("RawlsGCN")
nclass = torch.unique(labels).shape
model = RawlsGCN()
model.fit(
    adj=adj,
    feats=features,
    labels=labels,
    idx_train=idx_train,
    idx_val=idx_val,
    idx_test=idx_test,
    enable_cude=True,
    device_number=0,
    model="rawlsgcn_graph",
    seed=0,
    num_epoch=100,
    lr=0.05,
    weight_decay=5e-4,
    hidden=64,
    dropout=0.5,
    loss="negative_log_likelihood",
)

(ACC, bias) = model.predict()

print("ACC:", ACC)
print("bias: ", bias)


print("REDRESS")
model = REDRESS(
    adj=adj,
    features=features,
    labels=labels,
    sens=sens,
    idx_train=idx_train,
    idx_val=idx_val,
    idx_test=idx_test,
    lr=0.003,
    hidden=16,
    dropout=0.6,
    weight_decay=5e-4,
    degree=2,
    model_name="GCN",
    top_k=10,
    sigma_1=2e-2,
    cuda=1,
    pre_train=1500,
    epochs=20,
    path="./",
)
model.fit(model_name="GCN")

(F1, ACC, AUCROC, individual_unfairness, GDIF) = model.predict()

print("ACC:", ACC)
print("AUCROC: ", AUCROC)
print("F1: ", F1)
print("individual_unfairness: ", individual_unfairness)
print("GDIF: ", GDIF)


print("UGE")
model = UGE()
model.fit(
    adj=adj,
    feats=features,
    labels=labels,
    idx_train=idx_train,
    sens=sens,
    model="gcn",
    debias_method="uge-w",
    debias_attr="gender",
    reg_weight=0.5,
    loss="entropy",
    lr=5e-3,
    weight_decay=5e-4,
    dim1=64,
    dim2=32,
    predictor="dot",
    seed=0,
    device=0,
    epochs=50,
)

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
) = model.predict(idx_test=idx_test, idx_val=idx_val)

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
