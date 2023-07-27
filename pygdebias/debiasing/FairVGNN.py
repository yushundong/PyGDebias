import argparse
from tqdm import tqdm
from torch import tensor
import warnings
warnings.filterwarnings('ignore')
import math
#from torch_geometric.utils import from_scipy_sparse_matrix
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.data import Data
from torch.nn import Linear
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import GINConv, SAGEConv
from torch.nn.utils import spectral_norm
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter
import random
import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
import torch
from torch_geometric.utils import from_scipy_sparse_matrix

def train(model, data, optimizer, args):
    model.train()
    optimizer.zero_grad()

    output, h = model(data.x, data.edge_index)
    preds = (output.squeeze() > 0).type_as(data.y)

    loss = {}
    loss['train'] = F.binary_cross_entropy_with_logits(
        output[data.train_mask], data.y[data.train_mask].unsqueeze(1).float().to(args.device))
    loss['val'] = F.binary_cross_entropy_with_logits(
        output[data.val_mask], data.y[data.val_mask].unsqueeze(1).float().to(args.device), weight=args.val_ratio)

    loss['train'].backward()
    optimizer.step()

    return loss


def evaluate(model, data, args):
    model.eval()

    with torch.no_grad():
        x_flip, edge_index1, edge_index2, mask1, mask2 = random_aug(
            data.x, data.edge_index, args)
        output, h = model(x_flip, data.edge_index, mask=torch.ones_like(
            data.edge_index[0, :], dtype=torch.bool))

        loss_ce = F.binary_cross_entropy_with_logits(
            output[data.val_mask], data.y[data.val_mask].unsqueeze(1), weight=args.val_ratio)

        # loss_cl = InfoNCE(h[data.train_mask], h[data.train_mask],
        #                   args.label_mask_pos, args.label_mask_neg, tau=0.5)

        loss_val = loss_ce

    accs, auc_rocs, F1s = {}, {}, {}

    pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)

    accs['val'] = pred_val.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum()
    accs['test'] = pred_test.eq(
        data.y[data.test_mask]).sum().item() / data.test_mask.sum()

    F1s['val'] = f1_score(data.y[data.val_mask].cpu(
    ).numpy(), pred_val.cpu().numpy())

    F1s['test'] = f1_score(data.y[data.test_mask].cpu(
    ).numpy(), pred_test.cpu().numpy())

    auc_rocs['val'] = roc_auc_score(
        data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(
        data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    parity, equality = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
    ).numpy(), data.sens[data.test_mask].cpu().numpy())

    return accs, auc_rocs, F1s, parity, equality, loss_val


def evaluate_finetune(encoder, classifier, data, args):
    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        h = encoder(data.x, data.edge_index)
        output = classifier(h)

    accs, auc_rocs, F1s = {}, {}, {}

    loss_val = F.binary_cross_entropy_with_logits(
        output[data.val_mask], data.y[data.val_mask].unsqueeze(1).float().to(args.device))

    pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)

    accs['val'] = pred_val.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum()
    accs['test'] = pred_test.eq(
        data.y[data.test_mask]).sum().item() / data.test_mask.sum()

    F1s['val'] = f1_score(data.y[data.val_mask].cpu(
    ).numpy(), pred_val.cpu().numpy())

    F1s['test'] = f1_score(data.y[data.test_mask].cpu(
    ).numpy(), pred_test.cpu().numpy())

    auc_rocs['val'] = roc_auc_score(
        data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(
        data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    parity, equality = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
    ).numpy(), data.sens[data.test_mask].cpu().numpy())

    return accs, auc_rocs, F1s, parity, equality, loss_val


def evaluate_exploration(x, model, data, args):
    model.eval()

    with torch.no_grad():
        outputs, loss_ce = [], 0
        for k in range(args.K):
            x = data.x.clone()
            # print(data.x.unique())
            x[:, args.corr_idx] = (torch.rand(
                len(args.corr_idx)) * (args.x_max[args.corr_idx] - args.x_min[args.corr_idx]) + args.x_min[args.corr_idx]).to(args.device)

            output, h2 = model(x, data.edge_index)
            outputs.append(output)

            loss_ce += F.binary_cross_entropy_with_logits(
                output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

        loss_val = loss_ce / args.K

        # output1, h1 = model(data.x, data.edge_index)
        # output2, h2 = model(x, data.edge_index)

        # loss_ce = F.binary_cross_entropy_with_logits(
        #     output2[data.val_mask], data.y[data.val_mask].unsqueeze(1))

        # loss_val = loss_ce

    output = torch.stack(outputs).mean(dim=0)

    accs, auc_rocs, F1s, paritys, equalitys = {}, {}, {}, {}, {}

    pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)

    accs['val'] = pred_val.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum()
    accs['test'] = pred_test.eq(
        data.y[data.test_mask]).sum().item() / data.test_mask.sum()

    F1s['val'] = f1_score(data.y[data.val_mask].cpu(
    ).numpy(), pred_val.cpu().numpy())

    F1s['test'] = f1_score(data.y[data.test_mask].cpu(
    ).numpy(), pred_test.cpu().numpy())

    auc_rocs['val'] = roc_auc_score(
        data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(
        data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    paritys['val'], equalitys['val'] = fair_metric(pred_val.cpu().numpy(), data.y[data.val_mask].cpu(
    ).numpy(), data.sens[data.val_mask].cpu().numpy())

    paritys['test'], equalitys['test'] = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
    ).numpy(), data.sens[data.test_mask].cpu().numpy())

    return accs, auc_rocs, F1s, paritys, equalitys, loss_val


def evaluate_ged(x, classifier, discriminator, generator, encoder, data, args):
    classifier.eval()
    generator.eval()
    discriminator.eval()
    encoder.eval()

    with torch.no_grad():
        if(args.f_mask == 'yes'):
            outputs, loss_e = [], 0
            feature_weights = generator()
            for k in range(args.K):
                x = data.x * F.gumbel_softmax(
                    feature_weights, tau=1, hard=False)[:, 0]

                h = encoder(x, data.edge_index)
                output = classifier(h)
                output2 = discriminator(h)

                if(args.adv == 'yes'):
                    loss_e += F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + args.sup_alpha * F.binary_cross_entropy_with_logits(
                        output[data.val_mask], data.y[data.val_mask].unsqueeze(1))
                else:
                    loss_e += F.binary_cross_entropy_with_logits(
                        output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

                outputs.append(output)

            loss_val = loss_e / args.K

            output = torch.stack(outputs).mean(dim=0)
        else:
            h = encoder(data.x, data.edge_index)
            output, h = classifier(h)

            if(args.adv == 'yes'):
                loss_val = F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + args.sup_alpha * F.binary_cross_entropy_with_logits(
                    output[data.val_mask], data.y[data.val_mask].unsqueeze(1))
            else:
                loss_val = F.binary_cross_entropy_with_logits(
                    output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

    accs, auc_rocs, F1s, paritys, equalitys = {}, {}, {}, {}, {}

    pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)

    accs['val'] = pred_val.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    accs['test'] = pred_test.eq(
        data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

    F1s['val'] = f1_score(data.y[data.val_mask].cpu(
    ).numpy(), pred_val.cpu().numpy())

    F1s['test'] = f1_score(data.y[data.test_mask].cpu(
    ).numpy(), pred_test.cpu().numpy())

    auc_rocs['val'] = roc_auc_score(
        data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(
        data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    paritys['val'], equalitys['val'] = fair_metric(pred_val.cpu().numpy(), data.y[data.val_mask].cpu(
    ).numpy(), data.sens[data.val_mask].cpu().numpy())

    paritys['test'], equalitys['test'] = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
    ).numpy(), data.sens[data.test_mask].cpu().numpy())

    return accs, auc_rocs, F1s, paritys, equalitys, loss_val


def evaluate_ged2(x, classifier, discriminator, generator, encoder, data, args):
    classifier.eval()
    generator.eval()
    encoder.eval()

    with torch.no_grad():
        if(args.f_mask == 'yes'):
            outputs, loss = [], 0
            feature_weights = generator()
            for k in range(args.K):
                x = data.x * F.gumbel_softmax(
                    feature_weights, tau=1, hard=False)[:, 0]

                h = encoder(x, data.edge_index)
                output = classifier(h)
                output2 = discriminator(h)

                # loss += F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + F.binary_cross_entropy_with_logits(
                #     output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

                outputs.append(output)

            loss_val = loss / args.K

            output = torch.stack(outputs).mean(dim=0)
        else:
            h = encoder(data.x, data.edge_index)
            output = classifier(h)
            output2 = discriminator(h)

            # loss_val = F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + F.binary_cross_entropy_with_logits(
            #     output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

    accs, auc_rocs, F1s, paritys, equalitys = {}, {}, {}, {}, {}

    pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)

    accs['val'] = pred_val.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    accs['test'] = pred_test.eq(
        data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

    F1s['val'] = f1_score(data.y[data.val_mask].cpu(
    ).numpy(), pred_val.cpu().numpy())

    F1s['test'] = f1_score(data.y[data.test_mask].cpu(
    ).numpy(), pred_test.cpu().numpy())

    auc_rocs['val'] = roc_auc_score(
        data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(
        data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    paritys['val'], equalitys['val'] = fair_metric(pred_val.cpu().numpy(), data.y[data.val_mask].cpu(
    ).numpy(), data.sens[data.val_mask].cpu().numpy())

    paritys['test'], equalitys['test'] = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
    ).numpy(), data.sens[data.test_mask].cpu().numpy())

    return accs, auc_rocs, F1s, paritys, equalitys


def evaluate_ged3(x, classifier, discriminator, generator, encoder, data, args):
    classifier.eval()
    generator.eval()
    encoder.eval()

    with torch.no_grad():
        if(args.f_mask == 'yes'):
            outputs, loss = [], 0
            feature_weights = generator()
            for k in range(args.K):
                x = data.x * F.gumbel_softmax(
                    feature_weights, tau=1, hard=True)[:, 0]

                h = encoder(x, data.edge_index, data.adj_norm_sp)
                output = classifier(h)
                # output2 = discriminator(h)

                # loss += F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + F.binary_cross_entropy_with_logits(
                #     output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

                outputs.append(output)

            # loss_val = loss / args.K

            output = torch.stack(outputs).mean(dim=0)
        else:
            h = encoder(data.x, data.edge_index, data.adj_norm_sp)
            output = classifier(h)
            # output2 = discriminator(h)

            # loss_val = F.mse_loss(output.view(-1), 0.5 * torch.ones_like(output.view(-1))) + F.binary_cross_entropy_with_logits(
            #     output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

    accs, auc_rocs, F1s, paritys, equalitys = {}, {}, {}, {}, {}

    pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)

    accs['val'] = pred_val.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    accs['test'] = pred_test.eq(
        data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()


    accs['test_sens0']=pred_test[data.sens[data.test_mask]==0].eq(
        data.y[data.test_mask][data.sens[data.test_mask]==0]).float().mean().item()

    accs['test_sens1']=pred_test[data.sens[data.test_mask]==1].eq(
        data.y[data.test_mask][data.sens[data.test_mask]==1]).float().mean().item()

    F1s['val'] = f1_score(data.y[data.val_mask].cpu(
    ).numpy(), pred_val.cpu().numpy())


    loss_fn=torch.nn.BCELoss()
    F1s['val_loss']=loss_fn(output[data.val_mask].squeeze().sigmoid(), torch.tensor(data.y[data.val_mask]).float()).item()

    F1s['test'] = f1_score(data.y[data.test_mask].cpu(
    ).numpy(), pred_test.cpu().numpy())

    F1s['test_sens0']= f1_score(data.y[data.test_mask][data.sens[data.test_mask]==0].cpu(
    ).numpy(), pred_test[data.sens[data.test_mask]==0].cpu().numpy())
    F1s['test_sens1']= f1_score(data.y[data.test_mask][data.sens[data.test_mask]==1].cpu(
    ).numpy(), pred_test[data.sens[data.test_mask]==1].cpu().numpy())

    auc_rocs['val'] = roc_auc_score(
        data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(
        data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    auc_rocs['test_sens0'] = roc_auc_score(
        data.y[data.test_mask][data.sens[data.test_mask]==0].cpu().numpy(), output[data.test_mask][data.sens[data.test_mask]==0].detach().cpu().numpy())
    auc_rocs['test_sens1'] = roc_auc_score(
        data.y[data.test_mask][data.sens[data.test_mask]==1].cpu().numpy(), output[data.test_mask][data.sens[data.test_mask]==1].detach().cpu().numpy())


    paritys['val'], equalitys['val'] = fair_metric(pred_val.cpu().numpy(), data.y[data.val_mask].cpu(
        ).numpy(), data.sens[data.val_mask].cpu().numpy())

    paritys['test'], equalitys['test'] = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
    ).numpy(), data.sens[data.test_mask].cpu().numpy())

    return accs, auc_rocs, F1s, paritys, equalitys

def propagate(x, edge_index, edge_weight=None):
    """ feature propagation procedure: sparsematrix
    """
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    if(edge_weight == None):
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def propagate_mask(x, edge_index, mask_node=None):
    """ feature propagation procedure: sparsematrix
    """
    edge_index, _ = add_remaining_self_loops(
        edge_index, num_nodes=x.size(0))

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    if(mask_node == None):
        mask_node = torch.ones_like(x[:, 0])

    mask_node = mask_node[row]
    mask_node[row == col] = 1

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row] * \
          mask_node.view(-1, 1)

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def propagate2(x, edge_index):
    """ feature propagation procedure: sparsematrix
    """
    edge_index, _ = add_remaining_self_loops(
        edge_index, num_nodes=x.size(0))

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.allow_tf32 = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    # torch.use_deterministic_algorithms(True)


def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) -
                 sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) -
                   sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
    return parity.item(), equality.item()


def visual(model, data, sens, dataname):
    model.eval()

    print(data.y, sens)
    hidden = model.encoder(data.x, data.edge_index).cpu().detach().numpy()
    sens, data.y = sens.cpu().numpy(), data.y.cpu().numpy()
    idx_s0, idx_s1, idx_s2, idx_s3 = (sens == 0) & (data.y == 0), (sens == 0) & (
            data.y == 1), (sens == 1) & (data.y == 0), (sens == 1) & (data.y == 1)

    tsne_hidden = TSNE(n_components=2)
    tsne_hidden_x = tsne_hidden.fit_transform(hidden)

    tsne_input = TSNE(n_components=2)
    tsne_input_x = tsne_input.fit_transform(data.x.detach().cpu().numpy())

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    items = [tsne_input_x, tsne_hidden_x]
    names = ['input', 'hidden']

    for ax, item, name in zip(axs, items, names):
        ax.scatter(item[idx_s0][:, 0], item[idx_s0][:, 1], s=1,
                   c='red', marker='o', label='class 1, group1')
        ax.scatter(item[idx_s1][:, 0], item[idx_s1][:, 1], s=1,
                   c='blue', marker='o', label='class 2, group1')
        ax.scatter(item[idx_s2][:, 0], item[idx_s2][:, 1], s=10,
                   c='red', marker='', label='class 1, group2')
        ax.scatter(item[idx_s3][:, 0], item[idx_s3][:, 1], s=10,
                   c='blue', marker='+', label='class 2, group2')

        ax.set_title(name)
    ax.legend(frameon=0, loc='upper center',
              ncol=4, bbox_to_anchor=(-0.2, 1.2))

    plt.savefig(dataname + 'visual_tsne.pdf',
                dpi=1000, bbox_inches='tight')


def visual_sub(model, data, sens, dataname, k=50):
    idx_c1, idx_c2 = torch.where((sens == 0) == True)[
                         0], torch.where((sens == 1) == True)[0]

    idx_subc1, idx_subc2 = idx_c1[torch.randperm(
        idx_c1.shape[0])[:k]], idx_c2[torch.randperm(idx_c2.shape[0])[:k]]

    idx_sub = torch.cat([idx_subc1, idx_subc2]).cpu().numpy()
    sens = sens[idx_sub]
    y = data.y[idx_sub]

    model.eval()

    hidden = model.encoder(data.x, data.edge_index).cpu().detach().numpy()
    sens, y = sens.cpu().numpy(), y.cpu().numpy()
    idx_s0, idx_s1, idx_s2, idx_s3 = (sens == 0) & (y == 0), (sens == 0) & (
            y == 1), (sens == 1) & (y == 0), (sens == 1) & (y == 1)

    tsne_hidden = TSNE(n_components=2)
    tsne_hidden_x = tsne_hidden.fit_transform(hidden)

    tsne_input = TSNE(n_components=2)
    tsne_input_x = tsne_input.fit_transform(data.x.detach().cpu().numpy())

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    items = [tsne_input_x[idx_sub], tsne_hidden_x[idx_sub]]
    names = ['input', 'hidden']

    for ax, item, name in zip(axs, items, names):
        ax.scatter(item[idx_s0][:, 0], item[idx_s0][:, 1], s=1,
                   c='red', marker='.', label='group1 class1')
        ax.scatter(item[idx_s1][:, 0], item[idx_s1][:, 1], s=5,
                   c='red', marker='*', label='group1 class2')
        ax.scatter(item[idx_s2][:, 0], item[idx_s2][:, 1], s=1,
                   c='blue', marker='.', label='group2 class1')
        ax.scatter(item[idx_s3][:, 0], item[idx_s3][:, 1], s=5,
                   c='blue', marker='*', label='group2 class2')

        ax.set_title(name)
    ax.legend(frameon=0, loc='upper center',
              ncol=4, bbox_to_anchor=(-0.2, 1.2))

    plt.savefig(dataname + 'visual_tsne.pdf',
                dpi=1000, bbox_inches='tight')


def pos_neg_mask(label, nodenum, train_mask):
    pos_mask = torch.stack([(label == label[i]).float()
                            for i in range(nodenum)])
    neg_mask = 1 - pos_mask

    return pos_mask[train_mask, :][:, train_mask], neg_mask[train_mask, :][:, train_mask]


def pos_neg_mask_sens(sens_label, label, nodenum, train_mask):
    pos_mask = torch.stack([((label == label[i]) & (sens_label == sens_label[i])).float()
                            for i in range(nodenum)])
    neg_mask = torch.stack([((label == label[i]) & (sens_label != sens_label[i])).float()
                            for i in range(nodenum)])

    return pos_mask[train_mask, :][:, train_mask], neg_mask[train_mask, :][:, train_mask]


def similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


def InfoNCE(h1, h2, pos_mask, neg_mask, tau=0.2):
    num_nodes = h1.shape[0]

    sim = similarity(h1, h2) / tau
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)

    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)

    return loss.mean()


def random_aug(x, edge_index, args):
    x_flip = flip_sens_feature(x, args.sens_idx, args.flip_node_ratio)

    edge_index1 = random_mask_edge(edge_index, args)
    edge_index2 = random_mask_edge(edge_index, args)

    mask1 = random_mask_node(x, args)
    mask2 = random_mask_node(x, args)

    return x_flip, edge_index1, edge_index2, mask1, mask2


def random_aug2(x, edge_index, args):
    # x_flip = flip_sens_feature(x, args.sens_idx, args.flip_node_ratio)
    edge_index = random_mask_edge(edge_index, args)

    mask = random_mask_node(x, args)

    return edge_index, mask


def flip_sens_feature(x, sens_idx, flip_node_ratio):
    node_num = x.shape[0]
    idx = np.arange(0, node_num)
    samp_idx = np.random.choice(idx, size=int(
        node_num * flip_node_ratio), replace=False)

    x_flip = x.clone()
    x_flip[:, sens_idx] = 1 - x_flip[:, sens_idx]

    return x_flip


def random_mask_edge(edge_index, args):
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        node_num = edge_index.size(0)
        edge_index = torch.stack([row, col], dim=0)

        edge_num = edge_index.shape[1]
        idx = np.arange(0, edge_num)
        samp_idx = np.random.choice(idx, size=int(
            edge_num * args.mask_edge_ratio), replace=False)

        mask = torch.ones(edge_num, dtype=torch.bool)
        mask[samp_idx] = 0

        edge_index = edge_index[:, mask]

        edge_index = SparseTensor(
            row=edge_index[0], col=edge_index[1],
            value=None, sparse_sizes=(node_num, node_num),
            is_sorted=True)

    else:
        edge_index, _ = add_remaining_self_loops(
            edge_index)
        edge_num = edge_index.shape[1]
        idx = np.arange(0, edge_num)
        samp_idx = np.random.choice(idx, size=int(
            edge_num * args.mask_edge_ratio), replace=False)

        mask = torch.ones_like(edge_index[0, :], dtype=torch.bool)
        mask[samp_idx] = 0

        edge_index = edge_index[:, mask]

    return edge_index


def random_mask_node(x, args):
    node_num = x.shape[0]
    idx = np.arange(0, node_num)
    samp_idx = np.random.choice(idx, size=int(
        node_num * args.mask_node_ratio), replace=False)

    mask = torch.ones_like(x[:, 0])
    mask[samp_idx] = 0

    return mask


def consis_loss(ps, temp=0.5):
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p

    avg_p = sum_p / len(ps)

    sharp_p = (torch.pow(avg_p, 1. / temp) /
               torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()

    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1))
    loss = loss / len(ps)
    return 1 * loss



def visualize(embeddings, y, s):
    X_embed = TSNE(n_components=2, learning_rate='auto',
                   init='random').fit_transform(embeddings)

    group1 = (y == 0) & (s == 0)
    group2 = (y == 0) & (s == 1)
    group3 = (y == 1) & (s == 0)
    group4 = (y == 1) & (s == 1)

    plt.scatter(X_embed[group1, 0], X_embed[group1, 1],
                s=5, c='tab:blue', marker='o')
    plt.scatter(X_embed[group2, 0], X_embed[group2, 1],
                s=5, c='tab:orange', marker='s')
    plt.scatter(X_embed[group3, 0], X_embed[group3, 1],
                s=5, c='tab:blue', marker='o')
    plt.scatter(X_embed[group4, 0], X_embed[group4, 1],
                s=5, c='tab:orange', marker='s')

class channel_masker(nn.Module):
    def __init__(self, args):
        super(channel_masker, self).__init__()

        self.weights = nn.Parameter(torch.distributions.Uniform(
            0, 1).sample((args.num_features, 2)))

    def reset_parameters(self):
        self.weights = torch.nn.init.xavier_uniform_(self.weights)

    def forward(self):
        return self.weights


class MLP_discriminator(torch.nn.Module):
    def __init__(self, args):
        super(MLP_discriminator, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, 1)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None, mask_node=None):
        h = self.lin(h)

        return torch.sigmoid(h)


class MLP_encoder(torch.nn.Module):
    def __init__(self, args):
        super(MLP_encoder, self).__init__()
        self.args = args

        self.lin = Linear(args.num_features, args.hidden)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def clip_parameters(self, channel_weights):
        for i in range(self.lin.weight.data.shape[1]):
            self.lin.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                   self.args.clip_e * channel_weights[i])

        # self.lin.weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.lin.weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def forward(self, x, edge_index=None, mask_node=None):
        h = self.lin(x)

        return h


class GCN_encoder_scatter(torch.nn.Module):
    def __init__(self, args):
        super(GCN_encoder_scatter, self).__init__()

        self.args = args

        self.lin = Linear(args.num_features, args.hidden, bias=False)

        self.bias = Parameter(torch.Tensor(args.hidden))

    def clip_parameters(self, channel_weights):
        for i in range(self.lin.weight.data.shape[1]):
            self.lin.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                   self.args.clip_e * channel_weights[i])

        # self.lin.weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.lin.weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.fill_(0.0)

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.lin(x)
        h = propagate2(h, edge_index) + self.bias

        return h


class GCN_encoder_spmm(torch.nn.Module):
    def __init__(self, args):
        super(GCN_encoder_spmm, self).__init__()

        self.args = args

        self.lin = Linear(args.num_features, args.hidden, bias=False)

        self.bias = Parameter(torch.Tensor(args.hidden))

    def clip_parameters(self, channel_weights):
        for i in range(self.lin.weight.data.shape[1]):
            self.lin.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                   self.args.clip_e * channel_weights[i])

        # self.lin.weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.lin.weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.fill_(0.0)

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.lin(x)
        h = torch.spmm(adj_norm_sp, h) + self.bias
        # h = propagate2(h, edge_index) + self.bias

        return h


class GIN_encoder(nn.Module):
    def __init__(self, args):
        super(GIN_encoder, self).__init__()

        self.args = args

        self.mlp = nn.Sequential(
            nn.Linear(args.num_features, args.hidden),
            # nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            # nn.Linear(args.hidden, args.hidden),
        )

        self.conv = GINConv(self.mlp)

    def clip_parameters(self, channel_weights):
        for i in range(self.mlp[0].weight.data.shape[1]):
            self.mlp[0].weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                      self.args.clip_e * channel_weights[i])

        # self.mlp[0].weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.mlp[0].weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

        # for p in self.conv.parameters():
        #     p.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.conv(x, edge_index)
        return h


class SAGE_encoder(nn.Module):
    def __init__(self, args):
        super(SAGE_encoder, self).__init__()

        self.args = args

        self.conv1 = SAGEConv(args.num_features, args.hidden, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            nn.Dropout(p=args.dropout)
        )
        self.conv2 = SAGEConv(args.hidden, args.hidden, normalize=True)
        self.conv2.aggr = 'mean'

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def clip_parameters(self, channel_weights):
        for i in range(self.conv1.lin_l.weight.data.shape[1]):
            self.conv1.lin_l.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                           self.args.clip_e * channel_weights[i])

        for i in range(self.conv1.lin_r.weight.data.shape[1]):
            self.conv1.lin_r.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                           self.args.clip_e * channel_weights[i])

        # for p in self.conv1.parameters():
        #     p.data.clamp_(-self.args.clip_e, self.args.clip_e)
        # for p in self.conv2.parameters():
        #     p.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def forward(self, x, edge_index, adj_norm_sp):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        h = self.conv2(x, edge_index)
        return h


class MLP_classifier(torch.nn.Module):
    def __init__(self, args):
        super(MLP_classifier, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, args.num_classes)

    def clip_parameters(self):
        for p in self.lin.parameters():
            p.data.clamp_(-self.args.clip_c, self.args.clip_c)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None):
        h = self.lin(h)

        return h

def sens_correlation(features, sens_idx):
    if sens_idx==-1:
        sens_idx=features.shape[-1]-1
    corr = pd.DataFrame(np.array(features)).corr()
    return corr[sens_idx].to_numpy()
def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1
def sys_normalized_adjacency(adj):

    adj = sp.coo_matrix(adj.to_dense().numpy())
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)
def index_to_mask(node_num, index):
    mask = torch.zeros(node_num, dtype=torch.bool)
    mask[index] = 1

    return mask

def get_dataset(dataname, top_k,adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx):
    #if(dataname == 'credit'):
    #    load, label_num = load_credit, 6000
    #elif(dataname == 'bail'):
    #    load, label_num = load_bail, 100
    #elif(dataname == 'german'):
    #    load, label_num = load_german, 100

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)
    edge_index, _ = from_scipy_sparse_matrix(sp.coo_matrix(adj.to_dense().numpy()))
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    #adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens = load(
    #    dataset=dataname, label_number=label_num)

    x_max, x_min = torch.max(features, dim=0)[
                       0], torch.min(features, dim=0)[0]

    norm_features = feature_norm(features)
    norm_features[:, sens_idx] = features[:, sens_idx]
    features = norm_features
    corr_matrix = sens_correlation(features, sens_idx)
    corr_idx = np.argsort(-np.abs(corr_matrix))
    if(top_k > 0):
        # corr_idx = np.concatenate((corr_idx[:top_k], corr_idx[-top_k:]))
        corr_idx = corr_idx[:top_k]
    print('return')
    return Data(x=features, edge_index=edge_index, adj_norm_sp=adj_norm_sp, y=labels.float(), train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, sens=sens), sens_idx, corr_matrix, corr_idx, x_min, x_max


class args_class():
    def __init__(self):
        pass


class FairVGNN():

    def run(self, data, args):

        criterion = nn.BCELoss()
        acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(
            args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

        data = data.to(args.device)

        generator = channel_masker(args).to(args.device)


        optimizer_g = torch.optim.Adam([
            dict(params=generator.weights, weight_decay=args.g_wd)], lr=args.g_lr)

        discriminator = MLP_discriminator(args).to(args.device)
        optimizer_d = torch.optim.Adam([
            dict(params=discriminator.lin.parameters(), weight_decay=args.d_wd)], lr=args.d_lr)

        classifier = MLP_classifier(args).to(args.device)
        optimizer_c = torch.optim.Adam([
            dict(params=classifier.lin.parameters(), weight_decay=args.c_wd)], lr=args.c_lr)

        if (args.encoder == 'MLP'):
            encoder = MLP_encoder(args).to(args.device)
            optimizer_e = torch.optim.Adam([
                dict(params=encoder.lin.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
        elif (args.encoder == 'GCN'):
            if args.prop == 'scatter':
                encoder = GCN_encoder_scatter(args).to(args.device)
            else:
                encoder = GCN_encoder_spmm(args).to(args.device)
            optimizer_e = torch.optim.Adam([
                dict(params=encoder.lin.parameters(), weight_decay=args.e_wd),
                dict(params=encoder.bias, weight_decay=args.e_wd)], lr=args.e_lr)
        elif (args.encoder == 'GIN'):
            encoder = GIN_encoder(args).to(args.device)
            optimizer_e = torch.optim.Adam([
                dict(params=encoder.conv.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
        elif (args.encoder == 'SAGE'):
            encoder = SAGE_encoder(args).to(args.device)
            optimizer_e = torch.optim.Adam([
                dict(params=encoder.conv1.parameters(), weight_decay=args.e_wd),
                dict(params=encoder.conv2.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)

        for count in range(args.runs):
            seed_everything(count + args.seed)
            generator.reset_parameters()
            discriminator.reset_parameters()
            classifier.reset_parameters()
            encoder.reset_parameters()

            best_val_tradeoff = 0
            best_val_loss = math.inf
            for epoch in tqdm(range(0, args.epochs)):
                if (args.f_mask == 'yes'):
                    generator.eval()
                    feature_weights, masks, = generator(), []
                    for k in range(args.K):
                        mask = F.gumbel_softmax(
                            feature_weights, tau=1, hard=False)[:, 0]
                        masks.append(mask)

                # train discriminator to recognize the sensitive group
                discriminator.train()
                encoder.train()
                for epoch_d in range(0, args.d_epochs):
                    optimizer_d.zero_grad()
                    optimizer_e.zero_grad()

                    if (args.f_mask == 'yes'):
                        loss_d = 0

                        for k in range(args.K):
                            x = data.x * masks[k].detach()
                            h = encoder(x, data.edge_index, data.adj_norm_sp)
                            output = discriminator(h)

                            loss_d += criterion(output.view(-1),
                                                data.x[:, args.sens_idx])

                        loss_d = loss_d / args.K
                    else:
                        h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                        output = discriminator(h)

                        loss_d = criterion(output.view(-1),
                                           data.x[:, args.sens_idx])

                    loss_d.backward()
                    optimizer_d.step()
                    optimizer_e.step()

                # train classifier
                classifier.train()
                encoder.train()
                for epoch_c in range(0, args.c_epochs):
                    optimizer_c.zero_grad()
                    optimizer_e.zero_grad()

                    if (args.f_mask == 'yes'):
                        loss_c = 0
                        for k in range(args.K):
                            x = data.x * masks[k].detach()
                            h = encoder(x, data.edge_index, data.adj_norm_sp)
                            output = classifier(h)

                            loss_c += F.binary_cross_entropy_with_logits(
                                output[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(args.device))

                        loss_c = loss_c / args.K

                    else:
                        h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                        output = classifier(h)

                        loss_c = F.binary_cross_entropy_with_logits(
                            output[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(args.device))

                    loss_c.backward()

                    optimizer_e.step()
                    optimizer_c.step()

                # train generator to fool discriminator
                generator.train()
                encoder.train()
                discriminator.eval()
                for epoch_g in range(0, args.g_epochs):
                    optimizer_g.zero_grad()
                    optimizer_e.zero_grad()

                    if (args.f_mask == 'yes'):
                        loss_g = 0
                        feature_weights = generator()
                        for k in range(args.K):
                            mask = F.gumbel_softmax(
                                feature_weights, tau=1, hard=False)[:, 0]

                            x = data.x * mask
                            h = encoder(x, data.edge_index, data.adj_norm_sp)
                            output = discriminator(h)

                            loss_g += F.mse_loss(output.view(-1),
                                                 0.5 * torch.ones_like(output.view(-1))) + args.ratio * F.mse_loss(
                                mask.view(-1), torch.ones_like(mask.view(-1)))

                        loss_g = loss_g / args.K
                    else:
                        h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                        output = discriminator(h)

                        loss_g = F.mse_loss(output.view(-1),
                                            0.5 * torch.ones_like(output.view(-1)))

                    loss_g.backward()

                    optimizer_g.step()
                    optimizer_e.step()

                if (args.weight_clip == 'yes'):
                    if (args.f_mask == 'yes'):
                        weights = torch.stack(masks).mean(dim=0)
                    else:
                        weights = torch.ones_like(data.x[0])

                    encoder.clip_parameters(weights)

                accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged3(
                    data.x, classifier, discriminator, generator, encoder, data, args)

                # print(epoch, 'Acc:', accs['test'], 'AUC_ROC:', auc_rocs['test'], 'F1:', F1s['test'],
                #       'Parity:', tmp_parity['test'], 'Equality:', tmp_equality['test'])

                if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (
                        tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
                    test_acc = accs['test']
                    test_acc_sens0=accs['test_sens0']
                    test_acc_sens1=accs['test_sens1']

                    test_auc_roc = auc_rocs['test']
                    test_auc_roc_sens0=auc_rocs['test_sens0']
                    test_auc_roc_sens1=auc_rocs['test_sens1']


                    test_f1 = F1s['test']
                    test_f1_sens0= F1s['test_sens0']
                    test_f1_sens1= F1s['test_sens1']

                    test_parity, test_equality = tmp_parity['test'], tmp_equality['test']

                    best_val_tradeoff = auc_rocs['val'] + F1s['val'] + \
                                        accs['val'] - (tmp_parity['val'] + tmp_equality['val'])

                    self.val_loss=-accs['val']


            acc[count] = test_acc
            f1[count] = test_f1
            auc_roc[count] = test_auc_roc
            parity[count] = test_parity
            equality[count] = test_equality




        return acc, f1, auc_roc, parity, equality, test_acc_sens0, test_acc_sens1, test_auc_roc_sens0, test_auc_roc_sens1, test_f1_sens0, test_f1_sens1

    def fit(self,adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx, runs=1, epochs=200, d_epochs=5, g_epochs=5,
            c_epochs=5, g_lr=0.001, g_wd=0, d_lr=0.001, d_wd=0, c_lr=0.001, c_wd=0, e_lr=0.001, e_wd=0, early_stopping=0,
            prop='spmm', dropout=0.5, hidden=16, seed=1, encoder='GCN', K=10, top_k=10, clip_e=1, f_mask='yes',weight_clip='yes',ratio=1, alpha=1):

        #parser = argparse.ArgumentParser()

        args = args_class()
        args.runs=runs
        args.epochs=epochs
        args.d_epochs=d_epochs
        args.g_epochs=g_epochs
        args.c_epochs=c_epochs
        args.g_lr=g_lr
        args.g_wd=g_wd
        args.d_lr=d_lr
        args.d_wd=d_wd
        args.c_lr=c_lr
        args.c_wd=c_wd
        args.e_lr=e_lr
        args.e_wd=e_wd
        args.early_stopping=early_stopping
        args.prop=prop
        args.dropout=dropout
        args.hidden=hidden
        args.seed=seed
        args.encoder=encoder
        args.K=K
        args.top_k=top_k
        args.clip_e=clip_e
        args.f_mask=f_mask
        args.weight_clip=weight_clip
        args.ratio=ratio
        args.alpha=alpha
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        data, args.sens_idx, args.corr_sens, args.corr_idx, args.x_min, args.x_max = get_dataset(None, args.top_k, adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx)

        #args.num_features, args.num_classes = data.x.shape[1], len(data.y.unique()) - 1
        args.num_features, args.num_classes = data.x.shape[1], 1


        args.train_ratio, args.val_ratio = torch.tensor([
            (data.y[data.train_mask] == 0).sum(), (data.y[data.train_mask] == 1).sum()]), torch.tensor([
            (data.y[data.val_mask] == 0).sum(), (data.y[data.val_mask] == 1).sum()])
        args.train_ratio, args.val_ratio = torch.max(
            args.train_ratio) / args.train_ratio, torch.max(args.val_ratio) / args.val_ratio
        args.train_ratio, args.val_ratio = args.train_ratio[
                                               data.y[data.train_mask].long()], args.val_ratio[
                                               data.y[data.val_mask].long()]

        self.args=args
        print('running')
        self.acc, self.f1, self.auc_roc, self.parity, self.equality, self.test_acc_sens0, self.test_acc_sens1, self.test_auc_roc_sens0, self.test_auc_roc_sens1, self.test_f1_sens0, self.test_f1_sens1 = self.run(data, args)

    def predict(self):
        acc, f1, auc_roc, parity, equality, test_acc_sens0, test_acc_sens1, test_auc_roc_sens0, test_auc_roc_sens1, test_f1_sens0, test_f1_sens1 = self.acc, self.f1, self.auc_roc, self.parity, self.equality, self.test_acc_sens0, self.test_acc_sens1, self.test_auc_roc_sens0, self.test_auc_roc_sens1, self.test_f1_sens0, self.test_f1_sens1


        print('auc_roc:', np.mean(auc_roc))
        print('Acc:', np.mean(acc))
        print('f1:', np.mean(f1))
        print('parity:', np.mean(parity))
        print('equality:', np.mean(equality))


        return acc.item(), auc_roc.item(), f1.item(), test_acc_sens0,test_auc_roc_sens0,test_f1_sens0,   test_acc_sens1,  test_auc_roc_sens1,  test_f1_sens1, parity.item(), equality.item()

