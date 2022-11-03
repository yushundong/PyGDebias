import torch.nn as nn
import numpy as np
from models.GCN import GCN, GCN_Body
from models.GAT import GAT, GAT_body
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from utils import load_data, accuracy, load_pokec
import time
import torch


def get_model(nfeat, args):
    if args.model == "GCN":
        model = GCN_Body(nfeat, args.num_hidden, args.dropout)
    elif args.model == "GAT":
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT_body(args.num_layers, nfeat, args.num_hidden, heads, args.dropout, args.attn_drop,
                         args.negative_slope, args.residual)
    else:
        print("Model not implement")
        return

    return model


class FairGNN(nn.Module):

    def __init__(self, nfeat, args):
        super(FairGNN, self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        self.estimator = GCN(nfeat, args.hidden, 1, dropout)
        self.GNN = get_model(nfeat, args)
        self.classifier = nn.Linear(nhid, 1)
        self.adv = nn.Linear(nhid, 1)

        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters()) + list(self.estimator.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()

        self.G_loss = 0
        self.A_loss = 0

    def fair_metric(self, sens, labels, output, idx):
        val_y = labels[idx].cpu().numpy()
        idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
        idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

        idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

        pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
        parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1))
        equality = abs(sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1))

        return parity, equality

    def forward(self, g, x):
        s = self.estimator(g, x)
        z = self.GNN(g, x)
        y = self.classifier(z)
        return y, s

    def optimize(self, g, x, labels, idx_train, sens, idx_sens_train):
        self.train()

        ### update E, G
        self.adv.requires_grad_(False)
        self.optimizer_G.zero_grad()

        s = self.estimator(g, x)
        h = self.GNN(g, x)
        y = self.classifier(h)

        s_g = self.adv(h)

        s_score = torch.sigmoid(s.detach())
        # s_score = (s_score > 0.5).float()
        s_score[idx_sens_train] = sens[idx_sens_train].unsqueeze(1).float()
        y_score = torch.sigmoid(y)
        self.cov = torch.abs(torch.mean((s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))))

        self.cls_loss = self.criterion(y[idx_train], labels[idx_train].unsqueeze(1).float())
        self.adv_loss = self.criterion(s_g, s_score)

        self.G_loss = self.cls_loss + self.args.alpha * self.cov - self.args.beta * self.adv_loss
        self.G_loss.backward()
        self.optimizer_G.step()

        ## update Adv
        self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()
        s_g = self.adv(h.detach())
        self.A_loss = self.criterion(s_g, s_score)
        self.A_loss.backward()
        self.optimizer_A.step()



    def fit(self, g, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train, device='cuda'):


        # with args

        self = self.to(device)
        features = features.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)
        sens = sens.to(device)
        idx_sens_train = idx_sens_train.to(device)

        args = self.args
        t_total = time.time()
        best_result = {}
        best_fair = 100

        self.g = g
        self.x = features
        self.labels = labels
        self.sens = sens


        for epoch in range(args.epochs):
            t = time.time()
            self.train()
            self.optimize(g, features, labels, idx_train, sens, idx_sens_train)
            cov = self.cov
            cls_loss = self.cls_loss
            adv_loss = self.adv_loss
            self.eval()
            output, s = self(g, features)
            acc_val = accuracy(output[idx_val], labels[idx_val])
            roc_val = roc_auc_score(labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy())

            acc_sens = accuracy(s[idx_test], sens[idx_test])

            parity_val, equality_val = self.fair_metric(sens, labels, output, idx_val)

            acc_test = accuracy(output[idx_test], labels[idx_test])
            roc_test = roc_auc_score(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
            parity, equality = self.fair_metric(sens, labels, output, idx_test)
            if acc_val > args.acc and roc_val > args.roc:

                if best_fair > parity_val + equality_val:
                    best_fair = parity_val + equality_val

                    best_result['acc'] = acc_test.item()
                    best_result['roc'] = roc_test
                    best_result['parity'] = parity
                    best_result['equality'] = equality

                print("=================================")

                print('Epoch: {:04d}'.format(epoch + 1),
                      'cov: {:.4f}'.format(cov.item()),
                      'cls: {:.4f}'.format(cls_loss.item()),
                      'adv: {:.4f}'.format(adv_loss.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      "roc_val: {:.4f}".format(roc_val),
                      "parity_val: {:.4f}".format(parity_val),
                      "equality: {:.4f}".format(equality_val))
                print("Test:",
                      "accuracy: {:.4f}".format(acc_test.item()),
                      "roc: {:.4f}".format(roc_test),
                      "acc_sens: {:.4f}".format(acc_sens),
                      "parity: {:.4f}".format(parity),
                      "equality: {:.4f}".format(equality))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


        return(best_result, acc_sens)

    def predict(self, idx_test):

        output, s = self.forward(self.g, self.x)
        acc_test = accuracy(output[idx_test], self.labels[idx_test])
        roc_test = roc_auc_score(self.labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
        parity, equality = self.fair_metric(self.sens, self.labels, output, idx_test)
        acc_sens = accuracy(s[idx_test], self.sens[idx_test])

        print("Test:",
        "accuracy: {:.4f}".format(acc_test.item()),
        "roc: {:.4f}".format(roc_test),
        "acc_sens: {:.4f}".format(acc_sens),
        "parity: {:.4f}".format(parity),
        "equality: {:.4f}".format(equality))