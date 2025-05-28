#!/usr/bin/env python
# coding: utf-8

"""
egnn_model.py: Defines the EGNN architecture for DR prediction and coordinate regression.
Includes helper functions and optional training logic.
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Reproducibility
random.seed(2)
np.random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed(2)
torch.cuda.manual_seed_all(2)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Compute sum of data elements per segment."""
    out = data.new_zeros((num_segments, data.size(1)))
    scatter_add(data, segment_ids, out=out, dim=0)
    return out

def unsorted_segment_mean(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Compute mean of data elements per segment."""
    out_sum = unsorted_segment_sum(data, segment_ids, num_segments)
    count = data.new_zeros((num_segments, data.size(1)))
    ones = torch.ones_like(data)
    scatter_add(ones, segment_ids, out=count, dim=0)
    return out_sum / torch.clamp(count, min=1)

class E_GCL(nn.Module):
    """E(n)-GCL layer from Satorras et al."""
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True,
                 attention=False, normalize=False, coords_agg='mean', tanh=False, dropout=0.2):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.dropout = dropout

        edge_coords_nf = 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf), act_fn, nn.Dropout(dropout),
            nn.Linear(hidden_nf, hidden_nf), act_fn, nn.Dropout(dropout)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf), act_fn, nn.Dropout(dropout),
            nn.Linear(hidden_nf, output_nf), nn.Dropout(dropout)
        )
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp_list = [nn.Linear(hidden_nf, hidden_nf), act_fn, nn.Dropout(dropout), layer]
        if self.tanh:
            coord_mlp_list.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp_list)
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Dropout(dropout), nn.Sigmoid())

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, dim=1, keepdim=True)
        if self.normalize:
            norm = torch.sqrt(radial + self.epsilon)
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def edge_model(self, h_row, h_col, radial, edge_attr):
        out = torch.cat([h_row, h_col, radial, edge_attr], dim=1) if edge_attr is not None else torch.cat([h_row, h_col, radial], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            out = out * self.att_mlp(out)
        return out

    def node_model(self, x, edge_index, edge_feat, node_attr=None):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_feat, row, num_segments=x.size(0))
        agg = torch.cat([x, agg, node_attr], dim=1) if node_attr is not None else torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        return x + out if self.residual else out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0)) if self.coords_agg == 'sum' else unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        return coord + agg

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        h_row, h_col = h[row], h[col]
        e_ij = self.edge_model(h_row, h_col, radial, edge_attr)
        coord_new = self.coord_model(coord, edge_index, coord_diff, e_ij)
        h_new = self.node_model(h, edge_index, e_ij, node_attr)
        return h_new, coord_new, e_ij

class EGNN(nn.Module):
    """Stacks multiple E_GCL layers."""
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, n_layers=4, residual=True,
                 attention=False, normalize=False, coords_agg='mean', tanh=False, device='cpu', dropout=0.2):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)
        self.embedding_out = nn.Linear(hidden_nf, out_node_nf)
        for i in range(n_layers):
            self.add_module(f"gcl_{i}", E_GCL(
                input_nf=hidden_nf, output_nf=hidden_nf, hidden_nf=hidden_nf, edges_in_d=in_edge_nf,
                residual=residual, attention=attention, normalize=normalize, coords_agg=coords_agg,
                tanh=tanh, dropout=dropout
            ))
        self.to(device)

    def forward(self, h, coord, edge_index, edge_attr=None):
        h = self.embedding_in(h)
        for i in range(self.n_layers):
            h, coord, _ = self._modules[f"gcl_{i}"](h, edge_index, coord, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, coord

class EarlyStopping:
    """Stop training if validation loss doesn't improve."""
    def __init__(self, patience=10, verbose=False, delta=0, output_dir='./', filename='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.output_dir = output_dir
        self.filename = filename

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.output_dir, self.filename))
        self.val_loss_min = val_loss

class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean() if self.reduction == 'mean' else F_loss.sum() if self.reduction == 'sum' else F_loss

def prepare_data():
    """Prepare training and de novo datasets."""
    dir_path = "/home/hyojin0912/Activity/Data/GPCR_BR_Graph/"
    pt_files = [f for f in os.listdir(dir_path) if f.endswith('.pt')]
    df_dr = pd.read_csv(f"../Output/Final/AF2/GPCR_DR_sets_thr_2.0_2.0_0.5.csv")
    df_dr['Has_DR'] = df_dr.apply(lambda row: len(eval(row['DR-Ago'])) > 0 or len(eval(row['DR-Ant'])) > 0 or len(eval(row['DR-State'])) > 0, axis=1)
    df_dr_filtered = df_dr[df_dr['Has_DR']].copy()
    valid_uniprots = set(df_dr_filtered['UniProt_ID'].values)

    dataset = []
    de_novo_data = []
    for pt_file in pt_files:
        uniprot_id = pt_file.split('.')[0]
        graph = torch.load(os.path.join(dir_path, pt_file))
        graph.uniprot_id = uniprot_id
        if uniprot_id in valid_uniprots:
            dr_row = df_dr[df_dr['UniProt_ID'] == uniprot_id].iloc[0]
            all_drs = set(eval(dr_row['DR-Ago'])).union(eval(dr_row['DR-Ant'])).union(eval(dr_row['DR-State']))
            labels = torch.tensor([1 if res_num.item() in all_drs else 0 for res_num in graph.residue_numbers], dtype=torch.long)
            graph.y = labels
            dataset.append(graph)
        else:
            de_novo_data.append(graph)
    return dataset, de_novo_data

def reconstruct_graphs(node_indices, node_data):
    """Reconstruct graphs from node indices."""
    graph_dict = {}
    for idx in node_indices:
        node = node_data[idx]
        uniprot_id = node['uniprot_id']
        graph_dict.setdefault(uniprot_id, []).append(idx)

    graphs = []
    for uniprot_id, indices in graph_dict.items():
        sample_node = node_data[indices[0]]
        orig_graph = sample_node['graph']
        node_mask = torch.zeros(orig_graph.x.size(0), dtype=torch.bool)
        for idx in indices:
            node_mask[node_data[idx]['node_idx']] = True
        x = orig_graph.x[node_mask]
        pos = orig_graph.pos[node_mask]
        y = orig_graph.y[node_mask]
        residue_numbers = orig_graph.residue_numbers[node_mask]
        edge_mask = node_mask[orig_graph.edge_index[0]] & node_mask[orig_graph.edge_index[1]]
        edge_index = orig_graph.edge_index[:, edge_mask]
        node_idx_map = torch.zeros(orig_graph.x.size(0), dtype=torch.long)
        node_idx_map[node_mask] = torch.arange(x.size(0))
        edge_index = node_idx_map[edge_index]
        graphs.append(Data(x=x, pos=pos, edge_index=edge_index, y=y, residue_numbers=residue_numbers, uniprot_id=uniprot_id))
    return graphs

def train_model(train_loader, val_loader, model, device, output_dir, num_epochs=100, accumulate_steps=4):
    """Train the EGNN model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=20, verbose=True, output_dir=output_dir, filename='checkpoint.pt')
    criterion = FocalLoss(alpha=1, gamma=2).to(device)
    step = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            logits, _ = model(data.x, data.pos, data.edge_index)
            loss = criterion(logits.view(-1), data.y.float())
            loss.backward()
            step += 1
            if step % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                logits, _ = model(data.x, data.pos, data.edge_index)
                loss = criterion(logits.view(-1), data.y.float())
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
        early_stopping(val_loss, model)
        scheduler.step(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoint.pt')))
    return model

def main():
    """Train EGNN model and save predictions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = "../Output/Final/EGNN_DR_Pred/"
    os.makedirs(output_dir, exist_ok=True)

    dataset, de_novo_data = prepare_data()
    node_data = [
        {'uniprot_id': graph.uniprot_id, 'node_idx': i, 'x': graph.x[i], 'pos': graph.pos[i],
         'y': graph.y[i], 'residue_number': graph.residue_numbers[i], 'graph': graph}
        for graph in dataset for i in range(graph.x.size(0))
    ]
    labels = np.array([node['y'].item() for node in node_data])
    train_idx, test_idx = train_test_split(range(len(node_data)), test_size=0.2, stratify=labels, random_state=2)
    train_data = reconstruct_graphs(train_idx, node_data)
    test_data = reconstruct_graphs(test_idx, node_data)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    de_novo_loader = DataLoader(de_novo_data, batch_size=1, shuffle=False)

    model = EGNN(
        in_node_nf=1280, hidden_nf=128, out_node_nf=1, n_layers=4, residual=True, attention=True,
        normalize=True, coords_agg='mean', tanh=True, device=device, dropout=0.2
    )
    model = train_model(train_loader, test_loader, model, device, output_dir)

    def get_predictions(loader, model, device, is_de_novo=False):
        model.eval()
        all_uniprots, all_res_nums, all_labels, all_probs = [], [], [] if not is_de_novo else None, []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                logits, _ = model(data.x, data.pos, data.edge_index)
                probs = torch.sigmoid(logits).cpu().numpy().flatten().tolist()
                all_uniprots.extend([data.uniprot_id] * data.x.size(0))
                all_res_nums.extend(data.residue_numbers.cpu().numpy().tolist())
                all_probs.extend(probs)
                if not is_de_novo:
                    all_labels.extend(data.y.cpu().numpy().flatten().tolist())
        return all_uniprots, all_res_nums, all_labels, all_probs

    train_uniprots, train_res_nums, train_labels, train_probs = get_predictions(train_loader, model, device)
    test_uniprots, test_res_nums, test_labels, test_probs = get_predictions(test_loader, model, device)
    de_novo_uniprots, de_novo_res_nums, de_novo_labels, de_novo_probs = get_predictions(de_novo_loader, model, device, is_de_novo=True)

    pd.DataFrame({
        'UniProt_ID': train_uniprots, 'Residue_Number': train_res_nums, 'Label': train_labels, 'Prob_DR': train_probs
    }).to_csv(os.path.join(output_dir, 'train_predictions_seed2.csv'), index=False)
    pd.DataFrame({
        'UniProt_ID': test_uniprots, 'Residue_Number': test_res_nums, 'Label': test_labels, 'Prob_DR': test_probs
    }).to_csv(os.path.join(output_dir, 'test_predictions_seed2.csv'), index=False)
    if de_novo_uniprots:
        pd.DataFrame({
            'UniProt_ID': de_novo_uniprots, 'Residue_Number': de_novo_res_nums, 'Prob_DR': de_novo_probs
        }).to_csv(os.path.join(output_dir, 'de_novo_predictions_seed2.csv'), index=False)
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model_seed2.pt'))
    print("Training complete and predictions saved.")

if __name__ == "__main__":
    main()