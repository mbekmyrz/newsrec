import torch
import numpy as np


def compute_hits(y_pred_pos, y_pred_neg, K):
        '''
            compute Hits@K
            For each positive target node, the negative target nodes are the same.
            y_pred_neg is an array.
            rank y_pred_pos[i] against y_pred_neg for each i
        '''

        if len(y_pred_neg) < K:
            return {f'hits@{K}': 1.0}

        kth_score_in_negative_edges = np.sort(y_pred_neg)[-K]
        hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)

        return {f'hits@{K}': hitsK}


def evaluate_hits(val_pred_pos, val_pred_neg, test_pred_pos, test_pred_neg):
    results = {}
    for K in [20, 50, 100]:
        valid_hits = compute_hits(val_pred_pos, val_pred_neg, K)
        test_hits = compute_hits(test_pred_pos, test_pred_neg, K)
        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results


def gcn_normalization(adj_t):
    adj_t = adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t

