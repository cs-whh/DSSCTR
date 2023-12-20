import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
import torch
import random
from sklearn.neighbors import kneighbors_graph
import torch_geometric as torchgeo
import yaml
import argparse
nmi = normalized_mutual_info_score

def get_config(parser):
    parser.add_argument("--config", default='.config/cifar10.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    return config


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def cluster_accuracy(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)

def bulid_graph(features, K):
    return kneighbors_graph(features, K, mode='connectivity', metric='cosine')



def bulid_pyg_data(features, sparse_adj):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pyg_graph = torchgeo.data.Data()
    pyg_graph.x = torch.from_numpy(features).to(device)
    edge_index = torch.from_numpy(np.transpose(np.stack(sparse_adj.nonzero(), axis=1))).to(device)
    pyg_graph.edge_index = edge_index
    pyg_graph.edge_index = torchgeo.utils.to_undirected(pyg_graph.edge_index)
    pyg_graph.num_nodes = features.shape[0]
    return pyg_graph