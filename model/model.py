import torch
from torch import Tensor
from torch_geometric.nn import to_hetero
from torch_geometric.data import HeteroData
from model.layers import *


class Model(torch.nn.Module):
    def __init__(self, graph_metadata, 
                 num_user_feats=-1, num_item_feats=-1, 
                 num_user_nodes=-1, num_item_nodes=-1,
                 user_input_layer='', item_input_layer='', input_hidden_channels=64,
                 encoder='SAGE', predictor='DOT',
                 num_encoder_layers=2, num_predictor_layers=0,
                 encoder_hidden_channels=64, predictor_hidden_channels=32, dropout=0):
        super(Model, self).__init__()

        # 1. Input Layer
        self.user_emb = torch.nn.Embedding(num_user_nodes, input_hidden_channels) if 'emb' in user_input_layer else None
        self.user_lin = torch.nn.Linear(num_user_feats, input_hidden_channels) if 'lin' in user_input_layer else None

        self.item_emb = torch.nn.Embedding(num_item_nodes, input_hidden_channels) if 'emb' in item_input_layer else None
        self.item_lin = torch.nn.Linear(num_item_feats, input_hidden_channels) if 'lin' in item_input_layer else None

        # 2. GNN Layer - Encoder
        # First, instantiate homogeneous GNN
        # -1 to derive the size from the first inputs to model.forward()
        # a tuple = sizes of source and target dimensionalities
        encoder_in_channels = (-1, -1)
        self.encoder = create_gnn_layer(encoder_in_channels, encoder_hidden_channels, encoder_hidden_channels, num_encoder_layers, dropout, encoder)
        # Then, convert GNN model into a heterogeneous variant
        self.encoder = to_hetero(self.encoder, metadata=graph_metadata, aggr='mean')

        # 3. Predictor Layer - Decoder
        self.predictor = create_predictor_layer(encoder_hidden_channels, predictor_hidden_channels, num_predictor_layers, dropout, predictor)

    def reset_parameters(self):
        if self.user_emb is not None: torch.nn.init.xavier_uniform_(self.user_emb.weight)
        if self.user_lin is not None: self.user_lin.reset_parameters()

        if self.item_emb is not None: torch.nn.init.xavier_uniform_(self.item_emb.weight)
        if self.item_lin is not None: self.item_lin.reset_parameters()
        self.encoder.reset_parameters()
        self.predictor.reset_parameters()

    def forward(self, data: HeteroData) -> Tensor:
        # 1. Input Layer
        x_dict = {"user": 0, "item": 0}
        if self.user_emb is not None: x_dict["user"] = self.user_emb(data["user"].node_id)
        if self.user_lin is not None: x_dict["user"] += self.user_lin(data["user"].x)

        if self.item_emb is not None: x_dict["item"] = self.item_emb(data["item"].node_id)
        if self.item_lin is not None: x_dict["item"] += self.item_lin(data["item"].x)

        # 2. Encode
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.encoder(x_dict, data.edge_index_dict)
        
        # 3. Decode
        # Convert node embeddings to edge-level representations
        edge_feat_user = x_dict["user"][data["user", "rates", "item"].edge_label_index[0]]    # feat[ids of users in coo]
        edge_feat_item = x_dict["item"][data["user", "rates", "item"].edge_label_index[1]]    # feat[ids of items in coo]
        pred = self.predictor(edge_feat_user, edge_feat_item)

        return pred


def create_gnn_layer(input_channels, hidden_channels, out_channels, num_layers, dropout=0, layer_name='SAGE'):
    layer_name = layer_name.upper()
    if layer_name == 'SAGE':
        return SAGE(input_channels, hidden_channels, out_channels, num_layers, dropout)
    elif layer_name == 'GAT':
        return GAT(input_channels, hidden_channels, out_channels, num_layers, dropout)
    elif layer_name == 'GCN':
        return GCN(input_channels, hidden_channels, out_channels, num_layers, dropout)


def create_predictor_layer(input_channels, hidden_channels, num_layers, dropout=0, layer_name='DOT'):
    layer_name = layer_name.upper()
    if layer_name == 'MLP':
        return MLPPredictor(input_channels, hidden_channels, 1, num_layers, dropout)
    elif layer_name == 'MLPCAT':
        return MLPCatPredictor(input_channels, hidden_channels, 1, num_layers, dropout)
    elif layer_name == 'MLPDOT':
        return MLPDotPredictor(input_channels, 1, num_layers, dropout)
    elif layer_name == 'DOT':
        return DotPredictor()
    elif layer_name == 'BIL':
        return BilinearPredictor(hidden_channels)

