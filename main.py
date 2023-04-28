import time
import argparse
import json
import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from model.model import Model
from model.data import get_data_frames, create_graph, create_hetero_graph, create_data_loaders


def train(model, optimizer, criterion, data_loader, device, run, epochs, eval_steps):
    train_loss, valid_scores, test_scores = [], [], []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = total_examples = 0
        for batch_data in tqdm.tqdm(data_loader['train'], disable=True):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            
            pred = model(batch_data)
            ground_truth = batch_data['user', 'item'].edge_label

            loss = criterion(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        
        epoch_loss = total_loss / total_examples
        train_loss.append(epoch_loss)

        if epoch % eval_steps == 0:
            print(f"\nEpoch: {epoch:03d}, Train Loss: {epoch_loss:.4f}")

            val_auc_score, val_f1_score = test(model, data_loader['val'], device)
            test_auc_score, test_f1_score = test(model, data_loader['test'], device)
            
            valid_scores.append((val_auc_score, val_f1_score))
            test_scores.append((test_auc_score, test_f1_score))

            spent_time = time.time() - start_time
            eval_results = (f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Train Loss: {epoch_loss:.4f}, '
                            f'Valid AUC: {100 * val_auc_score:.2f}%, '
                            f'Valid F1: {100 * val_f1_score:.2f}%, '
                            f'Test AUC: {100 * test_auc_score:.2f}%, '
                            f'Test F1: {100 * test_f1_score:.2f}%')
            print(eval_results)
            print(f'---\nTraining Time Per Epoch: {spent_time / eval_steps: .4f} s\n---')
            start_time = time.time()

    return train_loss, valid_scores, test_scores


def test(model, data_loader, device):
    model.eval()
    predictions, ground_truths = [], []
    for batch_data in tqdm.tqdm(data_loader, disable=True):
        with torch.no_grad():
            batch_data = batch_data.to(device)
            predictions.append(model(batch_data))
            ground_truths.append(batch_data['user', 'item'].edge_label)

    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    ground_truths = torch.cat(ground_truths, dim=0).cpu().numpy()
    
    out_sigmoid = torch.Tensor(predictions).sigmoid().cpu().numpy()
    out_probabilities = np.rint(out_sigmoid)

    auc_sc = roc_auc_score(ground_truths, out_sigmoid)
    f1_sc = f1_score(ground_truths, out_probabilities)

    return auc_sc, f1_sc


def main():
    parser = argparse.ArgumentParser(description='News Recommender GNN Model')
    parser.add_argument("--data", default="cit_pt", choices=["adressa_no","adressa_tr",
                                                             "cit_pt","cit_tr","cit_en",
                                                             "cit_pten","cit_tren",
                                                             "mind"],
                        help = "default: %(default)s")
    parser.add_argument('--plm', default="ptbert", choices=["mbert","enbert","nbbert",
                                                               "ptbert","xlm","gpt2"],
                        help = "plm model for item nodes, default: %(default)s")    
    parser.add_argument('--use_seperate_test_data', action='store_true')
    parser.add_argument('--num_user_features', type=int, default=32,
                        help = "-1 for don't use user_features, default: %(default)i")
    parser.add_argument('--num_item_features', type=int, default=768,
                        help = "-1 for don't use item_features, default: %(default)i")
    parser.add_argument('--user_input_layer', default="lin", choices=["emb","lin"],
                        help = "input layer for user nodes, default: %(default)s")
    parser.add_argument('--item_input_layer', default="lin", choices=["emb","lin","emb+lin"],
                        help = "input layer for item nodes, default: %(default)s")
    parser.add_argument('--input_hidden_channels', type=int, default=64,
                        help = "default: %(default)i")
    parser.add_argument('--encoder', default="SAGE", choices=["SAGE","GAT","GCN"],
                        help = "default: %(default)s")
    parser.add_argument('--predictor', default="DOT", choices=["MLP","MLPDOT","DOT"],
                        help = "default: %(default)s")
    parser.add_argument('--encoder_layers', type=int, default=2,
                        help = "default: %(default)i")
    parser.add_argument('--predictor_layers', type=int, default=0,
                        help = "0 for DOT predictor, default: %(default)i")
    parser.add_argument('--encoder_hidden_channels', type=int, default=64,
                        help = "default: %(default)i")
    parser.add_argument('--predictor_hidden_channels', type=int, default=32,
                        help = "default: %(default)i")
    
    parser.add_argument('--dropout', type=float, default=0.0,
                        help = "default: %(default)f")
    parser.add_argument('--batch_size', type=int, default=1024,
                        help = "default: %(default)i")
    parser.add_argument('--lr', type=float, default=0.001,
                        help = "default: %(default)f")
    parser.add_argument('--epochs', type=int, default=-1,
                        help = "-1 to use model default value, default: %(default)i")
    parser.add_argument('--eval_steps', type=int, default=1,
                        help = "number of epochs at which logs printed, default: %(default)i")
    parser.add_argument('--device', type=int, default=0,
                        help = "default: %(default)i")
    parser.add_argument('--runs', type=int, default=1,
                        help = "default: %(default)i")
    args = parser.parse_args()
    print(str(args))

    # device setup
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Device: {device}")

    # get data
    with open('datasets/info.json', 'r') as f:
        datasets_info = json.load(f)
    
    users_csv = datasets_info[args.data]['users_csv']
    items_csv = datasets_info[args.data]['items_csv']
    user_naming = datasets_info[args.data]['user_naming']
    item_naming = datasets_info[args.data]['item_naming']
    feats_file = datasets_info[args.data]['plm'][args.plm]
    epochs = datasets_info[args.data]['epochs'] if args.epochs == -1 else args.epochs
    
    print(f'Loading user and item data frames for: {args.data}')
    df_users, df_items = get_data_frames(users_csv, items_csv, user_naming, item_naming)
    item_feature_tensor = torch.load(feats_file, map_location=device)
    user_features_init = '' if args.num_user_features == -1 else 'zero'

    disjoint_train_ratio = 0.3
    neg_sampling_ratio = 1.0
    num_neighbors = [10, 5]

    if args.use_seperate_test_data:
        print(f'Doing seperate user train and test data for {args.data}')
        users_train_csv = datasets_info[args.data]['users_train_csv']
        users_test_csv = datasets_info[args.data]['users_test_csv']
        
        print(f'Train:')
        df_users_train, _ = get_data_frames(users_train_csv, items_csv, user_naming, item_naming)
        train_edges_coo = create_graph(df_users_train, df_items, user_naming, item_naming)
        train_data = create_hetero_graph(edges_coo=train_edges_coo,
                                                user_features_init=user_features_init,
                                                user_feature_size=args.num_user_features,
                                                item_feature_tensor=item_feature_tensor)
        
        print(f'Test:')
        df_users_test, _ = get_data_frames(users_test_csv, items_csv, user_naming, item_naming)
        test_edges_coo = create_graph(df_users_test, df_items, user_naming, item_naming)
        test_data = create_hetero_graph(edges_coo=test_edges_coo,
                                        user_features_init=user_features_init,
                                        user_feature_size=args.num_user_features,
                                        item_feature_tensor=item_feature_tensor)
        
        val_ratio, test_ratio = 0.1, 0.1    
        train_loader, val_loader, _ = create_data_loaders(data=train_data, val_ratio=val_ratio, test_ratio=test_ratio, 
                                                                    disjoint_train_ratio=disjoint_train_ratio, 
                                                                    num_neighbors=num_neighbors, 
                                                                    neg_sampling_ratio=neg_sampling_ratio, 
                                                                    batch_size=args.batch_size, shuffle=True)
        val_ratio, test_ratio = 0.0, 0.9
        _, _, test_loader = create_data_loaders(data=test_data, val_ratio=val_ratio, test_ratio=test_ratio, 
                                                                    disjoint_train_ratio=disjoint_train_ratio, 
                                                                    num_neighbors=num_neighbors, 
                                                                    neg_sampling_ratio=neg_sampling_ratio, 
                                                                    batch_size=args.batch_size, shuffle=True)
        data = train_data
    else:
        val_ratio, test_ratio = 0.1, 0.1
        edges_coo = create_graph(df_users, df_items, user_naming, item_naming)
        data = create_hetero_graph(edges_coo=edges_coo,
                                   user_features_init=user_features_init,
                                   user_feature_size=args.num_user_features,
                                   item_feature_tensor=item_feature_tensor)
        train_loader, val_loader, test_loader = create_data_loaders(data=data, val_ratio=val_ratio, test_ratio=test_ratio, 
                                                                    disjoint_train_ratio=disjoint_train_ratio, 
                                                                    num_neighbors=num_neighbors, 
                                                                    neg_sampling_ratio=neg_sampling_ratio, 
                                                                    batch_size=args.batch_size, shuffle=True)
    
    data_loader = {
        'train': train_loader,
        'val':   val_loader,
        'test':  test_loader
    }

    # TODO Pre-compute GCN normalization
    # if args.gnn_model == 'GCN': data.adj_t = gcn_normalization(data.adj_t)

    # data contains the whole graph
    num_user_nodes = data["user"].num_nodes if 'emb' in args.user_input_layer else -1
    num_item_nodes = data["item"].num_nodes if 'emb' in args.item_input_layer else -1
    graph_metadata = data.metadata()

    model = Model(
        graph_metadata=graph_metadata,
        num_user_feats=args.num_user_features,
        num_item_feats=args.num_item_features, 
        num_user_nodes=num_user_nodes, 
        num_item_nodes=num_item_nodes, 
        user_input_layer=args.user_input_layer,
        item_input_layer=args.item_input_layer,
        input_hidden_channels=args.input_hidden_channels,
        encoder=args.encoder,
        predictor=args.predictor,
        num_encoder_layers=args.encoder_layers,
        num_predictor_layers=args.predictor_layers,
        encoder_hidden_channels=args.encoder_hidden_channels,
        predictor_hidden_channels=args.predictor_hidden_channels,
        dropout=args.dropout
    )
    model = model.to(device)
    print(model)

    for run in range(args.runs):
        # model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = F.binary_cross_entropy_with_logits
        train_loss, valid_scores, test_scores = train(model, optimizer, criterion,
                                                      data_loader, device, run, 
                                                      epochs, args.eval_steps)

    total_params = sum(p.numel() for param in model.parameters() for p in param)
    print(f'Total number of model parameters is {total_params}')


if __name__ == "__main__":
    main()
