import torch
import pandas as pd
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader


def get_data_frames(users_csv, items_csv, user_naming, item_naming):
    df_users = pd.read_csv(users_csv)
    print('--\nusers shape:', df_users.shape)
    print(f'unique (users, articles): ({len(df_users[user_naming].unique())}, {len(df_users[item_naming].unique())})')
    print(df_users.iloc[0])

    df_items = pd.read_csv(items_csv)
    print('--\nitems shape:', df_items.shape)
    print('unique articles:', len(df_items[item_naming].unique()))
    print(df_items.iloc[0])

    # print('Set of articles is same in two files:', set(df_users[item_naming].unique()) == set(df_items[item_naming].unique()))
    # assert set(df_users[item_naming].unique()) == set(df_items[item_naming].unique())
    # assert df_items.shape[0] == df_items[item_naming].nunique()
    
    return df_users, df_items


def create_graph(df_users, df_items, user_naming, item_naming):
    # Mapping of user/item IDs to consecutive values
    # Create a mapping from unique *user/item* indices to range [0, num_user/item_nodes):
    unique_user_id = df_users[user_naming].unique()
    unique_user_id = pd.DataFrame(data={
        'userId': unique_user_id,
        'mappedID': pd.RangeIndex(len(unique_user_id)),
    })

    unique_item_id = df_items[item_naming].unique()
    unique_item_id = pd.DataFrame(data={
        'itemId': unique_item_id,
        'mappedID': pd.RangeIndex(len(unique_item_id)),
    })

    # Perform merge to obtain the edges from users to items
    df_ratings_by_user_id = pd.merge(df_users[user_naming], unique_user_id,
                                     left_on=user_naming, right_on='userId', how='left')
    list_ratings_by_user_id = torch.from_numpy(df_ratings_by_user_id['mappedID'].values)
    
    df_ratings_by_item_id = pd.merge(df_users[item_naming], unique_item_id,
                                     left_on=item_naming, right_on='itemId', how='left')
    list_ratings_by_item_id = torch.from_numpy(df_ratings_by_item_id['mappedID'].values)

    # Random edges
    # if random_edges:
    #     list_ratings_by_user_id = torch.from_numpy(np.random.randint(low=0, high=len(edges_coo[0].unique())-1, size=len(edges_coo[0])))
    #     list_ratings_by_item_id = torch.from_numpy(np.random.randint(low=0, high=len(edges_coo[1].unique())-1, size=len(edges_coo[1])))
    #     edges_coo = torch.stack([ratings_user_id, ratings_item_id], dim=0)

    # With this, we construct our `edge_index` / kind of `adj_t` in COO format following PyG semantics
    edges_coo = torch.stack([list_ratings_by_user_id, list_ratings_by_item_id], dim=0)

    return edges_coo


def create_hetero_graph(edges_coo, user_features_init='zero', user_feature_size=768, 
                        item_feature_tensor=None, item_features_init='zero', item_feature_size=768):
    data = HeteroData()
    # Save node indices
    num_unique_users = len(edges_coo[0].unique())
    num_unique_items = len(edges_coo[1].unique())
    data["user"].node_id = torch.arange(num_unique_users)
    data["item"].node_id = torch.arange(num_unique_items)

    # Add the user node features
    if user_features_init == 'random':
        data["user"].x = torch.rand(num_unique_users, user_feature_size)
    elif user_features_init == 'zero':
        data["user"].x = torch.zeros(num_unique_users, user_feature_size)

    # Add the item node features
    if item_feature_tensor is not None:
        data["item"].x = item_feature_tensor
    elif item_features_init == 'random':
        data["item"].x = torch.rand(num_unique_items, item_feature_size)
    elif item_features_init == 'zero':
        data["item"].x = torch.zeros(num_unique_items, item_feature_size)

    data["user", "rates", "item"].edge_index = edges_coo

    # We also need to make sure to add the reverse edges from items to users
    # in order to let a GNN be able to pass messages in both directions.
    # We can leverage the `T.ToUndirected()` transform for this from PyG
    data = T.ToUndirected()(data)
    del data['item', 'rev_rates', 'user'].edge_label  # Remove "reverse" label

    print(data)
    print("=============================")
    print('user num_nodes:', data["user"].num_nodes)
    print('user num_features:', data["user"].num_features)
    print('item num_nodes:', data["item"].num_nodes)
    print('item num_features:', data["item"].num_features)
    print('num_edges user->item:', data["user", "rates", "item"].num_edges)
    print('num_edges item->user:', data["item", "rev_rates", "user"].num_edges)

    return data


def create_data_loaders(data, val_ratio=0.1, test_ratio=0.1, disjoint_train_ratio=0.3,
                        num_neighbors=[10,5], neg_sampling_ratio=1.0, batch_size=512, shuffle=True):
    # First split the set of edges into
    # training (80%), validation (10%), and testing edges (10%).
    # Across the training edges, we use 70% of edges for message passing,
    # and 30% of edges for supervision (to predict/forward, find loss and backprop).
    # We further want to generate fixed negative edges for evaluation with a ratio of 1:1
    # Negative edges during training will be generated on-the-fly.
    # We can leverage the `RandomLinkSplit()` transform for this from PyG:
    transform = T.RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        disjoint_train_ratio=disjoint_train_ratio,
        is_undirected=True,                             # new
        neg_sampling_ratio=neg_sampling_ratio,          # negative sampling for val and test
        add_negative_train_samples=False,               # negative samples for train generated on-the-fly
        edge_types=("user", "rates", "item"),
        rev_edge_types=("item", "rev_rates", "user")
    )

    train_data, val_data, test_data = transform(data)

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=num_neighbors,
        neg_sampling_ratio=neg_sampling_ratio,          # generate train neg samples on the fly
        subgraph_type="bidirectional",
        edge_label_index=(("user", "rates", "item"), train_data["user", "rates", "item"].edge_label_index),
        edge_label=train_data["user", "rates", "item"].edge_label,
        batch_size=batch_size,
        shuffle=shuffle                                 # shuffle=True
    )

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=num_neighbors,
        neg_sampling_ratio=0.0,                                 # already generated for val
        subgraph_type="bidirectional",
        edge_label_index=(("user", "rates", "item"), val_data["user", "rates", "item"].edge_label_index),
        edge_label=val_data["user", "rates", "item"].edge_label,
        batch_size=int((1 + neg_sampling_ratio) * batch_size),  # to account for already sampled neg edges
        shuffle=False                                           # (1 pos + n neg) * batch_size
    )

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=num_neighbors,
        neg_sampling_ratio=0.0,                                 # already generated for test
        subgraph_type="bidirectional",
        edge_label_index=(("user", "rates", "item"), test_data["user", "rates", "item"].edge_label_index),
        edge_label=test_data["user", "rates", "item"].edge_label,
        batch_size=int((1 + neg_sampling_ratio) * batch_size),
        shuffle=False                                           # no shuffle, same batches every time                                    
    )                                                           # therefore, pos/neg samples are not balanced
                                                                # within a batch

    # Inspect samples
    # for data_laoder in [train_loader, val_loader, test_loader]:
    #     sampled_data = next(iter(data_laoder))
    #     print(sampled_data)
    #     assert sampled_data["user", "rates", "item"].edge_label_index.size(1) == (1 + neg_sampling_ratio) * batch_size
    #     assert sampled_data["user", "rates", "item"].edge_label.min() >= 0
    #     assert sampled_data["user", "rates", "item"].edge_label.max() <= 1

    return train_loader, val_loader, test_loader

   

