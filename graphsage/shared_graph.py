import dgl
import torch
import torch.distributed as dist
from dgl.multiprocessing import shared_tensor


def create_shared_tensor(tensor, root_rank=0):
    rank = dist.get_rank()
    if rank == root_rank:
        broadcast_list = [tensor.shape, tensor.dtype]
    else:
        broadcast_list = [None, None]

    dist.broadcast_object_list(broadcast_list, root_rank)

    shared = shared_tensor(broadcast_list[0], broadcast_list[1])

    if rank == root_rank:
        shared.copy_(tensor)

    dist.barrier()

    return shared


def create_shared_graph(graph, num_classes, prob=None, root_rank=0):
    rank = dist.get_rank()

    if rank == root_rank:
        indptr = graph.adj_sparse('csc')[0]
        indices = graph.adj_sparse('csc')[1]
        features = graph.ndata['features']
        labels = graph.ndata['labels']
        train_idx = graph.nodes()[graph.ndata["train_mask"]]
        test_idx = graph.nodes()[graph.ndata["test_mask"]]
        val_idx = graph.nodes()[graph.ndata["val_mask"]]
        if prob:
            probs = torch.ones(graph.num_edges()).float()
    else:
        indptr = None
        indices = None
        features = None
        labels = None
        train_idx = None
        test_idx = None
        val_idx = None
        if prob:
            probs = None

    shared_indptr = create_shared_tensor(indptr)
    shared_indices = create_shared_tensor(indices)
    shared_features = create_shared_tensor(features)
    shared_labels = create_shared_tensor(labels)
    shared_train_idx = create_shared_tensor(train_idx)
    shared_test_idx = create_shared_tensor(test_idx)
    shared_val_idx = create_shared_tensor(val_idx)
    if prob:
        shared_probs = create_shared_tensor(probs)

    shared_graph = dgl.graph(('csc', (shared_indptr, shared_indices, [])))
    shared_graph.ndata['features'] = shared_features
    shared_graph.ndata['labels'] = shared_labels
    if prob:
        shared_graph.edata[prob] = shared_probs

    split_idx = {}
    split_idx['train_idx'] = shared_train_idx
    split_idx['test_idx'] = shared_test_idx
    split_idx['val_idx'] = shared_val_idx

    if rank == root_rank:
        broadcast_list = [num_classes]
    else:
        broadcast_list = [None]
    dist.broadcast_object_list(broadcast_list, root_rank)
    num_classes = broadcast_list[0]

    return shared_graph, split_idx, num_classes
