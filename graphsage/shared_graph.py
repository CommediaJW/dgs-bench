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
        print("start building shared graph")
        indptr = graph.adj_sparse('csc')[0]
        indices = graph.adj_sparse('csc')[1]
        features = graph.ndata.pop('features')
        labels = graph.ndata.pop('labels')
        train_mask = graph.ndata.pop("train_mask").bool()
        train_idx = graph.nodes()[train_mask]
        del train_mask
        test_mask = graph.ndata.pop("test_mask").bool()
        test_idx = graph.nodes()[test_mask]
        del test_mask
        val_mask = graph.ndata.pop("val_mask").bool()
        val_idx = graph.nodes()[val_mask]
        del val_mask
        if prob:
            probs = torch.rand(graph.num_edges()).float()
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

    graph.ndata.clear()
    graph.edata.clear()

    print("generating shared indptr")
    shared_indptr = create_shared_tensor(indptr)
    del indptr
    print("generating shared indices")
    shared_indices = create_shared_tensor(indices)
    del indices
    print("generating shared features")
    shared_features = create_shared_tensor(features)
    del features
    print("generating shared labels")
    shared_labels = create_shared_tensor(labels)
    del labels
    print("generating shared train_idx")
    shared_train_idx = create_shared_tensor(train_idx)
    del train_idx
    print("generating shared test_idx")
    shared_test_idx = create_shared_tensor(test_idx)
    del test_idx
    print("generating shared val_idx")
    shared_val_idx = create_shared_tensor(val_idx)
    del val_idx
    if prob:
        print("generating shared probs")
        shared_probs = create_shared_tensor(probs)
        del probs

    shared_graph = dgl.graph(('csc', (shared_indptr, shared_indices, [])))
    shared_graph.ndata['features'] = shared_features
    shared_graph.ndata['labels'] = shared_labels
    if prob:
        shared_graph.edata[prob] = shared_probs

    split_idx = {}
    split_idx['train_idx'] = shared_train_idx
    split_idx['test_idx'] = shared_test_idx
    split_idx['val_idx'] = shared_val_idx

    print("finish building shared graph")

    if rank == root_rank:
        broadcast_list = [num_classes]
    else:
        broadcast_list = [None]
    dist.broadcast_object_list(broadcast_list, root_rank)
    num_classes = broadcast_list[0]

    return shared_graph, split_idx, num_classes
