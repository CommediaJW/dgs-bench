import torch
import sys
import dgl


def get_local_subgraph(graph: dgl.DGLGraph, local_train_nids, num_hops):
    nodes = local_train_nids
    for _ in range(num_hops):
        local_subgraph = dgl.in_subgraph(graph, nodes)
        nodes = local_subgraph.nodes()
