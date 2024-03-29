import argparse
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import time
import numpy as np
from utils.models import SAGE
from utils.load_graph import load_papers400m_sparse, load_ogb
from GraphCache.cache import StructureCacheServer, FeatureCacheServer, get_available_memory
from GraphCache.dataloading import SeedGenerator

torch.ops.load_library("../GPU-Graph-Caching/build/libdgs.so")


def get_local_subgraph(graph: dgl.DGLGraph, local_train_nids, num_hops):
    nodes = local_train_nids
    for _ in range(num_hops):
        local_subgraph = dgl.in_subgraph(graph, nodes)
        nodes = local_subgraph.nodes()
    return local_subgraph


def preprocess(graph, fan_out, num_gpus):

    start = time.time()

    train_nids = torch.nonzero(graph.ndata['train_mask']).squeeze(1)
    train_nids = train_nids[torch.randperm(train_nids.shape[0])]
    num_train_nids_per_gpu = (train_nids.shape[0] + num_gpus - 1) // num_gpus

    avg_feature_size = graph.ndata["features"].numel(
    ) * graph.ndata["features"].element_size() / graph.num_nodes()

    local_train_nids_list = []
    feature_cache_nids_list = []

    for device in range(num_gpus):
        local_train_nids = train_nids[device *
                                      num_train_nids_per_gpu:(device + 1) *
                                      num_train_nids_per_gpu]
        local_part = get_local_subgraph(graph, local_train_nids, len(fan_out))
        out_degrees = local_part.out_degrees()
        local_nids = local_part.nodes()
        sorted_local_nids = local_nids[torch.argsort(out_degrees)]

        available_mem = get_available_memory(device, 8 * 1024 * 1024 * 1024)
        print("GPU {}, available memory size = {:.3f} GB".format(
            device, available_mem / 1024 / 1024 / 1024))

        cached_nids_num = min(int(available_mem / avg_feature_size),
                              sorted_local_nids.numel())
        cached_nids = sorted_local_nids[:cached_nids_num]

        local_train_nids_list.append(local_train_nids.numpy())
        feature_cache_nids_list.append(cached_nids.numpy())

    end = time.time()
    print("Preprocess time: {:.3f} seconds".format(end - start))

    return local_train_nids_list, feature_cache_nids_list


def run(rank, world_size, data, args):
    graph, num_classes, train_nids_list, feature_cache_nids_list = data

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)
    torch.cuda.reset_peak_memory_stats()

    # create model
    model = SAGE(graph.ndata['features'].shape[1], 256, num_classes)
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[rank],
                                                output_device=rank)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # partition train nids, create dataloader
    train_nids = torch.from_numpy(train_nids_list[rank])
    train_dataloader = SeedGenerator(train_nids.cuda(),
                                     args.batch_size,
                                     shuffle=True)

    # cache data
    indptr = graph.adj_sparse("csc")[0]
    torch.ops.dgs_ops._CAPI_tensor_pin_memory(indptr)
    indices = graph.adj_sparse("csc")[1]
    torch.ops.dgs_ops._CAPI_tensor_pin_memory(indices)
    if args.bias:
        probs = graph.edata.pop("probs")
        torch.ops.dgs_ops._CAPI_tensor_pin_memory(probs)
    features = graph.ndata.pop("features")
    torch.ops.dgs_ops._CAPI_tensor_pin_memory(features)
    labels = graph.ndata.pop("labels")
    torch.ops.dgs_ops._CAPI_tensor_pin_memory(labels)
    num_nodes = graph.num_nodes()
    del graph

    feature_cache_nids = torch.from_numpy(feature_cache_nids_list[rank])

    print("Rank {}, cache features...".format(rank))
    feature_server = FeatureCacheServer(features, device_id=rank)
    feature_server.cache_data(feature_cache_nids.cuda(),
                              feature_cache_nids.numel() >= num_nodes)

    print("Rank {}, cache structures...".format(rank))
    if args.bias:
        structure_server = StructureCacheServer(indptr,
                                                indices,
                                                probs,
                                                device_id=rank)
    else:
        structure_server = StructureCacheServer(indptr,
                                                indices,
                                                device_id=rank)
    structure_server.cache_data(torch.tensor([]), False)
    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]

    dist.barrier()

    if rank == 0:
        print('start training...')
    iteration_time_log = []
    sampling_time_log = []
    loading_time_log = []
    training_time_log = []
    epoch_iterations_log = []
    epoch_time_log = []
    for epoch in range(args.num_epochs):
        model.train()

        epoch_start = time.time()
        for it, seed_nids in enumerate(train_dataloader):
            torch.cuda.synchronize()
            sampling_start = time.time()
            frontier, seeds, blocks = structure_server.sample_neighbors(
                seed_nids, fan_out)
            blocks = [block.to(rank) for block in blocks]
            torch.cuda.synchronize()
            sampling_end = time.time()

            loading_start = time.time()
            batch_inputs = feature_server.fetch_data(frontier).cuda()
            batch_labels = labels.index_select(0, seeds.cpu()).cuda()
            torch.cuda.synchronize()
            loading_end = time.time()

            training_start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = F.cross_entropy(batch_pred, batch_labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            training_end = time.time()

            sampling_time_log.append(sampling_end - sampling_start)
            loading_time_log.append(loading_end - loading_start)
            training_time_log.append(training_end - training_start)
            iteration_time_log.append(training_end - sampling_start)

        torch.cuda.synchronize()
        epoch_end = time.time()
        epoch_iterations_log.append(it)
        epoch_time_log.append(epoch_end - epoch_start)

    print(
        "Rank {} | Sampling {:.3f} ms | Loading {:.3f} ms | Training {:.3f} ms | Iteration {:.3f} ms | Epoch iterations num {} | Epoch time {:.3f} ms"
        .format(rank,
                np.mean(sampling_time_log[3:]) * 1000,
                np.mean(loading_time_log[3:]) * 1000,
                np.mean(training_time_log[3:]) * 1000,
                np.mean(iteration_time_log[3:]) * 1000,
                np.mean(epoch_iterations_log),
                np.mean(epoch_time_log) * 1000))

    torch.ops.dgs_ops._CAPI_tensor_unpin_memory(indptr)
    torch.ops.dgs_ops._CAPI_tensor_unpin_memory(indices)
    if args.bias:
        torch.ops.dgs_ops._CAPI_tensor_unpin_memory(probs)
    torch.ops.dgs_ops._CAPI_tensor_unpin_memory(features)
    torch.ops.dgs_ops._CAPI_tensor_unpin_memory(labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpu',
                        default='8',
                        type=int,
                        help='The number GPU participated in the training.')
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument('--root',
                        default='dataset/',
                        help='Path of the dataset.')
    parser.add_argument("--batch-size",
                        default="1000",
                        type=int,
                        help="The number of seeds of sampling.")
    parser.add_argument('--fan-out', type=str, default='15,15,15')
    parser.add_argument('--bias',
                        action='store_true',
                        default=False,
                        help="Sample with bias.")
    parser.add_argument(
        "--dataset",
        default="ogbn-papers100M",
        choices=["ogbn-products", "ogbn-papers100M", "ogbn-papers400M"])
    args = parser.parse_args()
    print(args)

    if args.dataset == "ogbn-products":
        graph, num_classes = load_ogb("ogbn-products", root=args.root)
    elif args.dataset == "ogbn-papers100M":
        graph, num_classes = load_ogb("ogbn-papers100M", root=args.root)
    elif args.dataset == "ogbn-papers400M":
        graph, num_classes = load_papers400m_sparse(root=args.root)

    graph = graph.formats('csc')
    graph.create_formats_()
    graph.edata.clear()

    n_procs = min(args.num_gpu, torch.cuda.device_count())

    if args.bias:
        graph.edata['probs'] = torch.randn((graph.num_edges(), )).abs().float()

    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]

    train_nids_list, feature_cache_nids_list = preprocess(
        graph, fan_out, n_procs)

    data = graph, num_classes, train_nids_list, feature_cache_nids_list

    import torch.multiprocessing as mp
    mp.spawn(run, args=(n_procs, data, args), nprocs=n_procs)
