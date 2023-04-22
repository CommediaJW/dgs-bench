import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import time
import numpy as np
from utils.models import SAGE
from GraphCache.cache import get_available_memory
from GraphCache.dataloading import SeedGenerator, load_dataset
from utils.chunktensor_sampler import *

torch.ops.load_library("../Dist-GPU-sampling/build/libdgs.so")


def print_memory():
    print("max_memory_allocated: {:.2f} GB, max_memory_reserved {:.2f} GB".
          format(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
                 torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024))
    print("memory_allocated {:.2f} GB, memory_reserved {:.2f} GB".format(
        torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
        torch.cuda.memory_reserved() / 1024 / 1024 / 1024))


def run(rank, world_size, data, args):
    graph, num_classes, train_nids_list = data

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)

    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]
    train_nids = torch.from_numpy(train_nids_list[rank])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print_memory()

    # create model
    model = SAGE(graph["features"].shape[1], 256, num_classes)
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[rank],
                                                output_device=rank)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # create dataloader
    train_dataloader = SeedGenerator(train_nids.cuda(),
                                     args.batch_size,
                                     shuffle=True)

    available_mem = get_available_memory(rank, 7 * 1024 * 1024 * 1024)
    print("GPU {}, available memory size = {:.3f} GB".format(
        rank, available_mem / 1024 / 1024 / 1024))

    create_dgs_communicator(world_size)

    # chunktensor cache
    if dist.get_rank() == 0:
        print("create chunk features")
    chunk_features = create_chunktensor(graph["features"], world_size,
                                        available_mem, args.feat_cache_rate)

    if args.bias:
        if dist.get_rank() == 0:
            print("create chunk probs")
        chunk_probs = create_chunktensor(
            graph["probs"], world_size,
            max(
                available_mem -
                torch.ops.dgs_ops._CAPI_get_current_allocated(), 0),
            args.graph_cache_rate)

    if dist.get_rank() == 0:
        print("create chunk indices")
    chunk_indices = create_chunktensor(
        graph["indices"], world_size,
        max(available_mem - torch.ops.dgs_ops._CAPI_get_current_allocated(),
            0), args.graph_cache_rate)

    if dist.get_rank() == 0:
        print("create chunk indptr")
    chunk_indptr = create_chunktensor(
        graph["indptr"], world_size,
        max(available_mem - torch.ops.dgs_ops._CAPI_get_current_allocated(),
            0), args.graph_cache_rate)

    if args.bias:
        sampler = ChunkTensorSampler(fan_out,
                                     chunk_indptr,
                                     chunk_indices,
                                     chunk_probs=chunk_probs)
    else:
        sampler = ChunkTensorSampler(fan_out, chunk_indptr, chunk_indices)

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
            frontier, seeds, blocks = sampler.sample_blocks(seed_nids)
            blocks = [block.to(rank) for block in blocks]
            torch.cuda.synchronize()
            sampling_end = time.time()

            loading_start = time.time()
            batch_inputs = chunk_features._CAPI_index(frontier).cuda()
            batch_labels = torch.ops.dgs_ops._CAPI_index(
                graph["labels"], seeds)
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
    parser.add_argument('--fan-out', type=str, default='10,10,10')
    parser.add_argument('--bias',
                        action='store_true',
                        default=False,
                        help="Sample with bias.")
    parser.add_argument("--dataset",
                        default="ogbn-papers100M",
                        choices=["ogbn-products", "ogbn-papers100M"])
    parser.add_argument(
        '--feat-cache-rate',
        default='1',
        type=float,
        help=
        'The gpu cache rate of features. If the gpu memory is not enough, cache priority: features > probs > indices > indptr'
    )
    parser.add_argument(
        '--graph-cache-rate',
        default='1',
        type=float,
        help=
        'The gpu cache rate of graph structure tensors. If the gpu memory is not enough, cache priority: features > probs > indices > indptr'
    )
    args = parser.parse_args()
    torch.manual_seed(1)

    if args.dataset == "ogbn-products":
        graph, num_classes = load_dataset(args.root, "ogbn-products")
    elif args.dataset == "ogbn-papers100M":
        graph, num_classes = load_dataset(args.root, "ogbn-papers100M")

    n_procs = min(args.num_gpu, torch.cuda.device_count())
    args.num_gpu = n_procs
    print(args)

    if args.bias:
        graph["probs"] = torch.randn(
            (graph["indices"].shape[0], )).abs().float()

    # partition train nodes
    train_nids = graph.pop("train_idx")
    train_nids = torch.cat([
        torch.randint(0, graph["indptr"].numel() - 1,
                      (graph["indptr"].numel() // 10, )).long(), train_nids
    ]).unique()

    train_nids = train_nids[torch.randperm(train_nids.shape[0])]
    num_train_nids_per_gpu = (train_nids.shape[0] + n_procs - 1) // n_procs
    print("#train nodes {} | #train nodes per gpu {}".format(
        train_nids.shape[0], num_train_nids_per_gpu))
    train_nids_list = []
    for device in range(n_procs):
        local_train_nids = train_nids[device *
                                      num_train_nids_per_gpu:(device + 1) *
                                      num_train_nids_per_gpu]
        train_nids_list.append(local_train_nids.numpy())

    data = graph, num_classes, train_nids_list

    #graph["labels"] = torch.maximum(
    #    graph["labels"],
    #    torch.tensor([num_classes - 1], dtype=graph["labels"].dtype))
    #graph["labels"] = torch.minimum(
    #    graph["labels"], torch.tensor([0], dtype=graph["labels"].dtype))

    index = ~torch.isnan(graph["labels"])
    print(index)
    valid_label = graph["labels"][index]
    print(valid_label)

    graph["labels"][:] = 0
    graph["labels"][index] = valid_label

    print(graph["labels"].max(), graph["labels"].min())

    print(graph["features"].shape)
    print(graph["features"].stride())

    import torch.multiprocessing as mp
    mp.spawn(run, args=(n_procs, data, args), nprocs=n_procs)
