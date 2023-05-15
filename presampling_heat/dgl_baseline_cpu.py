import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torchmetrics.functional as MF
import time
import dgl
import numpy as np
from utils.models import SAGE


def load_ogb(name, root="dataset"):
    from ogb.nodeproppred import DglNodePropPredDataset

    print('load', name)
    data = DglNodePropPredDataset(name=name, root=root)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['features'] = graph.ndata.pop('feat')
    graph.ndata['labels'] = labels

    if name == "ogbn-papers100M":
        num_labels = 172
    else:
        num_labels = len(
            torch.unique(labels[torch.logical_not(torch.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx[
        'valid'], splitted_idx['test']
    train_mask = torch.zeros((graph.number_of_nodes(), ), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((graph.number_of_nodes(), ), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((graph.number_of_nodes(), ), dtype=torch.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    print('finish constructing', name)
    return graph, num_labels


def evaluate(model, valid_dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(valid_dataloader):
        with torch.no_grad():
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']
            batch_pred = model(blocks, batch_inputs)
            ys.append(batch_labels)
            y_hats.append(batch_pred)
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def run(rank, world_size, data, args):
    graph, num_classes = data

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=world_size,
                            rank=rank)

    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]
    train_nids = torch.nonzero(graph.ndata['train_mask']).squeeze(1)

    if rank == 0:
        print("create model...")
    # create model
    model = SAGE(graph.ndata["features"].shape[1], 256, num_classes)
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[rank],
                                                output_device=rank)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    if rank == 0:
        print("create sampler...")
    # create sampler
    if args.bias:
        # bias sampling
        sampler = dgl.dataloading.NeighborSampler(fan_out, prob='probs')
    else:
        # uniform sampling
        sampler = dgl.dataloading.NeighborSampler(fan_out)

    if rank == 0:
        print("create dataloader...")
    # create dataloader
    train_dataloader = dgl.dataloading.DataLoader(graph,
                                                  train_nids,
                                                  sampler,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  use_ddp=True)

    # valid
    if args.valid and rank == 0:
        valid_nids = torch.nonzero(graph.ndata['val_mask']).squeeze(1).cuda()
        valid_dataloader = dgl.dataloading.DataLoader(
            graph,
            valid_nids,
            sampler,
            device='cuda',
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            use_uva=True)

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
        sampling_start = time.time()
        for it, (input_nodes, output_nodes,
                 blocks) in enumerate(train_dataloader):
            blocks = [block.to(rank) for block in blocks]
            torch.cuda.synchronize()
            sampling_end = time.time()

            loading_start = time.time()
            batch_inputs = torch.index_select(graph.ndata["features"], 0,
                                              input_nodes).cuda()
            batch_labels = torch.index_select(graph.ndata["labels"], 0,
                                              output_nodes).cuda()
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

            sampling_start = time.time()

        torch.cuda.synchronize()
        epoch_end = time.time()
        epoch_iterations_log.append(it)
        epoch_time_log.append(epoch_end - epoch_start)

        if args.valid and rank == 0:
            acc = evaluate(model, valid_dataloader, num_classes)
            print("Epoch {}, valid acc = {:.3f}".format(epoch, acc))
        dist.barrier()

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
    parser.add_argument('--valid', action='store_true', default=False)
    args = parser.parse_args()
    torch.manual_seed(1)

    n_procs = min(args.num_gpu, torch.cuda.device_count())
    args.num_gpu = n_procs
    print(args)

    if args.dataset == "ogbn-products":
        graph, num_classes = load_ogb("ogbn-products", root=args.root)
    elif args.dataset == "ogbn-papers100M":
        graph, num_classes = load_ogb("ogbn-papers100M", root=args.root)
    graph = graph.formats('csc')
    graph.create_formats_()
    graph.edata.clear()

    print("create CSC formats...")
    eid = graph.adj_sparse('csc')[2]
    if args.bias:
        probs = torch.randn((graph.num_edges(), )).abs().float()
        graph.edata["probs"] = probs[torch.argsort(eid)]

    data = graph, num_classes

    import torch.multiprocessing as mp
    mp.spawn(run, args=(n_procs, data, args), nprocs=n_procs)
