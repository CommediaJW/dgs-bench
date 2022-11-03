from dgl.dataloading.base import BlockSampler
import torch
import dgl
import torch.distributed as dist


def get_available_memory(device, model_mem_used, num_node):
    available_mem = torch.cuda.mem_get_info(
        device)[1] - model_mem_used - torch.cuda.max_memory_allocated(
        ) - 0.3 * torch.cuda.max_memory_reserved(
        ) - 2 * 1024 * 1024 * 1024 - num_node
    available_mem = max(available_mem, 16)
    available_mem = torch.tensor([available_mem]).long().cuda()
    dist.all_reduce(available_mem, dist.ReduceOp.MIN)
    return available_mem.cpu().numpy()[0]


class ChunkTensorSampler(BlockSampler):

    def __init__(self,
                 fanouts,
                 g,
                 model_mem_used=0,
                 cache_rate=0.0,
                 prob=None,
                 replace=False,
                 prefetch_node_feats=None,
                 prefetch_labels=None,
                 prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        torch.cuda.reset_peak_memory_stats()
        self.fanouts = fanouts
        self.prob = prob
        self.replace = replace
        self.device = torch.cuda.current_device()
        self.model_mem_used = model_mem_used

        indptr = g.adj_sparse("csc")[0]
        indices = g.adj_sparse("csc")[1]
        if self.prob:
            probs = g.edata[self.prob]

        self.num_node = g.num_nodes()

        comm_size = dist.get_world_size()

        indptr_cached_size = min(
            indptr.numel() * indptr.element_size() // comm_size,
            get_available_memory(self.device, self.model_mem_used,
                                 self.num_node),
            indptr.numel() * indptr.element_size() * cache_rate)
        indptr_cached_size = int(max(indptr_cached_size, 16))
        self.chunk_indptr = torch.classes.dgs_classes.ChunkTensor(
            indptr, indptr_cached_size)
        if dist.get_rank() == 0:
            print("Cache indptr per GPU {} MB, cache_rate = {:.2f}".format(
                indptr_cached_size / 1024 / 1024,
                indptr_cached_size / (indptr.element_size() * indptr.numel())))

        if self.prob:
            _need_mem = (indices.numel() * indices.element_size() +
                         probs.numel() * probs.element_size()) // comm_size
            _cached_mem = min(
                _need_mem,
                get_available_memory(self.device, self.model_mem_used,
                                     self.num_node))

            indices_cached_size = int(
                min(
                    _cached_mem /
                    (indices.element_size() + probs.element_size()) *
                    indices.element_size(),
                    indices.numel() * indices.element_size() * cache_rate))
            indices_cached_size = int(max(indices_cached_size, 16))
            probs_cached_size = int(
                min(
                    _cached_mem /
                    (indices.element_size() + probs.element_size()) *
                    probs.element_size(),
                    probs.numel() * probs.element_size() * cache_rate))
            probs_cached_size = int(max(probs_cached_size, 16))
            self.chunk_indices = torch.classes.dgs_classes.ChunkTensor(
                indices, indices_cached_size)
            if dist.get_rank() == 0:
                print(
                    "Cache indices per GPU {} MB, cache rate = {:.2f}".format(
                        indices_cached_size / 1024 / 1024,
                        indices_cached_size /
                        (indices.element_size() * indices.numel())))

            self.chunk_probs = torch.classes.dgs_classes.ChunkTensor(
                probs, probs_cached_size)
            if dist.get_rank() == 0:
                print("Cache probs per GPU {} MB, cache_rate = {:.2f}".format(
                    probs_cached_size / 1024 / 1024, probs_cached_size /
                    (probs.element_size() * probs.numel())))

        else:
            indices_cached_size = min(
                indices.numel() * indices.element_size() // comm_size,
                get_available_memory(self.device, self.model_mem_used,
                                     self.num_node),
                indices.numel() * indices.element_size() * cache_rate)
            indices_cached_size = int(max(indices_cached_size, 16))
            self.chunk_indices = torch.classes.dgs_classes.ChunkTensor(
                indices, indices_cached_size)
            if dist.get_rank() == 0:
                print(
                    "Cache indices per GPU {} MB, cache rate = {:.2f}".format(
                        indices_cached_size / 1024 / 1024,
                        indices_cached_size /
                        (indices.element_size() * indices.numel())))

    def __del__(self):
        del self.chunk_indices
        del self.chunk_indptr
        if (self.prob):
            del self.chunk_probs

    def get_available_mem(self):
        return get_available_memory(self.device, self.model_mem_used,
                                    self.num_node)

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        seeds = seed_nodes
        blocks = []
        for fan_out in reversed(self.fanouts):
            if (self.prob):
                coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_probs_with_chunk_tensor(
                    seeds, self.chunk_indptr, self.chunk_indices,
                    self.chunk_probs, fan_out, self.replace)
            else:
                coo_row, coo_col = torch.ops.dgs_ops._CAPI_sample_neighbors_with_chunk_tensor(
                    seeds, self.chunk_indptr, self.chunk_indices, fan_out,
                    self.replace)

            frontier, (coo_row,
                       coo_col) = torch.ops.dgs_ops._CAPI_tensor_relabel(
                           [seeds, coo_col], [coo_row, coo_col])
            block = dgl.create_block((coo_col, coo_row),
                                     num_src_nodes=frontier.numel(),
                                     num_dst_nodes=seeds.numel())
            block.srcdata[dgl.NID] = frontier
            block.dstdata[dgl.NID] = seeds
            blocks.insert(0, block)

            seeds = frontier

        return frontier, seed_nodes, blocks
