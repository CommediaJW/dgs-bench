from dgl.dataloading.base import BlockSampler
import torch
import dgl
import torch.distributed as dist


class ChunkTensorSampler(BlockSampler):

    def __init__(self,
                 fanouts,
                 g,
                 feat_mem_used=0,
                 model_mem_used=0,
                 avaliable_mem=None,
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
        self.fanouts = fanouts
        self.prob = prob
        self.replace = replace
        self.available_mem = avaliable_mem

        indptr = g.adj_sparse("csc")[0]
        indices = g.adj_sparse("csc")[1]
        if self.prob:
            probs = g.edata[self.prob]

        if self.available_mem is None:
            device = torch.cuda.current_device()
            max_device_mem = torch.cuda.get_device_properties(
                device).total_memory
            _available_mem = torch.tensor(
                [(max_device_mem - model_mem_used * 2 - feat_mem_used)]).long().cuda()
            print(_available_mem)

            print(max_device_mem - model_mem_used * 2 - feat_mem_used)

            dist.all_reduce(_available_mem, dist.ReduceOp.MIN)
            self.available_mem = _available_mem.cpu().numpy()[0]

        print("Available mem for ChunkTensor {} MB",
              self.available_mem / 1024 / 1024)

        comm_size = dist.get_world_size()

        indptr_cached_size = min(indptr.numel() * indptr.element_size() // comm_size,
                                 self.available_mem)
        self.chunk_indptr = torch.classes.dgs_classes.ChunkTensor(
            indptr, indptr_cached_size)
        self.available_mem = self.available_mem - indptr_cached_size
        print("Cache indptr per GPU {} MB".format(
            indptr_cached_size / 1024 / 1024))

        if self.prob:
            _need_mem = (indices.numel() * indices.element_size() +
                         probs.numel() * probs.element_size()) // comm_size
            _cached_mem = min(_need_mem, self.available_mem)

            indices_cached_size = int(_cached_mem / (
                indices.element_size() + probs.element_size()) * indices.element_size())
            probs_cached_size = int(_cached_mem / (
                indices.element_size() + probs.element_size()) * probs.element_size())

            self.chunk_indices = torch.classes.dgs_classes.ChunkTensor(
                indices, indices_cached_size)
            print("Cache indices per GPU {} MB".format(
                indices_cached_size / 1024 / 1024))

            self.chunk_probs = torch.classes.dgs_classes.ChunkTensor(
                probs, probs_cached_size)
            print("Cache probs per GPU {} MB".format(
                indices_cached_size / 1024 / 1024))

            self.available_mem = self.available_mem - \
                indices_cached_size - probs_cached_size

        else:
            indices_cached_size = min(indices.numel() *
                                      indices.element_size() // comm_size, self.available_mem)
            self.chunk_indices = torch.classes.dgs_classes.ChunkTensor(
                indices, indices_cached_size)
            self.available_mem = self.available_mem - indices_cached_size
            print("Cache indices per GPU {} MB".format(
                indices_cached_size / 1024 / 1024))

    def __del__(self):
        del self.chunk_indices
        del self.chunk_indptr
        if (self.prob):
            del self.chunk_probs

    def get_available_mem(self):
        return self.available_mem

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
