import torch
import torch.distributed as dist
import time


class GraphCacheServer:

    def __init__(self, nfeats, gpuid=0):
        self.nfeats = nfeats
        self.feat_dim = self.nfeats.size(1)
        self.gpuid = gpuid
        self.hash_key = None
        self.hash_val = None
        self.reordered = False
        self.gpu_cached_data = None

    def get_cache_nid(self, graph, capability):
        type_size = self.nfeats.element_size()
        if capability >= self.nfeats.numel() * type_size:
            if self.gpuid == 0:
                print("[Pagraph] cache rate = 1.00")
            reorder = False
            return torch.arange(graph.num_nodes()).cuda(self.gpuid), reorder

        else:
            start = time.time()
            if '_P' in graph.ndata and True:
                sort_nids = torch.argsort(graph.ndata['_P'], descending=True)
            elif 'out_degrees' in graph.ndata:
                sort_nids = torch.argsort(graph.ndata['out_degrees'],
                                          descending=True)
            else:
                if 'csr' not in graph.formats(
                )['created'] and 'coo' not in graph.formats()['created']:
                    csr_graph = graph.formats('csr')
                    csr_graph.create_formats_()
                out_degrees = csr_graph.out_degrees()
                sort_nids = torch.argsort(out_degrees, descending=True)
            cache_node_num = capability // (self.feat_dim * type_size)
            cache_nids = sort_nids[:cache_node_num]
            end = time.time()

            if self.gpuid == 0:
                print("[Pagraph] sorting nodes takes {:.3f} s".format(end -
                                                                      start))
                print("[Pagraph] cache rate = {:.2f}".format(
                    capability / (self.nfeats.numel() * type_size)))

            reorder = True
            return cache_nids.cuda(self.gpuid), reorder

    def cache_data(self, cache_nids, reorder=True):
        self.reordered = reorder
        if self.reordered:

            torch.cuda.synchronize()
            start = time.time()
            cache_node_num = cache_nids.numel()
            nid_in_gpu = torch.arange(cache_node_num).cuda(self.gpuid)
            self.hash_key, self.hash_val = torch.ops.dgs_ops._CAPI_create_hash_map(
                cache_nids, nid_in_gpu)
            self.hash_key.requires_grad_(False)
            self.hash_val.requires_grad_(False)
            torch.cuda.synchronize()
            end = time.time()
            print("[Pagraph] gpu {} creating hash map takes {:.3f} s".format(
                self.gpuid, end - start))

            self.gpu_cached_data = self.nfeats[cache_nids].cuda(self.gpuid)
            self.nfeats = self.nfeats.pin_memory()
            print("[Pagraph] gpu {} cache {} MB data".format(
                self.gpuid,
                self.gpu_cached_data.numel() *
                self.gpu_cached_data.element_size() / 1024 / 1024))
        else:
            self.gpu_cached_data = self.nfeats.cuda(self.gpuid)
            print("[Pagraph] gpu {} cache {} MB data".format(
                self.gpuid,
                self.nfeats.numel() * self.nfeats.element_size() / 1024 /
                1024))

    def fetch_data(self, nids):
        if not self.reordered:
            return torch.index_select(self.gpu_cached_data, 0,
                                      nids.cuda(self.gpuid))
        else:
            return torch.ops.dgs_ops._CAPI_fetch_data(self.nfeats,
                                                      self.gpu_cached_data,
                                                      nids.cuda(self.gpuid),
                                                      self.hash_key,
                                                      self.hash_val)
