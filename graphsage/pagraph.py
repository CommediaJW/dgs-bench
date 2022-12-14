import torch


class GraphCacheServer:

    def __init__(self, nfeats, node_num, gpuid=0):
        self.nfeats = nfeats
        self.total_dim = self.nfeats.size(1)
        self.gpuid = gpuid
        self.node_num = node_num
        self.hash_key = None
        self.hash_val = None
        self.full_cached = False
        self.gpu_fix_cache = None

    def auto_cache(self, dgl_g, cache_rate=1.0):

        # Step1: get available GPU memory
        peak_allocated_mem = torch.cuda.max_memory_allocated(device=self.gpuid)
        peak_cached_mem = torch.cuda.max_memory_reserved(device=self.gpuid)
        total_mem = torch.cuda.get_device_properties(self.gpuid).total_memory
        available = total_mem - peak_allocated_mem - 0.3 * peak_cached_mem \
            - 1024 * 1024 * 1024 - self.node_num  # in bytes

        # Stpe2: get capability
        csize = self.nfeats[0][0].element_size()
        capability = max(0, int(0.8 * available / (self.total_dim * csize)))
        if cache_rate != 1.0:
            capability = min(capability, int(self.node_num * cache_rate))

        # Step3: cache
        if capability >= self.node_num:
            # fully cache
            print(
                "GPU {} | Pagraph fully cache the feature data, size = {} MB".
                format(self.gpuid,
                       self.node_num * self.total_dim * csize / 1024 / 1024))
            full_nids = torch.arange(self.node_num).cuda(self.gpuid)
            self.cache_fix_data(full_nids, self.nfeats, is_full=True)

        else:
            # choose top-cap out-degree nodes to cache
            if "_P" in dgl_g.ndata and True:
                sort_nid = torch.argsort(dgl_g.ndata["_P"], descending=True)
            elif "out_degrees" in dgl_g.ndata and True:
                sort_nid = torch.argsort(dgl_g.ndata["out_degrees"],
                                         descending=True)
            else:
                out_degrees = dgl_g.out_degrees()
                sort_nid = torch.argsort(out_degrees, descending=True)

            print(
                "GPU {} | Pagraph cache part of the feature data, size = {} MB, cache rate = {:.2f}"
                .format(self.gpuid,
                        capability * self.total_dim * csize / 1024 / 1024,
                        capability / self.node_num))

            cache_nid = sort_nid[:capability].cuda(self.gpuid)
            data = self.nfeats[cache_nid]
            self.cache_fix_data(cache_nid, data, is_full=False)

    def cache_fix_data(self, nids, data, is_full=False):
        self.full_cached = is_full
        if not is_full:
            rows = nids.numel()
            nids_in_gpu = torch.arange(rows).cuda(self.gpuid)
            self.hash_key, self.hash_val = torch.ops.dgs_ops._CAPI_create_hash_map(
                nids, nids_in_gpu)
            self.hash_key.requires_grad_(False)
            self.hash_val.requires_grad_(False)
            self.nfeats = self.nfeats.pin_memory()
            self.gpu_fix_cache = data.cuda(self.gpuid)
        else:
            self.gpu_fix_cache = data.cuda(self.gpuid)

    def fetch_data(self, nids):
        if self.full_cached:
            return torch.index_select(self.gpu_fix_cache, 0,
                                      nids.to(self.gpuid))
        else:
            return torch.ops.dgs_ops._CAPI_fetch_data(self.nfeats,
                                                      self.gpu_fix_cache,
                                                      nids.to(self.gpuid),
                                                      self.hash_key,
                                                      self.hash_val)
