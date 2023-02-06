# Single node training

## Dependencies

- CUDA 11.3

- PyTorch 1.12.1

- DGL 0.9.1

- NumPy 1.23.5

- OGB 1.3.5

- NCCL 2.10.3

## Dataset

- ogbn-products

- ogbn-papers100M

- ogbn-papers400M

## How to run

### Baseline

```bash
python3 baseline.py \
--num-gpu 8 \
--dataset ogbn-papers400M \
--root dataset/ \
--model graphsage \
--batch-size 1000 \
--bias
```

args:

- `--num-gpu` The number of GPUs.

- `--dataset` The dataset, support `ogbn-products`, `ogbn-papers100M` and `ogbn-papers400M`

- `--root` The directory which stores the dataset.

- `--model` The model of training, support `graphsage` and `gat`.

- `--batch-size` The number of seeds in each iteration.

- `--bias` Sample neighbors with bias.

### Chunktensor version

```bash
python3 chunktensor.py \
--num-gpu 8 \
--dataset ogbn-papers400M \
--root dataset/ \
--model graphsage \
--batch-size 1000 \
--libdgs ../Dist-GPU-sampling/build/libdgs.so \
--cache-rate 0.4 \
--bias
```

args:

- `--libdgs` The directory of `libdgs.so`.

- `--cache-rate`

  The cache rate of features and graph structure tensors. If gpu memory is not enough, cache priority: features > probs > indices > indptr.

  Note: this is the cache rate of all the gpus. For example, if `--num-gpu 2` and `--cache-rate 0.4` are set, the cache rate of each gpu is `0.2`.
