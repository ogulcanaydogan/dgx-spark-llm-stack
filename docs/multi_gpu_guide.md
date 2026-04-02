# Multi-GPU Guide (DGX Spark Cluster with NVLink)

This guide closes the roadmap item with a reproducible preflight on Spark and a practical multi-node launch recipe.

## Scope

- This iteration validates **distributed readiness** and command flow.
- It does **not** claim production multi-node benchmark numbers.
- Success criterion: Spark preflight script passes and emits artifact JSON.

## Cluster Prerequisites

- Minimum 2 nodes for real multi-node runs.
- Nodes can reach each other over the selected network interface.
- Matching CUDA + PyTorch major/minor versions across nodes.
- NCCL available in the runtime environment.
- SSH access to each node for launch orchestration.

## Spark Preflight (Single Command)

Run:

```bash
./scripts/smoke_multi_gpu_preflight.sh
```

Expected:

- Script exits `0`
- Log contains `torchrun single-rank NCCL smoke PASS`
- Log contains `PASS: multi-gpu preflight completed`

Generated artifact:

- `artifacts/benchmarks/multi-gpu-preflight-<date>.json`

Required JSON fields:

- `hostname`
- `torch_version`
- `cuda_version`
- `cuda_device_count`
- `nccl_available`
- `torchrun_single_rank_nccl_ok`
- `timestamp_utc`

## Multi-Node `torchrun` Recipe (2 Nodes)

Node 0:

```bash
MASTER_ADDR=10.0.0.10
MASTER_PORT=29500
NNODES=2
NPROC_PER_NODE=1
NODE_RANK=0

NCCL_DEBUG=INFO \
NCCL_SOCKET_IFNAME=eth0 \
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train.py
```

Node 1:

```bash
MASTER_ADDR=10.0.0.10
MASTER_PORT=29500
NNODES=2
NPROC_PER_NODE=1
NODE_RANK=1

NCCL_DEBUG=INFO \
NCCL_SOCKET_IFNAME=eth0 \
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train.py
```

Optional fallback when IB path is unstable:

```bash
export NCCL_IB_DISABLE=1
```

## NVLink Topology Notes

Check topology:

```bash
nvidia-smi topo -m
```

Interpretation:

- `NV#` links between GPUs indicate direct NVLink class connectivity.
- `PHB/PIX/SYS` paths indicate PCIe/system routing and usually higher communication cost.

On single-GPU Spark setups, NVLink matrix may be limited; this guide still validates distributed software path via NCCL preflight.

## Troubleshooting

1. Rendezvous timeout
- Verify `MASTER_ADDR`, `MASTER_PORT`, `--node_rank` and firewall rules.

2. NCCL init failure
- Set `NCCL_DEBUG=INFO` and check interface selection with `NCCL_SOCKET_IFNAME`.
- Try `NCCL_IB_DISABLE=1` for interface isolation.

3. Rank mismatch
- Ensure all nodes use identical `--nnodes` and unique `--node_rank`.

4. `torchrun` not found
- Ensure user-local binaries are on PATH (for example `$HOME/.local/bin`).
