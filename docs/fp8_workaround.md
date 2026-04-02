# FP8 Workaround on DGX Spark (Blackwell `sm_121`)

TransformerEngine FP8 (MXFP8 path) is not currently usable on GB10/Blackwell in DGX Spark.
Use BF16 as the stable fallback for training.

## Why FP8 Fails Here

On `sm_121`, TransformerEngine's MXFP8 path fails with an unsupported-architecture assertion.
Typical error signature:

```text
MXFP8 ... not supported on 12.0+ architectures yet.
```

## One-Command Validation (Fail + Pass)

Run this smoke on Spark:

```bash
./scripts/smoke_fp8_workaround.sh
```

The script runs inside pinned NGC image `nvcr.io/nvidia/pytorch:25.11-py3` with GPU flags:

- FP8 test (`MXFP8BlockScaling + fp8_autocast`) must fail with unsupported message.
- BF16 fallback test must pass on the same model.

## Expected Output Signatures

```text
mxfp8_result fail
mxfp8_error ... not supported ...
bf16_workaround ok (4, 256) torch.bfloat16
[fp8-smoke] PASS: FP8 workaround validated on DGX Spark
```

## Recommended Training Fallback

- Keep precision at BF16 (`bf16=True` in training args).
- Do not enable TransformerEngine FP8/MXFP8 on DGX Spark.
- Keep attention path on SDPA where flash-attention is unsupported.

Minimal guidance:

```python
training_args = TrainingArguments(
    bf16=True,
    fp16=False,
)
```

```bash
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
```

## Notes

- This is a practical workaround, not a permanent fix.
- Re-run the smoke after major updates to CUDA, PyTorch, or TransformerEngine.
