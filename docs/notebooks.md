# Example Notebooks (DGX Spark)

This guide provides three minimal notebooks for inference, fine-tuning, and evaluation on DGX Spark.

## Included Notebooks

- `notebooks/01_inference.ipynb` — short inference benchmark smoke run.
- `notebooks/02_finetuning_lora.ipynb` — short LoRA training throughput smoke run.
- `notebooks/03_evaluation_perplexity.ipynb` — short perplexity evaluation smoke run.

Default model profile in all notebooks: `Qwen/Qwen2.5-0.5B-Instruct`.

## Spark Preflight

```bash
python3 --version
nvidia-smi
```

If `jupyter` is missing, install in user space:

```bash
python3 -m pip install --user notebook nbconvert ipykernel
python3 -m ipykernel install --user --name python3 --display-name "Python 3"
```

## Run One Notebook

From repo root:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_inference.ipynb \
  --output 01_inference.executed.ipynb --output-dir artifacts/notebooks
```

Repeat for the other notebooks by changing the file name.

## Run Full Smoke (Recommended)

```bash
./scripts/smoke_notebooks.sh
```

What the script does:

- Executes all three notebooks in order with `nbconvert --execute`.
- Writes per-notebook logs to `artifacts/notebooks/logs/`.
- Validates JSON outputs and required fields.
- Exits non-zero on first failure with a short log tail.

## Expected Outputs

- `artifacts/notebooks/inference-smoke.json`
- `artifacts/notebooks/training-smoke.json`
- `artifacts/notebooks/eval-smoke.json`

Required fields checked:

- Inference: `tokens_per_sec`, `gpu_memory_gb`
- Training: `samples_per_sec`, `peak_memory_gb`
- Evaluation: `perplexity`, `tokens_per_sec`

## Deterministic Fallback (Model Access)

If `Qwen/Qwen2.5-0.5B-Instruct` cannot be downloaded in your environment, switch notebook commands to `microsoft/phi-2` for local verification, then rerun the same smoke flow.

## Spark Troubleshooting

### `jupyter: command not found`

Install user-local Jupyter:

```bash
python3 -m pip install --user notebook nbconvert ipykernel
export PATH="$HOME/.local/bin:$PATH"
```

### GPU not visible

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Notebook execution timeout or kernel crash

- Close stale kernels and rerun only the failed notebook.
- Reduce smoke parameters in notebook command cells (`--tokens`, `--steps`, `--subset-size`) for quick validation.
