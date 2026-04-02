# Power and Thermal Profiling (DGX Spark)

This guide captures operational power and thermal behavior on DGX Spark while running the vLLM continuous batching workload.

## Why This Matters

Power draw, GPU temperature, and utilization determine whether a serving setup is stable under sustained load. This profile gives a reproducible snapshot for operational checks.

## One-Command Run

```bash
./scripts/profile_power_thermal.sh
```

Default workload is:

```bash
./scripts/smoke_vllm_continuous_batching.sh
```

## Configurable Parameters

You can override behavior with environment variables:

- `SAMPLE_INTERVAL_MS` (default: `1000`)
- `WORKLOAD_CMD` (default: `./scripts/smoke_vllm_continuous_batching.sh`)
- `RESULT_JSON_PATH` (default: `artifacts/benchmarks/power-thermal-continuous-batching-<date>.json`)
- `RAW_CSV_PATH` (default: `/tmp/power_thermal_samples.csv`)
- `WORKLOAD_LOG_PATH` (default: `/tmp/power_thermal_workload.log`)

Example:

```bash
SAMPLE_INTERVAL_MS=500 \
RESULT_JSON_PATH=artifacts/benchmarks/power-thermal-continuous-batching-2026-04-02.json \
./scripts/profile_power_thermal.sh
```

## Expected PASS Signatures

Console output should include:

- `PASS: power/thermal profiling completed`
- Summary line with:
  - `samples=...`
  - `duration_s=...`
  - `avg_temp_c=...`
  - `max_temp_c=...`
  - `avg_power_w=...`
  - `max_power_w=...`
  - `avg_gpu_util_pct=...`
  - `workload_exit=0`

## Output Files

- Raw samples: `/tmp/power_thermal_samples.csv`
- Workload log: `/tmp/power_thermal_workload.log`
- Summary JSON: `artifacts/benchmarks/power-thermal-continuous-batching-<date>.json`

Expected JSON fields:

- `sample_count` (`> 0`)
- `workload_exit_code` (`0`)
- `duration_s`
- `metrics.avg_temp_c`, `metrics.max_temp_c`
- `metrics.avg_power_w`, `metrics.max_power_w`
- `metrics.avg_gpu_util_pct`, `metrics.max_gpu_util_pct`

## Spark Troubleshooting

1. `nvidia-smi` returns `N/A`
Use a supported query set (`temperature.gpu`, `power.draw`, `utilization.gpu`) as already configured in the script. If all metrics are unavailable, script exits non-zero by design.

2. Docker or GPU runtime errors in workload
Run the continuous batching smoke directly first:

```bash
./scripts/smoke_vllm_continuous_batching.sh
```

3. Workload timeout or model download issues
Check `/tmp/power_thermal_workload.log` and verify network/auth for model pulls.

4. Permission/runtime mismatch on Spark
Verify preflight manually:

```bash
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits
```
