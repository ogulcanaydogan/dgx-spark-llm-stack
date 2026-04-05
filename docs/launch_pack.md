# Community Launch Pack (7-Day)

Copy-ready launch content for community channels after roadmap closure.

## Primary Links

- Repo: `https://github.com/ogulcanaydogan/dgx-spark-llm-stack`
- Quickstart: `https://github.com/ogulcanaydogan/dgx-spark-llm-stack/blob/main/docs/quickstart.md`
- Benchmarks: `https://github.com/ogulcanaydogan/dgx-spark-llm-stack/blob/main/docs/benchmarks.md`
- Compatibility: `https://github.com/ogulcanaydogan/dgx-spark-llm-stack/blob/main/COMPATIBILITY.md`
- Troubleshooting: `https://github.com/ogulcanaydogan/dgx-spark-llm-stack/blob/main/docs/troubleshooting.md`
- TensorRT-LLM evidence artifact: `https://github.com/ogulcanaydogan/dgx-spark-llm-stack/blob/main/artifacts/benchmarks/tensorrt-llm-attention-sinks-2026-04-03.json`

## CTA (Use Everywhere)

- "Start with `./install.sh`, then run `python scripts/verify_install.py`."
- "If you are on DGX Spark/GB10 (`sm_121`), share your model + config + results via Issues."

## Day 1 — X Thread + GitHub Announcement

### X Thread Draft

1. Built and open-sourced a production-ready LLM stack for DGX Spark (GB10, `sm_121`, ARM64, CUDA 13.0).  
2. Includes pre-built wheels, reproducible builds, benchmark artifacts, serving runbooks, and compatibility docs.  
3. What works in practice: PyTorch + transformers + PEFT/TRL + BitsAndBytes + vLLM (container path).  
4. Known caveats are explicit: Triton needs env fix, flash-attn uses SDPA fallback, FP8 uses BF16 fallback.  
5. TensorRT-LLM attention-sinks now has deterministic legacy-fail + stable-pass evidence.  
6. Start here: https://github.com/ogulcanaydogan/dgx-spark-llm-stack  
7. CTA: run `./install.sh` then `python scripts/verify_install.py`; share model/config/result in Issues.

### GitHub Announcement Draft

Title: `DGX Spark LLM Stack v1: GB10/sm_121 roadmap fully completed`

Body:

DGX Spark (`GB10`, `sm_121`, ARM64, CUDA 13.0) has practical ecosystem gaps. This repository now provides a complete, reproducible path to run LLM workloads with explicit compatibility guidance.

What is shipped:
- Pre-built release wheels + checksum-verified installer
- Benchmark artifacts (inference/training/quality)
- vLLM serving docs and smoke flows
- Troubleshooting and compatibility matrix for known caveats
- TensorRT-LLM attention-sinks deterministic validation (legacy fail + stable pass)

Start here:
- Quickstart: https://github.com/ogulcanaydogan/dgx-spark-llm-stack/blob/main/docs/quickstart.md
- Compatibility: https://github.com/ogulcanaydogan/dgx-spark-llm-stack/blob/main/COMPATIBILITY.md
- Benchmarks: https://github.com/ogulcanaydogan/dgx-spark-llm-stack/blob/main/docs/benchmarks.md
- Troubleshooting: https://github.com/ogulcanaydogan/dgx-spark-llm-stack/blob/main/docs/troubleshooting.md
- TensorRT evidence: https://github.com/ogulcanaydogan/dgx-spark-llm-stack/blob/main/artifacts/benchmarks/tensorrt-llm-attention-sinks-2026-04-03.json

CTA:
- Run `./install.sh` and `python scripts/verify_install.py`
- If you are on DGX Spark/GB10 (`sm_121`), open an Issue with model + config + result

## Day 2 — Reddit `r/LocalLLaMA`

Title: `DGX Spark (GB10/sm_121) LLM stack: pre-built wheels, benchmarks, and serving runbooks`

Body skeleton:
- Problem: official ecosystem lag on `sm_121`
- What this repo provides
- Hard evidence links (benchmarks + compatibility + TensorRT artifact)
- Ask for feedback from Spark users

## Day 3 — Reddit `r/nvidia`

Title: `Open-source DGX Spark software stack for GB10 (CUDA 13.0, ARM64)`

Body skeleton:
- Hardware-specific friction points
- Reproducible installation and validation flow
- Performance + operational docs
- CTA to test and report

## Day 4 — Show HN

Title: `Show HN: DGX Spark LLM stack for GB10/sm_121 (wheels, benchmarks, runbooks)`

Comment skeleton:
- Why this exists
- What is validated
- What is still caveat-only
- Direct links + CTA

## Day 5-7 — Forum & HF Replies Template

Use this short response template:

"If you are running DGX Spark (GB10 / `sm_121`), this open-source stack provides reproducible install + compatibility matrix + benchmark evidence. Start with quickstart and verify script, then use troubleshooting for known caveats. Happy to help with your specific model/config."

Then append:
- repo link
- compatibility link
- troubleshooting link

## Tracking Sheet Fields

For each post/channel log:
- Date/time (UTC)
- Channel and URL
- Stars before/after 24h
- Traffic delta (views/unique)
- Release download delta
- Top feedback themes
