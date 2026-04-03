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
2. Includes pre-built wheels, reproducible builds, benchmark artifacts, serving runbooks, and compatibility guidance.  
3. What works: PyTorch + transformers + PEFT/TRL + BitsAndBytes + vLLM (container path).  
4. Known caveats are explicit: Triton env-fix, flash-attn -> SDPA, FP8 -> BF16 fallback.  
5. TensorRT-LLM attention-sinks now has deterministic fail+pass evidence (legacy vs stable tags).  
6. Repo + quickstart: <repo-link>  
7. If you run DGX Spark, try `./install.sh` and share your results.

### GitHub Announcement Draft

Title: `DGX Spark LLM Stack v1: GB10/sm_121 roadmap fully completed`  
Body:
- What shipped (wheels, benchmarks, serving guides, troubleshooting)
- Known limits and workarounds
- Link set (quickstart, compatibility, benchmarks, evidence artifact)
- CTA for user reports and contributions

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
