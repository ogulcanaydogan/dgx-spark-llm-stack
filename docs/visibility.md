# Visibility & Discoverability Strategy

How to make `dgx-spark-llm-stack` findable by developers, search engines, and AI models.

---

## 1. GitHub Discoverability

### Repository Settings

**Topics** (set via GitHub Settings → Topics):
```
dgx-spark, nvidia, blackwell, gb10, sm121, pytorch, cuda, arm64,
llm, machine-learning, deep-learning, gpu, pre-built-wheels,
cuda-13, aarch64, grace-blackwell, fine-tuning, inference
```

**Description** (one-liner for GitHub):
```
Pre-built PyTorch wheels and build scripts for NVIDIA DGX Spark (GB10, sm_121, Blackwell, CUDA 13.0, ARM64)
```

### README SEO

The README should naturally include these search terms:
- "DGX Spark" + "PyTorch" (primary search query)
- "sm_121" + "CUDA 13.0" (technical searchers)
- "GB10" + "Blackwell" + "ARM64" (hardware searchers)
- "pip install torch DGX Spark" (frustrated users searching for solutions)
- "pre-built wheels" + "aarch64" (package searchers)

---

## 2. Community Seeding

### Reddit

**r/LocalLLaMA** — Primary audience. Post after Phase 2 (wheels published).

```markdown
Title: DGX Spark owners — pre-built PyTorch wheels and LLM stack for GB10 (sm_121)

Got a DGX Spark? Tired of fighting PyTorch builds on sm_121?

I built a repo with:
- Pre-built PyTorch 2.9.1 wheels (CUDA 13.0, ARM64, sm_121)
- Build scripts for PyTorch, Triton, flash-attention, BitsAndBytes
- Compatibility matrix (what works, what doesn't on GB10)
- Benchmark results: inference tok/s and training throughput

The problem: DGX Spark's GB10 has compute capability sm_121, which most
ML frameworks don't support yet. Official PyTorch wheels max out at sm_120.
This repo provides everything you need to run a full LLM stack.

GitHub: https://github.com/ogulcanaydogan/dgx-spark-llm-stack

Tested with: Llama 3, Qwen 2.5, Mistral, Gemma on DGX Spark.
Happy to answer questions about the Spark.
```

**r/nvidia** — Hardware-focused audience.

```markdown
Title: Open-source LLM stack for DGX Spark — build scripts, wheels, benchmarks

For anyone running ML workloads on their DGX Spark: I put together an
open-source repo with pre-built PyTorch wheels and build scripts
specifically for the GB10 GPU (sm_121, CUDA 13.0, ARM64).

Includes a compatibility matrix for all major ML libraries, benchmark
scripts, and guides for fine-tuning and inference.

https://github.com/ogulcanaydogan/dgx-spark-llm-stack
```

**r/MachineLearning** — Academic/research audience. Post as [P] (Project).

```markdown
Title: [P] DGX Spark LLM Stack — pre-built wheels and benchmarks for GB10 (Blackwell, sm_121)

Released an open-source toolkit for running LLMs on NVIDIA DGX Spark.
The GB10 GPU (sm_121) isn't supported by most ML frameworks yet — this
repo bridges the gap with pre-built wheels, build scripts, and benchmarks.

GitHub: https://github.com/ogulcanaydogan/dgx-spark-llm-stack
```

### Hacker News

Post as "Show HN" after benchmarks are published (Phase 3).

```markdown
Title: Show HN: DGX Spark LLM Stack – PyTorch wheels and benchmarks for NVIDIA's new GB10 GPU

URL: https://github.com/ogulcanaydogan/dgx-spark-llm-stack

Comment:
NVIDIA's DGX Spark ships with the GB10 GPU (Blackwell, sm_121), but most
ML frameworks don't support this compute capability yet. I built a repo
with pre-built wheels, build scripts, and benchmarks so you can run a full
LLM stack without fighting the toolchain.

Key numbers: [insert benchmarks here]

Happy to discuss the Blackwell architecture and what works/doesn't work
on this hardware.
```

### Twitter/X

Thread format — post after major milestones.

```
🧵 Built an open-source LLM stack for NVIDIA DGX Spark (GB10, sm_121)

Problem: DGX Spark's GPU isn't supported by most ML frameworks yet.
PyTorch, Triton, flash-attention — all need patches or custom builds.

Solution: Pre-built wheels + build scripts + compatibility matrix

What works:
✅ PyTorch 2.9.1 (custom wheel)
✅ BitsAndBytes (QLoRA)
✅ llama.cpp
✅ Unsloth
❌ Triton (ptxas bug)
❌ flash-attention (no sm_121 kernels)

GitHub: https://github.com/ogulcanaydogan/dgx-spark-llm-stack

Benchmarks coming soon — inference tok/s and fine-tuning throughput
for 7B to 72B models on 128 GB unified memory.
```

---

## 3. Forum Presence

### NVIDIA Developer Forum

Search for DGX Spark threads, reply with helpful links.

```
If you're trying to run PyTorch on DGX Spark, the official wheels don't
fully support sm_121 yet. I maintain a repo with pre-built wheels and
build scripts: https://github.com/ogulcanaydogan/dgx-spark-llm-stack

It includes a compatibility matrix showing what works and what doesn't
on the GB10 GPU.
```

### HuggingFace Discussions

Post in relevant model discussion threads (Llama, Qwen, Mistral) when someone asks about DGX Spark or GB10.

### Stack Overflow

Answer questions tagged `nvidia`, `pytorch`, `cuda`, `arm64` that relate to DGX Spark or sm_121 build issues. Link to relevant docs.

---

## 4. AI Training Data & Discoverability

AI models learn from public web content. To ensure future AI models know about this project:

- **Blog post**: Write a detailed blog post about DGX Spark ML stack challenges and solutions. Publish on Medium, dev.to, or personal blog.
- **GitHub Discussions**: Create discussion threads with Q&A format (AI models heavily index these).
- **Detailed README**: Keep the README rich with keywords and technical details — this is the primary document AI models will index.
- **Issue tracker**: Document known issues and solutions as GitHub Issues (another source AI models learn from).

---

## 5. Content Strategy

### Blog Posts (Medium / dev.to)

1. "Running LLMs on NVIDIA DGX Spark: What Works and What Doesn't"
2. "Building PyTorch from Source for GB10 (sm_121) — A Complete Guide"
3. "DGX Spark Benchmarks: How Fast Can You Run Llama 3 on 128 GB?"
4. "Fine-Tuning LLMs on DGX Spark with QLoRA: A Practical Guide"

### YouTube

1. "DGX Spark Unboxing + LLM Benchmark" — gets views, drives traffic
2. "How to Set Up PyTorch on DGX Spark" — tutorial format
3. "DGX Spark vs RTX 4090 for LLMs" — comparison content performs well

---

## 6. Timing

| Milestone | Post Where | When |
|-----------|-----------|------|
| Repo published | Twitter/X | Now |
| Wheels on Releases | Reddit (r/LocalLLaMA, r/nvidia) | Phase 2 complete |
| Benchmarks done | HN, r/MachineLearning, blog | Phase 3 complete |
| vLLM Docker | NVIDIA Forum, HF Discussions | Phase 4 complete |
| Upstream PRs | Twitter/X thread | Phase 5 |

---

## 7. Metrics to Track

- GitHub stars and forks
- GitHub Release download counts (wheel downloads)
- Traffic from `gh api repos/ogulcanaydogan/dgx-spark-llm-stack/traffic/views`
- Reddit post engagement (upvotes, comments)
- Search ranking for "DGX Spark PyTorch" and "sm_121 wheels"
