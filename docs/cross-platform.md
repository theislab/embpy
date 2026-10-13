# Cross-Platform & Hardware Guide

`embpy` is designed to run everywhere, from laptops to distributed high-performance computing (HPC) clusters across Linux, macOS, and Windows.

---

## OS Constraints

### POSIX-Specific Library Gating (Windows)
Certain underlying genomics libraries (`pysam`, `pyensembl`) rely on POSIX system calls and C-compilers that do not build cleanly on native Windows without Cygwin/WSL.

- `embpy` gates these packages conditionally on Windows.
- Sequence extraction, protein embedding, molecule modeling, and AnnData transformations run 100% natively on Windows using pure-Python and standard cross-platform libraries.
- For workflows requiring direct local `.bam` or indexed `.bed` file slicing via `pysam`, use WSL2 or Linux containers.

### Path Handling and File Encodings
- All file loaders throughout `embpy` utilize `pathlib.Path` with UTF-8 encoding enforced by default.
- This prevents Windows-specific code page issues (e.g. `cp1252`) when parsing multi-byte biological identifiers and FASTQ/FASTA headers.

---

## Windows Multiprocessing Safety

Windows utilizes the `spawn` process creation method rather than the POSIX `fork`. In a spawned process, the Python interpreter starts fresh and re-imports the main module from scratch.

### The `__main__` Guard Requirement
When calling `BioEmbedder.embed(...)` with multiple background workers (`num_workers > 0`) or using parallel data loaders on Windows, your entry-point script **must** be guarded:

```python
import torch
from embpy import BioEmbedder

def main():
    embedder = BioEmbedder(device="cpu", num_workers=4)
    sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    embeddings = embedder.embed(sequences, entity_type="protein", model="esm2_8M")
    print(embeddings.shape)

if __name__ == "__main__":
    main()
```

> **Warning:** Omitting `if __name__ == '__main__':` on Windows will cause child processes to re-execute the top-level script recursively, triggering `RuntimeError` or freezing the execution pool.

---

## PyTorch Memory Access & GPU Guidelines

### Adaptive Batching (`batch_size="auto"`)
Embedding large batches of variable-length biological sequences on GPUs can lead to CUDA Out-Of-Memory (OOM) exceptions.

- `embpy` includes built-in OOM recovery: when `batch_size="auto"` is configured, runtime CUDA OOM errors are intercepted.
- The embedder automatically frees PyTorch cache via `torch.cuda.empty_cache()`, halves the active batch size, and retries inference transparently without crashing your pipeline.

### Windows Paging File & Memory Mapping (`disable_mmap`)
On Windows, loading large transformer checkpoints (such as ESM-2 650M or ProtT5) via memory mapping (`mmap`) can trigger:
```
OSError: [WinError 1455] The paging file is too small for this operation to complete
```
To prevent this, `embpy` configures `transformers` and `safetensors` model loaders with `disable_mmap=True` on memory-constrained systems, loading weights directly into addressable RAM instead of exhausting Windows page files.

### Volta (NVIDIA V100) Architecture Support
Standard PyTorch pip wheels (>= 2.5) deprecate or omit compute capability `sm_70` (Volta architecture). If deploying on V100 GPUs:
1. Use NVIDIA NGC PyTorch containers:
   ```bash
   docker run --gpus all -it nvcr.io/nvidia/pytorch:24.02-py3
   ```
2. Or install PyTorch compiled with Volta support.
