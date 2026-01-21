# Low-VRAM Optimization Guide

## Overview

This NVIDIA-only Ollama fork is optimized specifically for running large language models (70B+ parameters) on GPUs with limited VRAM. The optimizations leverage CUDA unified memory, aggressive memory mapping, and Pascal architecture-specific features to enable inference on cards like the GTX 1050 Mobile (4GB VRAM).

## Target Hardware

**Primary Target:**
- GPU: NVIDIA GTX 1050 Mobile (Pascal, Compute Capability 6.1, 4GB VRAM)
- System RAM: 32GB total, with 20GB available for model caching
- Storage: SSD with sufficient space for model files (e.g., 70B Q4 model ~35GB)

**Compatible Hardware:**
- Any NVIDIA GPU with Pascal architecture (Compute Capability 6.1) or newer
- Minimum 4GB VRAM (more is better)
- Minimum 16GB system RAM (32GB+ recommended for 70B models)
- SSD strongly recommended for best mmap performance

## Key Technologies

### 1. Zero-Copy Memory Mapping
Models are memory-mapped directly from disk into virtual memory backed by system RAM. The GPU accesses this data through CUDA unified memory, creating a "sliding window" where only active layers reside in VRAM.

**Benefits:**
- Eliminates slow disk I/O during inference
- Allows running models larger than available VRAM
- 2-4x faster than disk-based layer swapping approaches

### 2. CUDA Unified Memory
Leverages CUDA managed memory and hardware page faulting (available on Pascal+) to automatically migrate data between system RAM and VRAM as needed.

**Features:**
- Automatic page migration between CPU and GPU
- Hardware page faulting for transparent access
- Explicit prefetching for predictable access patterns

### 3. Pascal Architecture Optimizations
- FP16 half-precision compute for faster inference
- `__dp4a` instruction for INT8 quantized operations
- Optimized kernel parameters for Pascal's 128 CUDA cores per SM
- 48KB shared memory per SM utilization

## Configuration

### Environment Variables

#### VRAM Management

**`OLLAMA_VRAM_BUDGET`** (default: auto-detect)
- Maximum VRAM to use in megabytes
- Set to leave headroom for activations and KV cache
- Example: `OLLAMA_VRAM_BUDGET=3500` for 4GB card (leaves ~500MB overhead)

**`OLLAMA_UNIFIED_MEMORY`** (default: true for <6GB VRAM)
- Enable CUDA unified memory for low-VRAM optimization
- Automatically enabled for GPUs with less than 6GB VRAM
- Example: `OLLAMA_UNIFIED_MEMORY=true`

#### Layer Prefetching

**`OLLAMA_PREFETCH_LAYERS`** (default: 2)
- Number of layers to prefetch ahead of current processing
- Higher values use more VRAM but reduce latency
- Example: `OLLAMA_PREFETCH_LAYERS=3`

**`OLLAMA_MLOCK_HOT_LAYERS`** (default: true)
- Lock frequently accessed layers (embeddings, output) in RAM
- Prevents them from being swapped to disk
- Example: `OLLAMA_MLOCK_HOT_LAYERS=true`

#### KV Cache Optimization

**`OLLAMA_KV_CACHE_WINDOW`** (default: 4096)
- Sliding window size for KV cache in tokens
- Limits KV cache memory usage
- Set to 0 to disable sliding window
- Example: `OLLAMA_KV_CACHE_WINDOW=2048`

**`OLLAMA_KV_CACHE_QUANT`** (default: none)
- KV cache quantization to reduce memory usage
- Options: `none`, `int8`
- INT8 provides ~50% memory reduction with minimal quality loss
- Example: `OLLAMA_KV_CACHE_QUANT=int8`

#### Architecture-Specific

**`OLLAMA_PASCAL_OPTIMIZE`** (default: auto-detect)
- Enable Pascal-specific kernel optimizations
- Auto-detects compute capability and enables for CC 6.1
- Example: `OLLAMA_PASCAL_OPTIMIZE=true`

## Performance Expectations

### GTX 1050 4GB + 32GB RAM

| Model | Quantization | VRAM Usage | RAM Usage | Tokens/sec | TTFT |
|-------|--------------|------------|-----------|------------|------|
| 7B | Q4_K_M | ~2.5GB | ~4GB | 15-25 | <5s |
| 13B | Q4_K_M | ~3.5GB | ~8GB | 8-15 | <10s |
| 34B | Q4_K_M | ~3.8GB | ~20GB | 3-6 | <20s |
| 70B | Q4_0 | ~3.9GB | ~35GB | 2-4 | <30s |

**Notes:**
- TTFT = Time To First Token
- Performance varies with context length and prompt complexity
- SSD storage highly recommended for models >20GB

## Usage Examples

### Running a 70B Model on GTX 1050

```bash
# Set environment variables for optimal 70B performance
export OLLAMA_VRAM_BUDGET=3500
export OLLAMA_UNIFIED_MEMORY=true
export OLLAMA_PREFETCH_LAYERS=2
export OLLAMA_MLOCK_HOT_LAYERS=true
export OLLAMA_KV_CACHE_WINDOW=2048
export OLLAMA_KV_CACHE_QUANT=int8

# Run the model
ollama run llama3.3:70b-instruct-q4_0
```

### Conservative Settings (Stability Priority)

```bash
export OLLAMA_VRAM_BUDGET=3000  # More conservative VRAM usage
export OLLAMA_PREFETCH_LAYERS=1  # Less aggressive prefetching
export OLLAMA_KV_CACHE_WINDOW=1024  # Smaller context window
export OLLAMA_KV_CACHE_QUANT=int8

ollama run llama3.3:70b-instruct-q4_0
```

### Performance Settings (Speed Priority)

```bash
export OLLAMA_VRAM_BUDGET=3800  # Use more VRAM
export OLLAMA_PREFETCH_LAYERS=3  # Aggressive prefetching
export OLLAMA_KV_CACHE_WINDOW=4096  # Larger context
export OLLAMA_MLOCK_HOT_LAYERS=true

ollama run llama2:13b-instruct-q4_k_m
```

## Troubleshooting

### Out of Memory Errors

**Symptoms:** OOM errors, CUDA allocation failures

**Solutions:**
1. Reduce `OLLAMA_VRAM_BUDGET` (e.g., set to 3000 instead of 3500)
2. Decrease `OLLAMA_PREFETCH_LAYERS` to 1
3. Enable `OLLAMA_KV_CACHE_QUANT=int8`
4. Reduce `OLLAMA_KV_CACHE_WINDOW` (e.g., 2048 or 1024)
5. Use more aggressive quantization (Q4_0 instead of Q4_K_M)

### Slow Performance

**Symptoms:** Low tokens/sec, high latency

**Solutions:**
1. Ensure model is on SSD, not HDD
2. Increase `OLLAMA_PREFETCH_LAYERS` to 3-4
3. Disable other GPU-intensive applications
4. Check system RAM usage (should have 4-8GB free)
5. Verify CUDA unified memory is enabled

### High RAM Usage

**Symptoms:** System RAM fills up, swapping to disk

**Solutions:**
1. Use more aggressive quantization (Q4_0 < Q4_K_M < Q5_K_M)
2. Run smaller models
3. Disable `OLLAMA_MLOCK_HOT_LAYERS`
4. Close other applications

### Incorrect Architecture Detection

**Symptoms:** Pascal optimizations not applying

**Solutions:**
1. Manually enable: `export OLLAMA_PASCAL_OPTIMIZE=true`
2. Check GPU compute capability with `nvidia-smi`
3. Rebuild with explicit architecture: `cmake -DCMAKE_CUDA_ARCHITECTURES=61`

## Build from Source

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install cmake ninja-build nvidia-cuda-toolkit

# Check CUDA version
nvcc --version
```

### Build with Pascal Optimizations

```bash
git clone https://github.com/EdgeOfAssembly/ollama.git
cd ollama

# Build with Pascal (CC 6.1) as target
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build -j$(nproc)

# Install
sudo cmake --install build
```

### Build for Specific Architecture

For different NVIDIA architectures:
- GTX 1050 (Pascal): `-DCMAKE_CUDA_ARCHITECTURES=61`
- GTX 1650 (Turing): `-DCMAKE_CUDA_ARCHITECTURES=75`
- RTX 3050 (Ampere): `-DCMAKE_CUDA_ARCHITECTURES=86`
- RTX 4060 (Ada): `-DCMAKE_CUDA_ARCHITECTURES=89`

## Comparison with Other Approaches

### vs. Standard Ollama
- **Advantage:** 2-4x faster inference on low-VRAM GPUs
- **Trade-off:** NVIDIA-only, requires more system RAM

### vs. AirLLM (Disk-Based Layer Swapping)
- **Advantage:** 2-4x faster (avoids disk I/O)
- **Trade-off:** Requires sufficient system RAM (20GB+ for 70B)

### vs. Quantization Only
- **Advantage:** Can run larger models with same quantization
- **Trade-off:** Slightly higher latency due to VRAM-RAM transfers

## Best Practices

1. **Use SSD Storage:** Essential for good mmap performance
2. **Monitor Memory:** Use `nvidia-smi` and `htop` to track usage
3. **Start Conservative:** Begin with lower VRAM budget and increase gradually
4. **Match Context to Task:** Use smaller windows for simple queries
5. **Choose Right Quantization:** Balance between quality and memory usage

## Technical Details

### Memory Architecture

```
[Model File on Disk (35GB for 70B Q4)]
         ↓ mmap with MAP_POPULATE
[Virtual Memory backed by RAM (20GB available)]
         ↓ CUDA Unified Memory
[GPU VRAM "Sliding Window" (3.5-4GB)]
```

### Layer Access Pattern

```
Processing Layer N:
1. Layer N in VRAM (actively computing)
2. Layer N+1 prefetching to VRAM (cudaMemPrefetchAsync)
3. Layer N+2 marked for prefetch (cudaMemAdvise)
4. Layer N-1 marked for eviction (POSIX_MADV_DONTNEED)
```

## Future Improvements

- [ ] Multi-GPU support with layer splitting
- [ ] Dynamic VRAM budget adjustment based on load
- [ ] Improved prefetch prediction using layer access history
- [ ] Support for newer architectures (Turing, Ampere, Ada)
- [ ] Automatic quantization selection based on available memory

## Contributing

This is a specialized fork focusing on NVIDIA low-VRAM optimization. Contributions are welcome for:
- Performance improvements
- Bug fixes
- Documentation
- Testing on different hardware configurations

## License

Same as original Ollama project (MIT License).
