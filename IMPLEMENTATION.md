# Implementation Summary: NVIDIA Low-VRAM Ollama Fork

## What Has Been Implemented

This specialized fork of Ollama has been optimized specifically for NVIDIA GPUs with limited VRAM. The implementation focuses on infrastructure and configuration to enable future deep optimizations.

### ✅ Phase 1: Codebase Cleanup (COMPLETE)

**Removed Non-NVIDIA Backends:**
- Deleted `ml/backend/ggml/ggml/src/ggml-metal/` (Apple Metal)
- Deleted `ml/backend/ggml/ggml/src/ggml-vulkan/` (Vulkan)
- Deleted `ml/backend/ggml/ggml/src/ggml-hip/` (AMD ROCm/HIP)
- Deleted `x/ml/backend/mlx/` (Apple MLX)

**Updated Build System:**
- Modified `CMakeLists.txt` to remove all non-NVIDIA backend references
- Set Pascal architecture (Compute Capability 6.1) as default build target
- Removed macOS-specific build configurations
- Cleaned up HIP, Vulkan, and MLX build logic

**Result:** Clean, NVIDIA-only codebase that compiles significantly faster and produces smaller binaries.

### ✅ Phase 2: Configuration Infrastructure (COMPLETE)

**Environment Variables Added** (`envconfig/config.go`):

1. **`OLLAMA_VRAM_BUDGET`** (uint64, default: 0/auto-detect)
   - Maximum VRAM to use in megabytes
   - Example: `OLLAMA_VRAM_BUDGET=3500`

2. **`OLLAMA_PREFETCH_LAYERS`** (uint, default: 2)
   - Number of model layers to prefetch ahead
   - Example: `OLLAMA_PREFETCH_LAYERS=3`

3. **`OLLAMA_MLOCK_HOT_LAYERS`** (bool, default: true)
   - Lock frequently accessed layers in RAM
   - Example: `OLLAMA_MLOCK_HOT_LAYERS=true`

4. **`OLLAMA_KV_CACHE_WINDOW`** (uint, default: 4096)
   - Sliding window size for KV cache in tokens
   - Example: `OLLAMA_KV_CACHE_WINDOW=2048`

5. **`OLLAMA_KV_CACHE_QUANT`** (string, default: "none")
   - KV cache quantization: "none" or "int8"
   - Example: `OLLAMA_KV_CACHE_QUANT=int8`

6. **`OLLAMA_UNIFIED_MEMORY`** (bool, default: true for <6GB VRAM)
   - Enable CUDA unified memory
   - Example: `OLLAMA_UNIFIED_MEMORY=true`

7. **`OLLAMA_PASCAL_OPTIMIZE`** (bool, default: true/auto-detect)
   - Enable Pascal-specific optimizations
   - Example: `OLLAMA_PASCAL_OPTIMIZE=true`

**C++ Configuration System:**
- Created `llm/lowvram.h` with configuration structures
- Created `llm/lowvram.cpp` with environment variable parsing
- Added memory advice utilities (`MADV_WILLNEED`/`DONTNEED`)
- Added CUDA capability detection stubs

**Go Bindings:**
- Created `llm/lowvram.go` with CGo bindings
- Exposed configuration to Go layer

### ✅ Phase 3: Documentation (COMPLETE)

**Created `docs/low-vram-optimization.md`:**
- Comprehensive guide for low-VRAM optimization
- Hardware requirements and recommendations
- Environment variable documentation
- Usage examples for different scenarios
- Troubleshooting guide
- Performance expectations table
- Technical architecture details

**Updated `README.md`:**
- Added fork-specific introduction
- Quick start guide for low-VRAM setup
- Link to comprehensive documentation
- Noted differences from upstream Ollama

## What Still Needs Implementation

### 🔧 Phase 3: Memory Mapping Enhancements (PARTIAL)

**Remaining Work:**
- Integrate memory advice into llama.cpp's mmap implementation
- Add aggressive prefetching for sequential layer access
- Implement hot layer locking with `mlock()`
- Add layer access pattern tracking

**Files to Modify:**
- `llama/llama.cpp/src/llama-mmap.cpp`
- `llama/llama.cpp/src/llama-model-loader.cpp`

### 🔧 Phase 4: CUDA Unified Memory (TODO)

**Required Implementation:**
- Replace `cudaMalloc()` with `cudaMallocManaged()` for tensors
- Implement sliding window layer management
- Add `cudaMemPrefetchAsync()` for upcoming layers
- Add `cudaMemAdvise()` for access pattern hints
- Implement VRAM budget tracking and enforcement
- Add LRU eviction when budget exceeded

**Files to Modify:**
- `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu`
- Create new file: `ml/backend/ggml/ggml/src/ggml-cuda/unified-memory.cu`

### 🔧 Phase 5: KV Cache Optimizations (TODO)

**Required Implementation:**
- Sliding window attention for bounded memory
- INT8 quantization for KV cache
- Dynamic window size adjustment

**Files to Modify:**
- `kvcache/causal.go`
- `llama/llama.cpp/src/llama-context.cpp` (if needed)

### 🔧 Phase 6: Testing & Benchmarking (TODO)

**Required Implementation:**
- Create benchmark suite in `benchmarks/low_vram_benchmark.go`
- Add memory profiling hooks
- Create regression tests
- Performance comparison scripts

## How to Use This Fork

### Build from Source

```bash
git clone https://github.com/EdgeOfAssembly/ollama.git
cd ollama

# Build with Pascal optimization (GTX 1050)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build -j$(nproc)
sudo cmake --install build
```

### Configure for Low-VRAM

```bash
# For GTX 1050 4GB + 32GB RAM running 70B model
export OLLAMA_VRAM_BUDGET=3500
export OLLAMA_UNIFIED_MEMORY=true
export OLLAMA_PREFETCH_LAYERS=2
export OLLAMA_MLOCK_HOT_LAYERS=true
export OLLAMA_KV_CACHE_WINDOW=2048
export OLLAMA_KV_CACHE_QUANT=int8

ollama run llama3.3:70b-instruct-q4_0
```

## Technical Architecture

### Current State

```
┌─────────────────────────────────────────┐
│    Ollama Server (Go)                   │
│  - Environment variable config ✅       │
│  - Low-VRAM config structures ✅        │
├─────────────────────────────────────────┤
│    LLM Backend (C++)                    │
│  - Configuration parsing ✅             │
│  - Memory advice utilities ✅           │
│  - CUDA capability detection ✅         │
├─────────────────────────────────────────┤
│    GGML/llama.cpp Backend              │
│  - Standard mmap (needs enhancement) ⚠️  │
│  - Standard CUDA malloc (needs UM) ⚠️   │
│  - Standard KV cache ⚠️                 │
└─────────────────────────────────────────┘
```

### Target State (After Full Implementation)

```
┌─────────────────────────────────────────┐
│    Model File (Disk - 35GB for 70B Q4)  │
│           ↓ mmap + MAP_POPULATE          │
├─────────────────────────────────────────┤
│    Virtual Memory (RAM - 20GB free)      │
│  - Aggressive prefetching                │
│  - Hot layers mlock'd                    │
│           ↓ CUDA Unified Memory          │
├─────────────────────────────────────────┤
│    GPU VRAM (4GB sliding window)         │
│  - Active layer in VRAM                  │
│  - Next layers prefetching               │
│  - Old layers evicted                    │
│  - KV cache quantized                    │
└─────────────────────────────────────────┘
```

## Key Files and Their Purpose

### Configuration
- `envconfig/config.go` - Go environment variable definitions ✅
- `llm/lowvram.h` - C++ configuration structures ✅
- `llm/lowvram.cpp` - C++ configuration implementation ✅
- `llm/lowvram.go` - Go/C++ bindings ✅

### Build System
- `CMakeLists.txt` - Root build configuration (NVIDIA-only) ✅
- `ml/backend/ggml/ggml/src/ggml-cuda/CMakeLists.txt` - CUDA build config ✅

### Documentation
- `docs/low-vram-optimization.md` - User guide ✅
- `README.md` - Fork introduction ✅
- `IMPLEMENTATION.md` - This document ✅

### Future Implementation Sites
- `llama/llama.cpp/src/llama-mmap.cpp` - Memory mapping enhancements ⚠️
- `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu` - CUDA unified memory ⚠️
- `kvcache/causal.go` - KV cache optimizations ⚠️

## Performance Expectations

### Baseline (Standard Ollama)
- **GTX 1050 4GB:** Cannot run 70B models (OOM)
- **GTX 1050 4GB:** 13B Q4 at 10-12 tokens/sec

### Target (This Fork - Fully Implemented)
- **GTX 1050 4GB + 32GB RAM:** 70B Q4 at 2-4 tokens/sec
- **GTX 1050 4GB + 32GB RAM:** 13B Q4 at 12-15 tokens/sec
- **No OOM crashes** with proper configuration

### Current State (Infrastructure Only)
- Same as baseline (deep optimizations not yet implemented)
- Configuration system ready for future implementation
- Clean build without non-NVIDIA backends

## Next Steps for Contributors

If you want to complete this implementation:

1. **Start with Memory Mapping:**
   - Patch `llama/llama.cpp/src/llama-mmap.cpp`
   - Add layer-aware prefetching
   - Implement hot layer locking

2. **Then CUDA Unified Memory:**
   - Modify `ggml-cuda.cu` allocation functions
   - Add managed memory wrappers
   - Implement sliding window logic

3. **Finally KV Cache:**
   - Implement sliding window in `kvcache/causal.go`
   - Add INT8 quantization option
   - Test with long contexts

4. **Test and Benchmark:**
   - Create comprehensive benchmarks
   - Test on various GPUs (Pascal, Turing, Ampere)
   - Document real-world performance

## Compatibility

### Supported
- NVIDIA GPUs with Compute Capability 6.1+ (Pascal and newer)
- Linux with CUDA Toolkit 11.0+
- Windows with CUDA Toolkit 11.0+

### Not Supported
- AMD GPUs (removed HIP backend)
- Intel GPUs (removed SYCL backend)
- Apple Silicon (removed Metal/MLX backends)
- Vulkan-based systems

## License

Same as upstream Ollama (MIT License).

## Credits

- Original Ollama: https://github.com/ollama/ollama
- llama.cpp: https://github.com/ggml-org/llama.cpp
- GGML: https://github.com/ggerganov/ggml

---

**Status:** Infrastructure complete, deep optimizations pending implementation.

**Last Updated:** 2026-01-21
