#pragma once

// Low-VRAM optimization header for NVIDIA GPUs
// This header defines configuration and utilities for running large models
// on GPUs with limited VRAM using CUDA unified memory and aggressive caching

#include <cstdint>
#include <cstdlib>
#include <string>

namespace ollama {
namespace lowvram {

// Configuration structure for low-VRAM optimizations
struct Config {
    // VRAM budget in bytes (0 = auto-detect)
    uint64_t vram_budget;
    
    // Number of layers to prefetch ahead
    uint32_t prefetch_layers;
    
    // Lock hot layers (embeddings, output) in RAM
    bool mlock_hot_layers;
    
    // KV cache sliding window size in tokens (0 = disabled)
    uint32_t kv_cache_window;
    
    // KV cache quantization type (0 = none, 8 = int8)
    uint32_t kv_cache_quant;
    
    // Enable CUDA unified memory
    bool unified_memory;
    
    // Enable Pascal-specific optimizations
    bool pascal_optimize;
    
    // Load configuration from environment variables
    static Config from_env();
    
    // Default configuration
    static Config defaults();
};

// Get the global low-VRAM configuration
Config& get_config();

// Initialize low-VRAM optimizations
void init();

// Check if CUDA unified memory is available
bool is_unified_memory_available();

// Get compute capability of the first CUDA device
int get_cuda_compute_capability();

// Advise on memory usage for a region
void advise_memory_willneed(void* addr, size_t size);
void advise_memory_dontneed(void* addr, size_t size);

} // namespace lowvram
} // namespace ollama
