#include "lowvram.h"

#include <cstring>
#include <iostream>
#include <sstream>

#ifdef __linux__
#include <sys/mman.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace ollama {
namespace lowvram {

// Global configuration instance
static Config g_config;

Config& get_config() {
    return g_config;
}

Config Config::defaults() {
    Config cfg;
    cfg.vram_budget = 0;  // Auto-detect
    cfg.prefetch_layers = 2;
    cfg.mlock_hot_layers = true;
    cfg.kv_cache_window = 4096;
    cfg.kv_cache_quant = 0;  // None by default
    cfg.unified_memory = true;  // Enable for low-VRAM by default
    cfg.pascal_optimize = true;  // Auto-detect
    return cfg;
}

Config Config::from_env() {
    Config cfg = defaults();
    
    // OLLAMA_VRAM_BUDGET
    if (const char* env = std::getenv("OLLAMA_VRAM_BUDGET")) {
        cfg.vram_budget = std::stoull(env) * 1024 * 1024;  // Convert MB to bytes
    }
    
    // OLLAMA_PREFETCH_LAYERS
    if (const char* env = std::getenv("OLLAMA_PREFETCH_LAYERS")) {
        cfg.prefetch_layers = std::stoul(env);
    }
    
    // OLLAMA_MLOCK_HOT_LAYERS
    if (const char* env = std::getenv("OLLAMA_MLOCK_HOT_LAYERS")) {
        std::string val(env);
        cfg.mlock_hot_layers = (val == "true" || val == "1" || val == "yes");
    }
    
    // OLLAMA_KV_CACHE_WINDOW
    if (const char* env = std::getenv("OLLAMA_KV_CACHE_WINDOW")) {
        cfg.kv_cache_window = std::stoul(env);
    }
    
    // OLLAMA_KV_CACHE_QUANT
    if (const char* env = std::getenv("OLLAMA_KV_CACHE_QUANT")) {
        std::string val(env);
        if (val == "int8") {
            cfg.kv_cache_quant = 8;
        }
    }
    
    // OLLAMA_UNIFIED_MEMORY
    if (const char* env = std::getenv("OLLAMA_UNIFIED_MEMORY")) {
        std::string val(env);
        cfg.unified_memory = (val == "true" || val == "1" || val == "yes");
    }
    
    // OLLAMA_PASCAL_OPTIMIZE
    if (const char* env = std::getenv("OLLAMA_PASCAL_OPTIMIZE")) {
        std::string val(env);
        cfg.pascal_optimize = (val == "true" || val == "1" || val == "yes");
    }
    
    return cfg;
}

void init() {
    g_config = Config::from_env();
    
    // Log configuration
    std::cerr << "[LowVRAM] Initialization:" << std::endl;
    std::cerr << "  VRAM Budget: " << (g_config.vram_budget / (1024*1024)) << " MB" << std::endl;
    std::cerr << "  Prefetch Layers: " << g_config.prefetch_layers << std::endl;
    std::cerr << "  Mlock Hot Layers: " << (g_config.mlock_hot_layers ? "true" : "false") << std::endl;
    std::cerr << "  KV Cache Window: " << g_config.kv_cache_window << std::endl;
    std::cerr << "  KV Cache Quant: " << (g_config.kv_cache_quant == 8 ? "int8" : "none") << std::endl;
    std::cerr << "  Unified Memory: " << (g_config.unified_memory ? "true" : "false") << std::endl;
    std::cerr << "  Pascal Optimize: " << (g_config.pascal_optimize ? "true" : "false") << std::endl;
}

bool is_unified_memory_available() {
#ifdef __CUDACC__
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Unified memory available on compute capability 3.0+
    // Page faulting (optimal) available on compute capability 6.0+ (Pascal)
    return (prop.major >= 3);
#else
    return false;
#endif
}

int get_cuda_compute_capability() {
#ifdef __CUDACC__
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    return prop.major * 10 + prop.minor;
#else
    return 0;
#endif
}

void advise_memory_willneed(void* addr, size_t size) {
#ifdef __linux__
    // Advise kernel that this memory will be needed soon
    posix_madvise(addr, size, POSIX_MADV_WILLNEED);
#endif
}

void advise_memory_dontneed(void* addr, size_t size) {
#ifdef __linux__
    // Advise kernel that this memory is no longer needed
    posix_madvise(addr, size, POSIX_MADV_DONTNEED);
#endif
}

} // namespace lowvram
} // namespace ollama
