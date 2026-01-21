package llm

// #include "lowvram.h"
// #cgo CXXFLAGS: -std=c++17
import "C"

import (
	"github.com/ollama/ollama/envconfig"
)

// InitLowVRAM initializes the low-VRAM optimization system
func InitLowVRAM() {
	// The C++ side will read environment variables directly
	// This is just a placeholder to ensure initialization happens
	// In a full implementation, we could pass Go config to C++
}

// GetLowVRAMConfig returns the current low-VRAM configuration as a string
func GetLowVRAMConfig() string {
	config := "Low-VRAM Configuration:\n"
	config += "  VRAM Budget: " + envconfig.Var("OLLAMA_VRAM_BUDGET") + " MB\n"
	config += "  Prefetch Layers: " + envconfig.Var("OLLAMA_PREFETCH_LAYERS") + "\n"
	config += "  Mlock Hot Layers: " + envconfig.Var("OLLAMA_MLOCK_HOT_LAYERS") + "\n"
	config += "  KV Cache Window: " + envconfig.Var("OLLAMA_KV_CACHE_WINDOW") + "\n"
	config += "  KV Cache Quant: " + envconfig.Var("OLLAMA_KV_CACHE_QUANT") + "\n"
	config += "  Unified Memory: " + envconfig.Var("OLLAMA_UNIFIED_MEMORY") + "\n"
	config += "  Pascal Optimize: " + envconfig.Var("OLLAMA_PASCAL_OPTIMIZE") + "\n"
	return config
}
