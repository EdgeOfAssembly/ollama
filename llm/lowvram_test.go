package llm

import (
	"os"
	"testing"

	"github.com/ollama/ollama/envconfig"
)

func TestLowVRAMEnvVars(t *testing.T) {
	// Set test environment variables
	testVars := map[string]string{
		"OLLAMA_VRAM_BUDGET":      "3500",
		"OLLAMA_PREFETCH_LAYERS":  "3",
		"OLLAMA_MLOCK_HOT_LAYERS": "true",
		"OLLAMA_KV_CACHE_WINDOW":  "2048",
		"OLLAMA_KV_CACHE_QUANT":   "int8",
		"OLLAMA_UNIFIED_MEMORY":   "true",
		"OLLAMA_PASCAL_OPTIMIZE":  "true",
	}

	// Set environment variables
	for k, v := range testVars {
		os.Setenv(k, v)
	}
	defer func() {
		// Clean up
		for k := range testVars {
			os.Unsetenv(k)
		}
	}()

	// Test that environment variables are set correctly
	if envconfig.Var("OLLAMA_VRAM_BUDGET") != "3500" {
		t.Errorf("Expected VRAM budget 3500, got %s", envconfig.Var("OLLAMA_VRAM_BUDGET"))
	}

	if envconfig.Var("OLLAMA_PREFETCH_LAYERS") != "3" {
		t.Errorf("Expected prefetch layers 3, got %s", envconfig.Var("OLLAMA_PREFETCH_LAYERS"))
	}

	if envconfig.Var("OLLAMA_KV_CACHE_QUANT") != "int8" {
		t.Errorf("Expected KV cache quant int8, got %s", envconfig.Var("OLLAMA_KV_CACHE_QUANT"))
	}
}

func TestLowVRAMConfigFunctions(t *testing.T) {
	// Test that VramBudget function exists and can be called
	budget := envconfig.VramBudget()
	if budget < 0 {
		t.Error("VramBudget returned negative value")
	}

	// Test that PrefetchLayers function exists and can be called
	prefetch := envconfig.PrefetchLayers()
	if prefetch < 0 {
		t.Error("PrefetchLayers returned negative value")
	}

	// Test that KvCacheWindow function exists and can be called
	window := envconfig.KvCacheWindow()
	if window < 0 {
		t.Error("KvCacheWindow returned negative value")
	}
}
