/**
 * FluxCore Entity — CUDA Header
 * 
 * Core data structures and kernel declarations for the three architectural leaps:
 * 1. Dynamic Attractor Genesis
 * 2. Active Inference  
 * 3. Hierarchical Rule Extraction
 */

#ifndef FLUXCORE_ENTITY_CUH
#define FLUXCORE_ENTITY_CUH

#include <cuda_runtime.h>

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════
#define FC_MAX_MEMORIES 64
#define FC_BLOCK_SIZE 256
#define FC_WARP_SIZE 32

// ═══════════════════════════════════════════════════════════════════════════
// DYNAMIC ATTRACTOR FIELD (Device-Side)
// ═══════════════════════════════════════════════════════════════════════════
struct FCAttractorField {
    float* memories;        // [maxMemories * dim]
    float* meta;            // [maxMemories * 3] — lastUsed, useCount, birthTick
    int* active;            // [maxMemories] — 1 if slot occupied
    int* count;             // [1] — number of active memories
    int* tick;              // [1] — global tick counter
    int maxMemories;
    int dim;
    float spawnThreshold;
    float mergeThreshold;
    int pruneThreshold;
};

// ═══════════════════════════════════════════════════════════════════════════
// FLUXCORE STATE
// ═══════════════════════════════════════════════════════════════════════════
struct FCState {
    float* self;            // [dim] — current self-state
    float* prev;            // [dim] — previous self-state
    float* velocity;        // [dim] — unnormalized velocity
    float* output;          // [dim] — projected output
    float* action;          // [dim] — control signal (active inference)
    int dim;
};

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL DECLARATIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Find best matching memory for current reality
 */
__global__ void fcFindBestMemory(
    const float* reality,
    const FCAttractorField field,
    float* bestSim,
    int* bestIdx
);

/**
 * Core fold operation with dynamic memory selection
 */
__global__ void fcFold(
    const float* reality,
    FCState state,
    FCAttractorField field,
    float baseLr,
    float memLr,
    float k,
    float memW,
    float* surpriseOut
);

/**
 * Normalize a vector
 */
__global__ void fcNormalize(float* vec, int dim);

/**
 * Velocity update and active inference output
 */
__global__ void fcVelocityOutput(
    FCState state,
    float velDecay,
    float velGain,
    float velScale,
    float actionGain
);

/**
 * Merge converged memories
 */
__global__ void fcMergeMemories(FCAttractorField field);

/**
 * Prune stale memories
 */
__global__ void fcPruneMemories(FCAttractorField field);

// ═══════════════════════════════════════════════════════════════════════════
// HOST API
// ═══════════════════════════════════════════════════════════════════════════

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize a FluxCore entity on GPU
 */
cudaError_t fcCreateEntity(
    FCState* state,
    FCAttractorField* field,
    int dim,
    int maxMemories,
    float spawnThreshold,
    float mergeThreshold,
    int pruneThreshold
);

/**
 * Destroy a FluxCore entity
 */
cudaError_t fcDestroyEntity(FCState state, FCAttractorField field);

/**
 * Single forward pass through FluxCore entity
 * 
 * @param h_reality Host reality input [dim]
 * @param h_action  Host action output [dim] (control signal)
 * @param h_surprise Host surprise output [1]
 */
cudaError_t fcForward(
    FCState state,
    FCAttractorField field,
    const float* h_reality,
    float* h_action,
    float* h_surprise,
    float baseLr,
    float memLr,
    float k,
    float memW,
    float velDecay,
    float velGain,
    float velScale,
    float actionGain
);

/**
 * Get memory statistics
 */
cudaError_t fcGetMemoryStats(
    FCAttractorField field,
    int* count,
    float* ages,
    float* useCounts,
    int maxStats
);

#ifdef __cplusplus
}
#endif

#endif // FLUXCORE_ENTITY_CUH
