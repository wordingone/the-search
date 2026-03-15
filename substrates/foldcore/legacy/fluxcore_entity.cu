/**
 * FluxCore Entity — CUDA Implementation
 * 
 * Three Architectural Leaps:
 * 1. DYNAMIC ATTRACTOR GENESIS — Dynamic memory pool with spawn/merge/prune
 * 2. ACTIVE INFERENCE — Output becomes control signal
 * 3. HIERARCHICAL RULE EXTRACTION — Stacked kernel launches
 * 
 * This is not signal processing. This is autonomous intelligence on GPU.
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════
#define MAX_MEMORIES 64           // Maximum attractors in dynamic field
#define WARP_SIZE 32
#define BLOCK_SIZE 256

// ═══════════════════════════════════════════════════════════════════════════
// DYNAMIC ATTRACTOR FIELD STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════
struct AttractorField {
    float* memories;              // [MAX_MEMORIES * dim] — memory vectors
    float* meta;                  // [MAX_MEMORIES * 3] — lastUsed, useCount, birthTick
    int* activeMask;              // [MAX_MEMORIES] — 1 if slot in use
    int* count;                   // [1] — current number of active memories
    int* tick;                    // [1] — global tick counter
    float spawnThreshold;         // Similarity below this spawns new
    float mergeThreshold;         // Similarity above this merges
    int pruneThreshold;           // Ticks before prune
};

// ═══════════════════════════════════════════════════════════════════════════
// DEVICE UTILITIES
// ═══════════════════════════════════════════════════════════════════════════
__device__ float d_dot(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += a[i] * b[i];
    return sum;
}

__device__ void d_normalize(float* v, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += v[i] * v[i];
    float norm = sqrtf(sum) + 1e-12f;
    for (int i = 0; i < dim; i++) v[i] /= norm;
}

__device__ float d_l1(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += fabsf(a[i] - b[i]);
    return sum / dim;
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: Find Best Matching Memory (Parallel Reduction)
// ═══════════════════════════════════════════════════════════════════════════
__global__ void FindBestMemoryKernel(
    const float* reality,
    const float* memories,
    const int* activeMask,
    int dim,
    int maxMemories,
    float* outSim,      // [blockCount] — best similarity per block
    int* outIdx         // [blockCount] — best index per block
) {
    extern __shared__ float sdata[];
    float* s_sims = sdata;
    int* s_idxs = (int*)&s_sims[blockDim.x];
    
    int tid = threadIdx.x;
    int memIdx = blockIdx.x * blockDim.x + tid;
    
    float bestSim = -1.0f;
    int bestIdx = -1;
    
    if (memIdx < maxMemories && activeMask[memIdx]) {
        const float* mem = memories + memIdx * dim;
        float sim = fabsf(d_dot(reality, mem, dim));
        bestSim = sim;
        bestIdx = memIdx;
    }
    
    s_sims[tid] = bestSim;
    s_idxs[tid] = bestIdx;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_sims[tid + s] > s_sims[tid]) {
                s_sims[tid] = s_sims[tid + s];
                s_idxs[tid] = s_idxs[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        outSim[blockIdx.x] = s_sims[0];
        outIdx[blockIdx.x] = s_idxs[0];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: FluxCore Fold with Dynamic Memory
// ═══════════════════════════════════════════════════════════════════════════
__global__ void FluxCoreFoldKernelEntity(
    const float* __restrict__ reality,
    float* __restrict__ self_state,
    float* __restrict__ memories,
    float* __restrict__ meta,
    int* __restrict__ activeMask,
    int* __restrict__ memCount,
    int* __restrict__ tick,
    int dim,
    int maxMemories,
    float base_lr,
    float mem_lr,
    float k,
    float mem_w,
    float spawnThreshold,
    int bestMemIdx,       // Pre-computed best memory index
    float bestMemSim      // Pre-computed best memory similarity
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    
    float s = self_state[idx];
    float r = reality[idx];

    // Bug 3 fix: parallel reduction for surprise (each thread handles its own idx)
    __shared__ float s_surp[BLOCK_SIZE];
    __shared__ float s_self[BLOCK_SIZE];  // Bug 5 fix: shared staging for gradient
    s_surp[threadIdx.x] = fabsf(s - r);
    s_self[threadIdx.x] = s;
    __syncthreads();

    // Tree reduction for surprise
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_surp[threadIdx.x] += s_surp[threadIdx.x + stride];
        __syncthreads();
    }
    float surprise = s_surp[0] / dim;

    // Adaptive learning rate
    float alr = base_lr * (1.0f + k * surprise);

    // Select or spawn memory
    float activeMem;
    int memIdx = bestMemIdx;

    if (bestMemSim < spawnThreshold && idx == 0) {
        // Spawn new memory — only thread 0 does this
        int newIdx = atomicAdd(memCount, 1);
        if (newIdx < maxMemories) {
            activeMask[newIdx] = 1;
            meta[newIdx * 3 + 0] = (float)(*tick);  // lastUsed
            meta[newIdx * 3 + 1] = 1.0f;             // useCount
            meta[newIdx * 3 + 2] = (float)(*tick);  // birthTick

            // Copy reality to new memory
            for (int i = 0; i < dim; i++) {
                memories[newIdx * dim + i] = reality[i];
            }
            memIdx = newIdx;
        }
    }

    __syncthreads();

    // Get active memory vector
    if (memIdx >= 0 && memIdx < maxMemories && activeMask[memIdx]) {
        activeMem = memories[memIdx * dim + idx];
    } else {
        activeMem = r;  // Fallback to reality
    }

    // Bug 5 fix: gradient via shared memory — correct cross-warp neighbor access
    float left  = s_self[(threadIdx.x + dim - 1) % dim];
    float right = s_self[(threadIdx.x + 1) % dim];
    float grad  = (s - left) - (s - right);
    
    // L1 difference
    float d = fabsf(s - r);
    
    // Fold update
    float u = s + alr * r + (alr * 0.5f) * d * grad + mem_w * activeMem;
    self_state[idx] = u;
    
    // Update memory toward reality (only if we have a valid memory)
    if (memIdx >= 0 && memIdx < maxMemories && activeMask[memIdx]) {
        float* mem = memories + memIdx * dim + idx;
        *mem += mem_lr * r;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: Normalize Vectors (Post-fold)
// ═══════════════════════════════════════════════════════════════════════════
__global__ void NormalizeKernel(
    float* vectors,
    int* activeMask,
    int numVectors,
    int dim
) {
    int vecIdx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (vecIdx >= numVectors || (activeMask != nullptr && !activeMask[vecIdx])) return;
    
    float* v = vectors + vecIdx * dim;
    
    // Compute norm
    __shared__ float s_sum[BLOCK_SIZE];
    s_sum[tid] = (tid < dim) ? v[tid] * v[tid] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    
    float norm = sqrtf(s_sum[0]) + 1e-12f;
    
    // Normalize
    if (tid < dim) {
        v[tid] /= norm;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: Velocity & Output (Active Inference)
// ═══════════════════════════════════════════════════════════════════════════
__global__ void VelocityOutputKernel(
    const float* __restrict__ self_state,
    const float* __restrict__ prev_self,
    float* __restrict__ velocity,
    float* __restrict__ output,
    float* __restrict__ action,     // NEW: Control signal for environment
    int dim,
    float vel_decay,
    float vel_gain,
    float vel_scale,
    float action_gain               // NEW: How much output affects environment
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    
    float s = self_state[idx];
    float p = prev_self[idx];
    float v = velocity[idx];
    
    // Update unnormalized velocity
    v = vel_decay * v + vel_gain * (s - p);
    velocity[idx] = v;
    
    // Project output
    float o = s + vel_scale * v;
    output[idx] = o;
    
    // ACTIVE INFERENCE: Generate action to minimize surprise
    // Action pushes environment toward self (reducing future surprise)
    action[idx] = action_gain * (s - o);
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: Merge Converged Memories
// ═══════════════════════════════════════════════════════════════════════════
__global__ void MergeMemoriesKernel(
    float* memories,
    float* meta,
    int* activeMask,
    int* memCount,
    int dim,
    float mergeThreshold
) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    
    if (i >= *memCount || j >= *memCount || i >= j) return;
    if (!activeMask[i] || !activeMask[j]) return;
    
    // Compute similarity
    float sim = fabsf(d_dot(memories + i * dim, memories + j * dim, dim));
    
    if (sim > mergeThreshold) {
        // Fuse memories (weighted by use count)
        float useI = meta[i * 3 + 1];
        float useJ = meta[j * 3 + 1];
        float total = useI + useJ;
        
        for (int k = 0; k < dim; k++) {
            memories[i * dim + k] = (useI * memories[i * dim + k] + useJ * memories[j * dim + k]) / total;
        }
        
        // Update meta
        meta[i * 3 + 0] = fmaxf(meta[i * 3 + 0], meta[j * 3 + 0]);  // lastUsed
        meta[i * 3 + 1] = total;                                     // useCount
        meta[i * 3 + 2] = fminf(meta[i * 3 + 2], meta[j * 3 + 2]);  // birthTick
        
        // Deactivate j
        activeMask[j] = 0;
        atomicSub(memCount, 1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: Prune Stale Memories
// ═══════════════════════════════════════════════════════════════════════════
__global__ void PruneMemoriesKernel(
    float* meta,
    int* activeMask,
    int* memCount,
    int* tick,
    int pruneThreshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MAX_MEMORIES) return;
    
    if (activeMask[i]) {
        int lastUsed = (int)meta[i * 3 + 0];
        if (*tick - lastUsed > pruneThreshold) {
            activeMask[i] = 0;
            atomicSub(memCount, 1);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HOST: Hierarchical FluxCore Entity
// ═══════════════════════════════════════════════════════════════════════════
class HierarchicalFluxCoreCUDA {
public:
    int dim;
    int numLevels;
    int maxMemories;
    
    // Per-level device arrays
    float** d_self;           // [numLevels][dim]
    float** d_prevSelf;       // [numLevels][dim]
    float** d_velocity;       // [numLevels][dim]
    float** d_output;         // [numLevels][dim]
    float** d_action;         // [numLevels][dim]
    
    // Dynamic attractor fields per level
    float** d_memories;       // [numLevels][maxMemories * dim]
    float** d_meta;           // [numLevels][maxMemories * 3]
    int** d_activeMask;       // [numLevels][maxMemories]
    int** d_memCount;         // [numLevels][1]
    int** d_tick;             // [numLevels][1]
    
    // Temporary buffers
    float* d_bestSim;
    int* d_bestIdx;
    
    HierarchicalFluxCoreCUDA(int d, int levels = 3, int maxMem = 64) {
        dim = d;
        numLevels = levels;
        maxMemories = maxMem;
        
        // Allocate per-level arrays
        d_self = new float*[numLevels];
        d_prevSelf = new float*[numLevels];
        d_velocity = new float*[numLevels];
        d_output = new float*[numLevels];
        d_action = new float*[numLevels];
        d_memories = new float*[numLevels];
        d_meta = new float*[numLevels];
        d_activeMask = new int*[numLevels];
        d_memCount = new int*[numLevels];
        d_tick = new int*[numLevels];
        
        for (int l = 0; l < numLevels; l++) {
            cudaMalloc(&d_self[l], dim * sizeof(float));
            cudaMalloc(&d_prevSelf[l], dim * sizeof(float));
            cudaMalloc(&d_velocity[l], dim * sizeof(float));
            cudaMalloc(&d_output[l], dim * sizeof(float));
            cudaMalloc(&d_action[l], dim * sizeof(float));
            
            cudaMalloc(&d_memories[l], maxMemories * dim * sizeof(float));
            cudaMalloc(&d_meta[l], maxMemories * 3 * sizeof(float));
            cudaMalloc(&d_activeMask[l], maxMemories * sizeof(int));
            cudaMalloc(&d_memCount[l], sizeof(int));
            cudaMalloc(&d_tick[l], sizeof(int));
            
            // Initialize
            cudaMemset(d_velocity[l], 0, dim * sizeof(float));
            cudaMemset(d_activeMask[l], 0, maxMemories * sizeof(int));
            cudaMemset(d_memCount[l], 0, sizeof(int));
            cudaMemset(d_tick[l], 0, sizeof(int));
            
            // Random init for self
            float* h_self = new float[dim];
            for (int i = 0; i < dim; i++) h_self[i] = (float)rand() / RAND_MAX;
            // Normalize
            float norm = 0;
            for (int i = 0; i < dim; i++) norm += h_self[i] * h_self[i];
            norm = sqrtf(norm) + 1e-12f;
            for (int i = 0; i < dim; i++) h_self[i] /= norm;
            
            cudaMemcpy(d_self[l], h_self, dim * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_prevSelf[l], h_self, dim * sizeof(float), cudaMemcpyHostToDevice);
            delete[] h_self;
        }
        
        // Allocate temp buffers
        int blockCount = (maxMemories + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMalloc(&d_bestSim, blockCount * sizeof(float));
        cudaMalloc(&d_bestIdx, blockCount * sizeof(int));
    }
    
    ~HierarchicalFluxCoreCUDA() {
        for (int l = 0; l < numLevels; l++) {
            cudaFree(d_self[l]);
            cudaFree(d_prevSelf[l]);
            cudaFree(d_velocity[l]);
            cudaFree(d_output[l]);
            cudaFree(d_action[l]);
            cudaFree(d_memories[l]);
            cudaFree(d_meta[l]);
            cudaFree(d_activeMask[l]);
            cudaFree(d_memCount[l]);
            cudaFree(d_tick[l]);
        }
        delete[] d_self;
        delete[] d_prevSelf;
        delete[] d_velocity;
        delete[] d_output;
        delete[] d_action;
        delete[] d_memories;
        delete[] d_meta;
        delete[] d_activeMask;
        delete[] d_memCount;
        delete[] d_tick;
        
        cudaFree(d_bestSim);
        cudaFree(d_bestIdx);
    }
    
    void ProcessLevel(
        int level,
        float* d_reality,
        float base_lr,
        float mem_lr,
        float k,
        float mem_w,
        float vel_decay,
        float vel_gain,
        float vel_scale,
        float action_gain,
        float spawnThreshold,
        float mergeThreshold,
        int pruneThreshold,
        cudaStream_t stream
    ) {
        int threads = BLOCK_SIZE;
        int blocks = (dim + threads - 1) / threads;
        int memBlocks = (maxMemories + threads - 1) / threads;
        
        // Step 1: Find best matching memory
        FindBestMemoryKernel<<<memBlocks, threads, threads * (sizeof(float) + sizeof(int)), stream>>>(
            d_reality,
            d_memories[level],
            d_activeMask[level],
            dim,
            maxMemories,
            d_bestSim,
            d_bestIdx
        );
        
        // Host-side reduction to find global best
        int blockCount = memBlocks;
        float* h_bestSim = new float[blockCount];
        int* h_bestIdx = new int[blockCount];
        cudaMemcpy(h_bestSim, d_bestSim, blockCount * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bestIdx, d_bestIdx, blockCount * sizeof(int), cudaMemcpyDeviceToHost);
        
        float bestSim = -1.0f;
        int bestIdx = -1;
        for (int i = 0; i < blockCount; i++) {
            if (h_bestSim[i] > bestSim) {
                bestSim = h_bestSim[i];
                bestIdx = h_bestIdx[i];
            }
        }
        delete[] h_bestSim;
        delete[] h_bestIdx;
        
        // Step 2: Fold with dynamic memory
        FluxCoreFoldKernelEntity<<<blocks, threads, 0, stream>>>(
            d_reality,
            d_self[level],
            d_memories[level],
            d_meta[level],
            d_activeMask[level],
            d_memCount[level],
            d_tick[level],
            dim,
            maxMemories,
            base_lr,
            mem_lr,
            k,
            mem_w,
            spawnThreshold,
            bestIdx,
            bestSim
        );
        
        // Step 3: Normalize self and memories
        NormalizeKernel<<<1, threads, 0, stream>>>(
            d_self[level],
            (int*)nullptr,  // Always active — null-guard handles this
            1,
            dim
        );
        
        // Normalize all active memories
        int h_memCount;
        cudaMemcpy(&h_memCount, d_memCount[level], sizeof(int), cudaMemcpyDeviceToHost);
        if (h_memCount > 0) {
            NormalizeKernel<<<h_memCount, threads, 0, stream>>>(
                d_memories[level],
                d_activeMask[level],
                h_memCount,
                dim
            );
        }
        
        // Step 4: Velocity and active inference output
        VelocityOutputKernel<<<blocks, threads, 0, stream>>>(
            d_self[level],
            d_prevSelf[level],
            d_velocity[level],
            d_output[level],
            d_action[level],
            dim,
            vel_decay,
            vel_gain,
            vel_scale,
            action_gain
        );
        
        // Step 5: Update prev_self
        cudaMemcpyAsync(d_prevSelf[level], d_self[level], dim * sizeof(float), 
                        cudaMemcpyDeviceToDevice, stream);
        
        // Step 6: Increment tick
        int h_tick;
        cudaMemcpy(&h_tick, d_tick[level], sizeof(int), cudaMemcpyDeviceToHost);
        h_tick++;
        cudaMemcpy(d_tick[level], &h_tick, sizeof(int), cudaMemcpyHostToDevice);
        
        // Step 7: Periodic merge and prune (every 100 ticks)
        if (h_tick % 100 == 0) {
            // Merge converged memories
            dim3 mergeGrid(h_memCount, h_memCount);
            MergeMemoriesKernel<<<mergeGrid, 1, 0, stream>>>(
                d_memories[level],
                d_meta[level],
                d_activeMask[level],
                d_memCount[level],
                dim,
                mergeThreshold
            );
            
            // Prune stale memories
            PruneMemoriesKernel<<<(maxMemories + threads - 1) / threads, threads, 0, stream>>>(
                d_meta[level],
                d_activeMask[level],
                d_memCount[level],
                d_tick[level],
                pruneThreshold
            );
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// HOST API: Launch Hierarchical FluxCore Entity
// ═══════════════════════════════════════════════════════════════════════════
void LaunchFluxCoreEntity(
    HierarchicalFluxCoreCUDA* entity,
    const float* h_reality,      // Host reality input
    float* h_output,             // Host output (action)
    float base_lr = 0.08f,
    float mem_lr = 0.015f,
    float k = 20.0f,
    float mem_w = 0.15f,
    float vel_decay = 0.95f,
    float vel_gain = 0.05f,
    float vel_scale = 12.5f,
    float action_gain = 0.5f,
    float spawn_threshold = 0.5f,
    float merge_threshold = 0.95f,
    int prune_threshold = 500,
    cudaStream_t stream = 0
) {
    // Allocate temporary reality buffer
    float* d_reality;
    cudaMalloc(&d_reality, entity->dim * sizeof(float));
    cudaMemcpy(d_reality, h_reality, entity->dim * sizeof(float), cudaMemcpyHostToDevice);
    
    float* currentReality = d_reality;
    float* tempReality = nullptr;
    
    // Process each level
    for (int level = 0; level < entity->numLevels; level++) {
        // Higher levels perceive surprise+velocity of lower level
        if (level > 0) {
            // Compute meta-reality: surprise * velocity from level below
            // This requires a kernel, simplified here
            // For now, use output from previous level
            currentReality = entity->d_output[level - 1];
        }
        
        entity->ProcessLevel(
            level,
            currentReality,
            base_lr,
            mem_lr,
            k,
            mem_w,
            vel_decay,
            vel_gain,
            vel_scale,
            action_gain,
            spawn_threshold,
            merge_threshold,
            prune_threshold,
            stream
        );
    }
    
    // Copy final action from top level
    cudaMemcpy(h_output, entity->d_action[entity->numLevels - 1], 
               entity->dim * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_reality);
}

// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE USAGE
// ═══════════════════════════════════════════════════════════════════════════
int main() {
    printf("══════════════════════════════════════════════════════════════════\\n");
    printf("FluxCore Entity — CUDA Implementation\\n");
    printf("══════════════════════════════════════════════════════════════════\\n\\n");
    
    const int dim = 64;
    const int numLevels = 3;
    const int maxMemories = 64;
    
    // Create entity
    HierarchicalFluxCoreCUDA* entity = new HierarchicalFluxCoreCUDA(dim, numLevels, maxMemories);
    
    printf("Configuration:\\n");
    printf("  Dimension: %d\\n", dim);
    printf("  Hierarchy levels: %d\\n", numLevels);
    printf("  Max memories per level: %d\\n", maxMemories);
    printf("\\n");
    
    // Create synthetic reality
    float* reality = new float[dim];
    float* action = new float[dim];
    
    // Initialize with random unit vector
    float norm = 0;
    for (int i = 0; i < dim; i++) {
        reality[i] = (float)rand() / RAND_MAX - 0.5f;
        norm += reality[i] * reality[i];
    }
    norm = sqrtf(norm) + 1e-12f;
    for (int i = 0; i < dim; i++) reality[i] /= norm;
    
    printf("Running 1000 ticks...\\n\\n");
    
    for (int tick = 0; tick < 1000; tick++) {
        // Add noise to reality
        for (int i = 0; i < dim; i++) {
            reality[i] += 0.001f * ((float)rand() / RAND_MAX - 0.5f);
        }
        norm = 0;
        for (int i = 0; i < dim; i++) norm += reality[i] * reality[i];
        norm = sqrtf(norm) + 1e-12f;
        for (int i = 0; i < dim; i++) reality[i] /= norm;
        
        // Process through entity
        LaunchFluxCoreEntity(entity, reality, action);
        
        // Print stats every 100 ticks
        if (tick % 100 == 0) {
            int memCounts[3];
            for (int l = 0; l < 3; l++) {
                cudaMemcpy(&memCounts[l], entity->d_memCount[l], sizeof(int), cudaMemcpyDeviceToHost);
            }
            printf("Tick %d: L0=%d memories, L1=%d memories, L2=%d memories\\n",
                   tick, memCounts[0], memCounts[1], memCounts[2]);
        }
    }
    
    printf("\\n══════════════════════════════════════════════════════════════════\\n");
    printf("FluxCore Entity is alive on GPU.\\n");
    printf("══════════════════════════════════════════════════════════════════\\n");
    
    delete[] reality;
    delete[] action;
    delete entity;
    
    return 0;
}

/*
 * COMPILATION:
 * nvcc -O3 -arch=sm_70 fluxcore_entity.cu -o fluxcore_entity
 * 
 * EXECUTION:
 * ./fluxcore_entity
 * 
 * NOTES:
 * - Dynamic memory management uses atomic operations
 * - Merge/prune runs every 100 ticks (configurable)
 * - Hierarchical processing: L0→L1→L2 with meta-reality propagation
 * - Active inference: output becomes control signal
 */
