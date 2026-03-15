/**
 * FluxCore Entity — CUDA Implementation (Compilable Version)
 * 
 * nvcc -O3 -arch=sm_70 fluxcore_entity_impl.cu -o fluxcore_entity
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 256
#define MAX_MEMORIES 64

// ═══════════════════════════════════════════════════════════════════════════
// DEVICE UTILITIES
// ═══════════════════════════════════════════════════════════════════════════
__device__ float d_dot(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        sum += a[i] * b[i];
    }
    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    // Bug 4 fix: inter-warp reduction via shared memory
    __shared__ float warpSums[32];
    int warpId = threadIdx.x / 32;
    int lane   = threadIdx.x & 31;
    if (lane == 0) warpSums[warpId] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        int numWarps = (blockDim.x + 31) / 32;
        for (int w = 0; w < numWarps; w++) total += warpSums[w];
        warpSums[0] = total;
    }
    __syncthreads();
    return warpSums[0];
}

__device__ void d_normalize(float* v, int dim) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        sum += v[i] * v[i];
    }
    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    // Bug 4 fix: inter-warp reduction via shared memory
    __shared__ float warpSums[32];
    int warpId = threadIdx.x / 32;
    int lane   = threadIdx.x & 31;
    if (lane == 0) warpSums[warpId] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        int numWarps = (blockDim.x + 31) / 32;
        for (int w = 0; w < numWarps; w++) total += warpSums[w];
        warpSums[0] = total;
    }
    __syncthreads();
    sum = warpSums[0];

    float norm = sqrtf(sum) + 1e-12f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        v[i] /= norm;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: FluxCore Fold with Dynamic Memory
// ═══════════════════════════════════════════════════════════════════════════
__global__ void FluxCoreFoldKernel(
    const float* __restrict__ reality,
    float* __restrict__ self_state,
    float* __restrict__ memories,
    float* __restrict__ meta,
    int* __restrict__ active,
    int* __restrict__ memCount,
    int* __restrict__ tick,
    int dim,
    int maxMemories,
    float base_lr,
    float mem_lr,
    float k,
    float mem_w,
    float spawnThreshold,
    float* surpriseOut
) {
    // Single-block strided kernel (launched <<<1, BLOCK_SIZE>>>)
    // Each thread handles indices: threadIdx.x, threadIdx.x+blockDim.x, ...

    // Step 1: Compute per-thread surprise (strided sum)
    __shared__ float warpSums[32];
    float localSurp = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        localSurp += fabsf(self_state[i] - reality[i]);
    }
    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2)
        localSurp += __shfl_down_sync(0xffffffff, localSurp, offset);
    // Inter-warp reduction
    int warpId = threadIdx.x / 32;
    int lane   = threadIdx.x & 31;
    if (lane == 0) warpSums[warpId] = localSurp;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        int numWarps = (blockDim.x + 31) / 32;
        for (int w = 0; w < numWarps; w++) total += warpSums[w];
        warpSums[0] = total;
    }
    __syncthreads();
    float surprise = warpSums[0] / dim;

    if (threadIdx.x == 0 && surpriseOut) {
        *surpriseOut = surprise;
    }

    // Step 2: Find best memory (thread 0 does this serially)
    __shared__ int bestIdx;
    __shared__ float bestSim;

    if (threadIdx.x == 0) {
        bestIdx = -1;
        bestSim = -1.0f;

        int h_memCount = *memCount;
        for (int m = 0; m < h_memCount && m < maxMemories; m++) {
            if (active[m]) {
                float sim = 0.0f;
                for (int i = 0; i < dim; i++) {
                    sim += memories[m * dim + i] * reality[i];
                }
                sim = fabsf(sim);
                if (sim > bestSim) {
                    bestSim = sim;
                    bestIdx = m;
                }
            }
        }

        // Spawn new memory if needed
        if (bestSim < spawnThreshold) {
            int newIdx = atomicAdd(memCount, 1);
            if (newIdx < maxMemories) {
                active[newIdx] = 1;
                meta[newIdx * 3 + 0] = (float)(*tick);
                meta[newIdx * 3 + 1] = 1.0f;
                meta[newIdx * 3 + 2] = (float)(*tick);
                for (int i = 0; i < dim; i++) {
                    memories[newIdx * dim + i] = reality[i];
                }
                bestIdx = newIdx;
                bestSim = 1.0f;
            }
        } else if (bestIdx >= 0) {
            meta[bestIdx * 3 + 0] = (float)(*tick);
            meta[bestIdx * 3 + 1] += 1.0f;
        }
    }

    __syncthreads();

    // Step 3: Fold — strided, read neighbors directly from global self_state
    // self_state hasn't been written yet this tick, so reads are consistent
    float alr = base_lr * (1.0f + k * surprise);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float s = self_state[i];
        float r = reality[i];
        float m = (bestIdx >= 0) ? memories[bestIdx * dim + i] : r;

        // Gradient: read neighbors from global memory (safe — no writes yet)
        float left  = self_state[(i + dim - 1) % dim];
        float right = self_state[(i + 1) % dim];
        float grad  = (s - left) - (s - right);

        float d = fabsf(s - r);
        float u = s + alr * r + (alr * 0.5f) * d * grad + mem_w * m;
        self_state[i] = u;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: Velocity and Active Inference Output
// ═══════════════════════════════════════════════════════════════════════════
__global__ void VelocityOutputKernel(
    float* __restrict__ self_state,
    float* __restrict__ prev_self,
    float* __restrict__ velocity,
    float* __restrict__ output,
    float* __restrict__ action,
    int dim,
    float vel_decay,
    float vel_gain,
    float vel_scale,
    float action_gain
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    
    float s = self_state[idx];
    float p = prev_self[idx];
    float v = velocity[idx];
    
    // Update velocity
    v = vel_decay * v + vel_gain * (s - p);
    velocity[idx] = v;
    
    // Project output
    float o = s + vel_scale * v;
    output[idx] = o;
    
    // ACTIVE INFERENCE: Generate action
    // Action pushes environment toward self (minimizing future surprise)
    action[idx] = action_gain * (s - o);
    
    // Update prev_self for next tick
    prev_self[idx] = s;
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: Normalize Self State
// ═══════════════════════════════════════════════════════════════════════════
__global__ void NormalizeKernel(float* vec, int dim) {
    __shared__ float warpSums[32];

    float localSum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        localSum += vec[i] * vec[i];
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        localSum += __shfl_down_sync(0xffffffff, localSum, offset);
    }

    // Inter-warp reduction
    int warpId = threadIdx.x / 32;
    int lane   = threadIdx.x & 31;
    if (lane == 0) warpSums[warpId] = localSum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        int numWarps = (blockDim.x + 31) / 32;
        for (int w = 0; w < numWarps; w++) total += warpSums[w];
        warpSums[0] = total;
    }
    __syncthreads();

    float norm = sqrtf(warpSums[0]) + 1e-12f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        vec[i] /= norm;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: Prune Stale Memories
// ═══════════════════════════════════════════════════════════════════════════
__global__ void PruneKernel(
    float* meta,
    int* active,
    int* memCount,
    int* tick,
    int maxMemories,
    int pruneThreshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= maxMemories) return;
    
    if (active[idx]) {
        int lastUsed = (int)meta[idx * 3 + 0];
        int currentTick = *tick;
        if (currentTick - lastUsed > pruneThreshold) {
            active[idx] = 0;
            atomicSub(memCount, 1);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HOST: FluxCore Entity Class
// ═══════════════════════════════════════════════════════════════════════════
class FluxCoreEntity {
public:
    int dim;
    int maxMemories;
    
    // State
    float* d_self;
    float* d_prev;
    float* d_velocity;
    float* d_output;
    float* d_action;
    
    // Dynamic memory field
    float* d_memories;
    float* d_meta;
    int* d_active;
    int* d_memCount;
    int* d_tick;
    
    // Temp
    float* d_surprise;
    
    // Parameters
    float spawnThreshold;
    float mergeThreshold;
    int pruneThreshold;
    
    FluxCoreEntity(int d, int maxMem = 64, 
                   float spawnT = 0.5f, 
                   float mergeT = 0.95f, 
                   int pruneT = 500) {
        dim = d;
        maxMemories = maxMem;
        spawnThreshold = spawnT;
        mergeThreshold = mergeT;
        pruneThreshold = pruneT;
        
        // Allocate state
        cudaMalloc(&d_self, dim * sizeof(float));
        cudaMalloc(&d_prev, dim * sizeof(float));
        cudaMalloc(&d_velocity, dim * sizeof(float));
        cudaMalloc(&d_output, dim * sizeof(float));
        cudaMalloc(&d_action, dim * sizeof(float));
        
        // Allocate memory field
        cudaMalloc(&d_memories, maxMemories * dim * sizeof(float));
        cudaMalloc(&d_meta, maxMemories * 3 * sizeof(float));
        cudaMalloc(&d_active, maxMemories * sizeof(int));
        cudaMalloc(&d_memCount, sizeof(int));
        cudaMalloc(&d_tick, sizeof(int));
        cudaMalloc(&d_surprise, sizeof(float));
        
        // Initialize
        cudaMemset(d_velocity, 0, dim * sizeof(float));
        cudaMemset(d_active, 0, maxMemories * sizeof(int));
        cudaMemset(d_memCount, 0, sizeof(int));
        cudaMemset(d_tick, 0, sizeof(int));
        
        // Random init self
        float* h_self = new float[dim];
        for (int i = 0; i < dim; i++) {
            h_self[i] = (float)rand() / RAND_MAX - 0.5f;
        }
        float norm = 0;
        for (int i = 0; i < dim; i++) norm += h_self[i] * h_self[i];
        norm = sqrtf(norm) + 1e-12f;
        for (int i = 0; i < dim; i++) h_self[i] /= norm;
        
        cudaMemcpy(d_self, h_self, dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_prev, h_self, dim * sizeof(float), cudaMemcpyHostToDevice);
        delete[] h_self;
    }
    
    ~FluxCoreEntity() {
        cudaFree(d_self);
        cudaFree(d_prev);
        cudaFree(d_velocity);
        cudaFree(d_output);
        cudaFree(d_action);
        cudaFree(d_memories);
        cudaFree(d_meta);
        cudaFree(d_active);
        cudaFree(d_memCount);
        cudaFree(d_tick);
        cudaFree(d_surprise);
    }
    
    void Forward(const float* h_reality, float* h_action, float* h_surprise,
                 float base_lr = 0.08f,
                 float mem_lr = 0.015f,
                 float k = 20.0f,
                 float mem_w = 0.15f,
                 float vel_decay = 0.95f,
                 float vel_gain = 0.05f,
                 float vel_scale = 12.5f,
                 float action_gain = 0.5f) {
        
        int threads = BLOCK_SIZE;
        int blocks = (dim + threads - 1) / threads;

        // Copy reality to device
        float* d_reality;
        cudaMalloc(&d_reality, dim * sizeof(float));
        cudaMemcpy(d_reality, h_reality, dim * sizeof(float), cudaMemcpyHostToDevice);

        // Step 1: Fold — single block with striding so intra-block reduction is correct
        FluxCoreFoldKernel<<<1, threads>>>(
            d_reality,
            d_self,
            d_memories,
            d_meta,
            d_active,
            d_memCount,
            d_tick,
            dim,
            maxMemories,
            base_lr,
            mem_lr,
            k,
            mem_w,
            spawnThreshold,
            d_surprise
        );
        
        // Step 2: Normalize self
        NormalizeKernel<<<1, threads>>>(d_self, dim);
        
        // Step 3: Velocity and output
        VelocityOutputKernel<<<blocks, threads>>>(
            d_self, d_prev, d_velocity, d_output, d_action,
            dim, vel_decay, vel_gain, vel_scale, action_gain
        );
        
        // Step 4: Increment tick
        int h_tick;
        cudaMemcpy(&h_tick, d_tick, sizeof(int), cudaMemcpyDeviceToHost);
        h_tick++;
        cudaMemcpy(d_tick, &h_tick, sizeof(int), cudaMemcpyHostToDevice);
        
        // Step 5: Periodic prune (every 100 ticks)
        if (h_tick % 100 == 0) {
            PruneKernel<<<(maxMemories + threads - 1) / threads, threads>>>(
                d_meta, d_active, d_memCount, d_tick, maxMemories, pruneThreshold
            );
        }
        
        // Copy results back
        cudaMemcpy(h_action, d_action, dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_surprise, d_surprise, sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(d_reality);
    }
    
    int GetMemoryCount() {
        int count;
        cudaMemcpy(&count, d_memCount, sizeof(int), cudaMemcpyDeviceToHost);
        return count;
    }
    
    void GetMemoryInfo(float* ages, float* useCounts, int maxStats) {
        float* h_meta = new float[maxMemories * 3];
        int* h_active = new int[maxMemories];
        int h_tick;
        
        cudaMemcpy(h_meta, d_meta, maxMemories * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_active, d_active, maxMemories * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_tick, d_tick, sizeof(int), cudaMemcpyDeviceToHost);
        
        int idx = 0;
        for (int i = 0; i < maxMemories && idx < maxStats; i++) {
            if (h_active[i]) {
                ages[idx] = h_tick - h_meta[i * 3 + 2];
                useCounts[idx] = h_meta[i * 3 + 1];
                idx++;
            }
        }
        
        delete[] h_meta;
        delete[] h_active;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// MAIN: Demonstration
// ═══════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("  FluxCore Entity — CUDA Implementation\n");
    printf("══════════════════════════════════════════════════════════════════\n\n");

    // Check CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("ERROR: No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    // Configuration — DIM from argv[1], default 64
    const int dim = (argc > 1) ? atoi(argv[1]) : 64;
    const int maxMemories = 64;
    const int numTicks = 2000;
    
    printf("Configuration:\n");
    printf("  Dimension: %d\n", dim);
    printf("  Max memories: %d\n", maxMemories);
    printf("  Ticks: %d\n\n", numTicks);
    
    // Create entity
    FluxCoreEntity* entity = new FluxCoreEntity(dim, maxMemories, 0.5f, 0.95f, 500);
    
    // Create test distributions (3 orthogonal)
    float** distributions = new float*[3];
    for (int i = 0; i < 3; i++) {
        distributions[i] = new float[dim];
        for (int j = 0; j < dim; j++) {
            distributions[i][j] = (float)rand() / RAND_MAX - 0.5f;
        }
        // Normalize
        float norm = 0;
        for (int j = 0; j < dim; j++) norm += distributions[i][j] * distributions[i][j];
        norm = sqrtf(norm) + 1e-12f;
        for (int j = 0; j < dim; j++) distributions[i][j] /= norm;
        
        // Make orthogonal to previous
        for (int k = 0; k < i; k++) {
            float dot = 0;
            for (int j = 0; j < dim; j++) dot += distributions[i][j] * distributions[k][j];
            for (int j = 0; j < dim; j++) distributions[i][j] -= dot * distributions[k][j];
        }
        // Renormalize
        norm = 0;
        for (int j = 0; j < dim; j++) norm += distributions[i][j] * distributions[i][j];
        norm = sqrtf(norm) + 1e-12f;
        for (int j = 0; j < dim; j++) distributions[i][j] /= norm;
    }
    
    // Buffers
    float* reality = new float[dim];
    float* action = new float[dim];
    float surprise;
    
    printf("Running simulation with 3 distribution switches...\n\n");
    
    int currentDist = 0;
    int switchInterval = numTicks / 3;
    
    for (int tick = 0; tick < numTicks; tick++) {
        // Switch distribution
        if (tick > 0 && tick % switchInterval == 0) {
            currentDist = (currentDist + 1) % 3;
            printf("  [Switch] Now using distribution %d\n", currentDist);
        }
        
        // Generate noisy reality
        for (int i = 0; i < dim; i++) {
            reality[i] = distributions[currentDist][i] + 0.001f * ((float)rand() / RAND_MAX - 0.5f);
        }
        // Normalize
        float norm = 0;
        for (int i = 0; i < dim; i++) norm += reality[i] * reality[i];
        norm = sqrtf(norm) + 1e-12f;
        for (int i = 0; i < dim; i++) reality[i] /= norm;
        
        // Forward pass
        entity->Forward(reality, action, &surprise);
        
        // Print stats
        if (tick % 250 == 0) {
            int memCount = entity->GetMemoryCount();
            printf("  Tick %4d: %d memories, surprise=%.4f\n", tick, memCount, surprise);
        }
    }
    
    // Final stats
    printf("\n══════════════════════════════════════════════════════════════════\n");
    printf("FINAL STATE\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    
    int finalCount = entity->GetMemoryCount();
    printf("Total memories: %d\n\n", finalCount);
    
    float* ages = new float[maxMemories];
    float* useCounts = new float[maxMemories];
    entity->GetMemoryInfo(ages, useCounts, maxMemories);
    
    printf("Memory Details:\n");
    for (int i = 0; i < finalCount && i < 10; i++) {
        printf("  Memory %d: age=%.0f ticks, uses=%.0f\n", i, ages[i], useCounts[i]);
    }
    if (finalCount > 10) {
        printf("  ... and %d more\n", finalCount - 10);
    }
    
    printf("\n══════════════════════════════════════════════════════════════════\n");
    printf("FluxCore Entity is alive on GPU.\n");
    printf("  • Dynamic attractor genesis: %s\n", finalCount > 1 ? "ACTIVE" : "INACTIVE");
    printf("  • Active inference: ACTIVE (action generated)\n");
    printf("  • Self-organization: Memories spawned on novelty\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    
    // Cleanup
    delete[] reality;
    delete[] action;
    delete[] ages;
    delete[] useCounts;
    for (int i = 0; i < 3; i++) delete[] distributions[i];
    delete[] distributions;
    delete entity;
    
    return 0;
}
