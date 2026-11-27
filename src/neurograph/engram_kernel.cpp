// NeuroGraph v2.0 Native OpenMP C++17 Engram Kernel
// Ultra-fast CPU/mmap Context-Gated External Knowledge Store (Zero GPU VRAM!)
// Compile: g++ -O3 -march=native -fopenmp -shared -fPIC engram_kernel.cpp -o libengram.so

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <omp.h>

extern "C" {

static constexpr int ENGRAM_WAYS = 4;
static constexpr int ENGRAM_CDIM = 16;

// Lock-free multi-threaded bucket-sharded ingestion of one Engram bank!
// keys: [N] int64, targets: [N] uint8, c_q: [N, 16] int8
// sig: [num_buckets, 4] int32, dst: [num_buckets, 4] uint8, w: [num_buckets, 4] uint16, c_vec: [num_buckets, 4, 16] int8
void engram_ingest_bank(
    const int64_t* __restrict__ keys,
    const uint8_t* __restrict__ targets,
    const int8_t*  __restrict__ c_q,
    int64_t N,
    int64_t num_buckets,
    int32_t* __restrict__ sig,
    uint8_t* __restrict__ dst,
    uint16_t* __restrict__ w,
    int8_t*  __restrict__ c_vec
) {
    const uint64_t mask = static_cast<uint64_t>(num_buckets - 1);

    #pragma omp parallel
    {
        const uint32_t nt  = static_cast<uint32_t>(omp_get_num_threads());
        const uint32_t tid = static_cast<uint32_t>(omp_get_thread_num());

        for (int64_t i = 0; i < N; ++i) {
            const uint64_t k = static_cast<uint64_t>(keys[i]);
            if (k == 0) continue;
            const uint32_t bkt = static_cast<uint32_t>(k & mask);
            if ((bkt % nt) != tid) continue;

            int32_t tag = static_cast<int32_t>((k >> 24) & 0x7FFFFFFFULL);
            if (tag == 0) tag = 1;
            const uint8_t y = targets[i];
            const int8_t* cq = c_q + i * ENGRAM_CDIM;

            const int64_t base = static_cast<int64_t>(bkt) * ENGRAM_WAYS;
            int match_j = -1;
            int emp_j = -1;
            int min_j = 0;
            uint16_t min_w = 65535;

            for (int j = 0; j < ENGRAM_WAYS; ++j) {
                const uint16_t wj = w[base + j];
                if (wj == 0) {
                    if (emp_j < 0) emp_j = j;
                } else {
                    if (sig[base + j] == tag && dst[base + j] == y) {
                        match_j = j;
                        break;
                    }
                    if (wj < min_w) {
                        min_w = wj;
                        min_j = j;
                    }
                }
            }

            if (match_j >= 0) {
                const int64_t idx = base + match_j;
                const uint16_t nw = (w[idx] < 65535) ? static_cast<uint16_t>(w[idx] + 1) : 65535;
                w[idx] = nw;
                // Online running centroid of Context Fingerprint (zero backprop!)
                const float alpha = 1.0f / static_cast<float>(nw < 32 ? nw : 32);
                const float keep = 1.0f - alpha;
                int8_t* cv = c_vec + idx * ENGRAM_CDIM;
                for (int d = 0; d < ENGRAM_CDIM; ++d) {
                    float v = keep * static_cast<float>(cv[d]) + alpha * static_cast<float>(cq[d]);
                    int iv = static_cast<int>(std::lround(v));
                    if (iv > 127) iv = 127;
                    if (iv < -127) iv = -127;
                    cv[d] = static_cast<int8_t>(iv);
                }
            } else if (emp_j >= 0 || min_w <= 1) {
                const int target_j = (emp_j >= 0) ? emp_j : min_j;
                const int64_t idx = base + target_j;
                sig[idx] = tag;
                dst[idx] = y;
                w[idx] = 1;
                std::memcpy(c_vec + idx * ENGRAM_CDIM, cq, ENGRAM_CDIM);
            }
        }
    }
}

// Fast OpenMP Sleep Pruning: clears unverified 1-hit noise slots (w < min_keep_w)
// Returns number of surviving active slots.
int64_t engram_sleep_prune_bank(
    int64_t num_buckets,
    uint16_t min_keep_w,
    int32_t* __restrict__ sig,
    uint8_t* __restrict__ dst,
    uint16_t* __restrict__ w,
    int8_t*  __restrict__ c_vec
) {
    const int64_t total = num_buckets * ENGRAM_WAYS;
    int64_t active = 0;

    #pragma omp parallel for reduction(+:active) schedule(static)
    for (int64_t idx = 0; idx < total; ++idx) {
        if (w[idx] == 0) continue;
        if (w[idx] < min_keep_w) {
            sig[idx] = 0;
            dst[idx] = 0;
            w[idx] = 0;
            std::memset(c_vec + idx * ENGRAM_CDIM, 0, ENGRAM_CDIM);
        } else {
            active += 1;
        }
    }
    return active;
}

// Count active slots in one bank
int64_t engram_count_active_bank(int64_t num_buckets, const uint16_t* __restrict__ w) {
    const int64_t total = num_buckets * ENGRAM_WAYS;
    int64_t active = 0;
    #pragma omp parallel for reduction(+:active) schedule(static)
    for (int64_t idx = 0; idx < total; ++idx) {
        if (w[idx] > 0) active += 1;
    }
    return active;
}

// Deterministic O(1) Prefetch Lookup for one bank across M queries!
// Accumulates into out_boost[M, 256] (caller zeroes out_boost before first bank/cartridge).
// Returns total number of queries that hit this bank (via out_hits) and sums gate values (via out_gate_sum).
void engram_lookup_bank(
    const int64_t* __restrict__ keys,
    const int8_t*  __restrict__ c_q,
    int64_t M,
    int64_t num_buckets,
    uint16_t min_infer_w,
    float scale,
    int use_context_gate,
    float gate_alpha,
    float gate_tau,
    float gate_min,
    const int32_t*  __restrict__ sig,
    const uint8_t*  __restrict__ dst,
    const uint16_t* __restrict__ w,
    const int8_t*   __restrict__ c_vec,
    float* __restrict__ out_boost,
    int64_t* out_hits,
    double* out_gate_sum
) {
    const uint64_t mask = static_cast<uint64_t>(num_buckets - 1);
    int64_t hits_acc = 0;
    double gate_acc = 0.0;

    #pragma omp parallel for if(M >= 64) reduction(+:hits_acc,gate_acc) schedule(static)
    for (int64_t i = 0; i < M; ++i) {
        const uint64_t k = static_cast<uint64_t>(keys[i]);
        if (k == 0) continue;
        const uint32_t bkt = static_cast<uint32_t>(k & mask);
        int32_t tag = static_cast<int32_t>((k >> 24) & 0x7FFFFFFFULL);
        if (tag == 0) tag = 1;

        const int64_t base = static_cast<int64_t>(bkt) * ENGRAM_WAYS;

        float w_tot = 0.0f;
        int hit_mask = 0;
        for (int j = 0; j < ENGRAM_WAYS; ++j) {
            const uint16_t wj = w[base + j];
            if (wj >= min_infer_w && sig[base + j] == tag) {
                w_tot += static_cast<float>(wj);
                hit_mask |= (1 << j);
            }
        }
        if (hit_mask == 0) continue;
        hits_acc += 1;

        const int8_t* cq = c_q + i * ENGRAM_CDIM;
        int32_t nq = 0;
        if (use_context_gate) {
            for (int d = 0; d < ENGRAM_CDIM; ++d) {
                int32_t v = static_cast<int32_t>(cq[d]);
                nq += v * v;
            }
        }

        float* ob = out_boost + i * 256;
        for (int j = 0; j < ENGRAM_WAYS; ++j) {
            if (!(hit_mask & (1 << j))) continue;
            const int64_t idx = base + j;
            const float wj = static_cast<float>(w[idx]);
            const float ev = std::tanh(std::log1p(wj) * 0.35f) * (wj / (w_tot + 2.0f));

            float g = 1.0f;
            if (use_context_gate && nq > 0) {
                const int8_t* cv = c_vec + idx * ENGRAM_CDIM;
                int32_t dot = 0;
                int32_t nc = 0;
                for (int d = 0; d < ENGRAM_CDIM; ++d) {
                    int32_t a = static_cast<int32_t>(cv[d]);
                    int32_t b = static_cast<int32_t>(cq[d]);
                    dot += a * b;
                    nc += a * a;
                }
                const float cos_sim = (nc > 0) ? (static_cast<float>(dot) / std::sqrt(static_cast<float>(nc) * static_cast<float>(nq))) : 0.0f;
                const float sg = 1.0f / (1.0f + std::exp(-gate_alpha * (cos_sim - gate_tau)));
                g = gate_min + (1.0f - gate_min) * sg;
            }
            gate_acc += static_cast<double>(g);
            ob[dst[idx]] += scale * ev * g;
        }
    }

    if (out_hits) *out_hits = hits_acc;
    if (out_gate_sum) *out_gate_sum = gate_acc;
}

static inline uint64_t engram_mix64(uint64_t h) {
    h ^= (h >> 30);
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= (h >> 27);
    h *= 0x94d049bb133111ebULL;
    h ^= (h >> 31);
    return h;
}

// Ultra-fast CPU 6-scale raw UTF-8 rolling hash (bit-exact with EngramContextEncoder!)
// bs24: [B, 24] int32 (-1 where out of history, 0..255 otherwise)
// out_keys: [6, B] int64 (spans 6, 9, 12, 15, 18, 24; 0 where prefix < span)
void engram_hash_bytes24_batch(
    const int32_t* __restrict__ bs24,
    int64_t B,
    int64_t* __restrict__ out_keys
) {
    static constexpr uint64_t PRIMES[24] = {
        17ULL, 1315423911ULL, 2654435761ULL, 1000003ULL, 1000033ULL, 1000037ULL,
        1000039ULL, 1000081ULL, 1000099ULL, 1000117ULL, 1000121ULL, 1000133ULL,
        1000151ULL, 1000159ULL, 1000171ULL, 1000183ULL, 1000187ULL, 1000193ULL,
        1000211ULL, 1000213ULL, 1000231ULL, 1000249ULL, 1000253ULL, 1000273ULL
    };
    for (int64_t i = 0; i < B; ++i) {
        const int32_t* row = bs24 + i * 24;
        uint64_t b0 = (row[0] >= 0) ? static_cast<uint64_t>(row[0]) : 0ULL;
        uint64_t h = engram_mix64(b0 + PRIMES[0]);
        for (int d = 1; d < 24; ++d) {
            int32_t rv = row[d];
            uint64_t bd = (rv >= 0) ? static_cast<uint64_t>(rv) : 0ULL;
            h = engram_mix64(h ^ ((bd + static_cast<uint64_t>(31 * d)) * PRIMES[d]));
            if (d == 5)       out_keys[0 * B + i] = (rv >= 0 && h != 0) ? static_cast<int64_t>(h) : 0;
            else if (d == 8)  out_keys[1 * B + i] = (rv >= 0 && h != 0) ? static_cast<int64_t>(h) : 0;
            else if (d == 11) out_keys[2 * B + i] = (rv >= 0 && h != 0) ? static_cast<int64_t>(h) : 0;
            else if (d == 14) out_keys[3 * B + i] = (rv >= 0 && h != 0) ? static_cast<int64_t>(h) : 0;
            else if (d == 17) out_keys[4 * B + i] = (rv >= 0 && h != 0) ? static_cast<int64_t>(h) : 0;
            else if (d == 23) out_keys[5 * B + i] = (rv >= 0 && h != 0) ? static_cast<int64_t>(h) : 0;
        }
    }
}

} // extern "C"
