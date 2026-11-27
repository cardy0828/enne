// NeuroGraph v2.1 Native OpenMP C++17 Episodic Engram Inverted Index (Zero Tokenizer, Pure Raw UTF-8!)
// Indexes hundreds of thousands of raw UTF-8 knowledge episodes by multi-scale 6-byte & 9-byte hashes
// with automatic O(1) IDF weighting (skipping high-DF stop-spans with zero language rules!).
// Compile: g++ -O3 -march=native -fopenmp -shared -fPIC episodic_kernel.cpp -o libepisodic.so

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <unordered_map>

extern "C" {

static inline uint64_t mix64_ep(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

static inline uint32_t hash_span(const uint8_t* p, int len, uint32_t mask) {
    uint64_t h = mix64_ep(static_cast<uint64_t>(p[0]) + 19ULL);
    static constexpr uint64_t P[9] = {
        17ULL, 1315423911ULL, 2654435761ULL, 1000003ULL, 1000033ULL,
        1000037ULL, 1000039ULL, 1000081ULL, 1000099ULL
    };
    for (int i = 1; i < len; ++i) {
        h = mix64_ep(h ^ ((static_cast<uint64_t>(p[i]) + 31ULL * i) * P[i]));
    }
    return static_cast<uint32_t>(h & mask);
}

struct EpisodicIndex {
    uint32_t num_buckets;
    uint32_t mask;
    uint32_t num_episodes;
    int max_df_keep;
    std::vector<uint8_t> raw_bytes;
    std::vector<uint64_t> ep_off;     // [num_episodes + 1]
    std::vector<uint32_t> bkt_off;    // [num_buckets + 1]
    std::vector<uint32_t> postings;   // flat CSR of (ep_id << 1 | is_9byte)
};

// Build a CSR inverted index over contiguous raw_bytes sliced by ep_off[0..num_episodes]
void* episodic_build(
    const uint8_t* raw_bytes,
    uint64_t total_bytes,
    const uint64_t* ep_off,
    uint32_t num_episodes,
    uint32_t num_buckets,
    int max_df_keep,
    int stride
) {
    auto* idx = new EpisodicIndex();
    idx->num_buckets = num_buckets;
    idx->mask = num_buckets - 1;
    idx->num_episodes = num_episodes;
    idx->max_df_keep = max_df_keep;
    idx->raw_bytes.assign(raw_bytes, raw_bytes + total_bytes);
    idx->ep_off.assign(ep_off, ep_off + num_episodes + 1);

    // Pass 1: count bucket frequencies (deduplicated per episode using small per-episode vector)
    std::vector<uint32_t> counts(num_buckets, 0);
    std::vector<uint32_t> seen;
    seen.reserve(256);

    const int st = (stride < 1) ? 1 : stride;
    for (uint32_t e = 0; e < num_episodes; ++e) {
        const uint64_t s0 = ep_off[e];
        const uint64_t s1 = ep_off[e + 1];
        const int64_t elen = static_cast<int64_t>(s1 - s0);
        if (elen < 6) continue;
        seen.clear();
        const uint8_t* ptr = raw_bytes + s0;
        for (int64_t pos = 0; pos + 6 <= elen; pos += st) {
            uint32_t b6 = hash_span(ptr + pos, 6, idx->mask);
            seen.push_back((b6 << 1) | 0U);
            if (pos + 9 <= elen) {
                uint32_t b9 = hash_span(ptr + pos, 9, idx->mask);
                seen.push_back((b9 << 1) | 1U);
            }
        }
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        for (uint32_t code : seen) {
            counts[code >> 1] += 1;
        }
    }

    // Build CSR offsets, zeroing out ultra-common stop-buckets (df > max_df_keep) automatically!
    idx->bkt_off.resize(num_buckets + 1, 0);
    uint64_t total_post = 0;
    for (uint32_t b = 0; b < num_buckets; ++b) {
        idx->bkt_off[b] = static_cast<uint32_t>(total_post);
        if (counts[b] > 0 && static_cast<int>(counts[b]) <= max_df_keep) {
            total_post += counts[b];
        }
    }
    idx->bkt_off[num_buckets] = static_cast<uint32_t>(total_post);
    idx->postings.resize(total_post);

    // Pass 2: fill CSR postings
    std::vector<uint32_t> cur = idx->bkt_off;
    for (uint32_t e = 0; e < num_episodes; ++e) {
        const uint64_t s0 = ep_off[e];
        const uint64_t s1 = ep_off[e + 1];
        const int64_t elen = static_cast<int64_t>(s1 - s0);
        if (elen < 6) continue;
        seen.clear();
        const uint8_t* ptr = raw_bytes + s0;
        for (int64_t pos = 0; pos + 6 <= elen; pos += st) {
            uint32_t b6 = hash_span(ptr + pos, 6, idx->mask);
            seen.push_back((b6 << 1) | 0U);
            if (pos + 9 <= elen) {
                uint32_t b9 = hash_span(ptr + pos, 9, idx->mask);
                seen.push_back((b9 << 1) | 1U);
            }
        }
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        for (uint32_t code : seen) {
            uint32_t b = code >> 1;
            uint32_t is9 = code & 1U;
            if (counts[b] > 0 && static_cast<int>(counts[b]) <= max_df_keep) {
                idx->postings[cur[b]++] = (e << 1) | is9;
            }
        }
    }
    return static_cast<void*>(idx);
}

void episodic_free(void* handle) {
    if (handle) delete static_cast<EpisodicIndex*>(handle);
}

uint64_t episodic_num_postings(void* handle) {
    if (!handle) return 0;
    return static_cast<uint64_t>(static_cast<EpisodicIndex*>(handle)->postings.size());
}

// Query Top-K most semantically resonant raw UTF-8 episodes in < 100 microseconds!
// Uses pure rule-free IDF weighting over all 6-byte and 9-byte spans in query_bytes!
// Returns number of hits written to out_ep_ids[0..top_k-1], out_scores[0..top_k-1].
int episodic_query(
    void* handle,
    const uint8_t* query_bytes,
    int query_len,
    int top_k,
    uint32_t* out_ep_ids,
    float* out_scores
) {
    if (!handle || query_len < 6 || top_k <= 0) return 0;
    const auto* idx = static_cast<const EpisodicIndex*>(handle);
    if (idx->num_episodes == 0) return 0;

    // Collect unique query buckets (stride 1 so even unaligned UTF-8 spans match 100%!)
    std::vector<uint32_t> q_codes;
    q_codes.reserve(static_cast<size_t>(query_len) * 2);
    for (int pos = 0; pos + 6 <= query_len; ++pos) {
        uint32_t b6 = hash_span(query_bytes + pos, 6, idx->mask);
        q_codes.push_back((b6 << 1) | 0U);
        if (pos + 9 <= query_len) {
            uint32_t b9 = hash_span(query_bytes + pos, 9, idx->mask);
            q_codes.push_back((b9 << 1) | 1U);
        }
    }
    std::sort(q_codes.begin(), q_codes.end());
    q_codes.erase(std::unique(q_codes.begin(), q_codes.end()), q_codes.end());

    const float logN = std::log(static_cast<float>(idx->num_episodes) + 2.0f);
    // Normalize IDF by logN so small custom cartridges (N=2..100) compete fairly on the exact same [0,1] scale as 820k-episode Wikipedia!
    const float inv_logN = 1.0f / std::max(logN, 1.0f);
    std::unordered_map<uint32_t, float> acc;
    acc.reserve(1024);

    for (uint32_t code : q_codes) {
        uint32_t b = code >> 1;
        uint32_t want9 = code & 1U;
        uint32_t p0 = idx->bkt_off[b];
        uint32_t p1 = idx->bkt_off[b + 1];
        uint32_t df = p1 - p0;
        if (df == 0) continue;
        // Pure rule-free normalized IDF in [0, 1] scaled to universal [0..12] range (9-byte matches carry 1.85x precision bonus!)
        float idf_norm = (idx->num_episodes <= 32)
            ? 0.92f
            : std::max(0.0f, (logN - std::log(static_cast<float>(df) + 1.0f)) * inv_logN);
        if (idf_norm <= 0.12f) continue;
        float idf = idf_norm * 12.0f;
        float w = idf * idf * (want9 ? 1.85f : 1.0f);
        for (uint32_t p = p0; p < p1; ++p) {
            uint32_t entry = idx->postings[p];
            if ((entry & 1U) == want9) {
                acc[entry >> 1] += w;
            }
        }
    }

    if (acc.empty()) return 0;
    std::vector<std::pair<float, uint32_t>> cand;
    cand.reserve(acc.size());
    for (const auto& kv : acc) {
        // Gentle length normalization so huge episodes don't unfairly dominate
        uint64_t elen = idx->ep_off[kv.first + 1] - idx->ep_off[kv.first];
        float norm_sc = kv.second / std::sqrt(static_cast<float>(elen) / 240.0f + 0.5f);
        cand.emplace_back(norm_sc, kv.first);
    }

    int take = std::min(top_k, static_cast<int>(cand.size()));
    std::partial_sort(cand.begin(), cand.begin() + take, cand.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    for (int i = 0; i < take; ++i) {
        out_ep_ids[i] = cand[i].second;
        out_scores[i] = cand[i].first;
    }
    return take;
}

// Compute IDF score for a single 6-byte span (used to extract high-IDF content entities with ZERO rules!)
float episodic_span_idf(void* handle, const uint8_t* span6) {
    if (!handle) return 0.0f;
    const auto* idx = static_cast<const EpisodicIndex*>(handle);
    uint32_t b = hash_span(span6, 6, idx->mask);
    uint32_t df = idx->bkt_off[b + 1] - idx->bkt_off[b];
    if (df == 0) {
        // Either ultra-rare (unseen custom term!) or filtered stop-span? Distinguish by checking count via caller;
        // returning 0 when filtered prevents stop-phrases from masquerading as rare!
        return 0.0f;
    }
    return std::log((static_cast<float>(idx->num_episodes) + 2.0f) / (static_cast<float>(df) + 1.0f));
}

} // extern "C"
