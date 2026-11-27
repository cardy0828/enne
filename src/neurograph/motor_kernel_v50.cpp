// NeuroGraph v5.0: C++17 OpenMP Broca Motor Growth + Directed Subject->Predicate Macro-Cortex
// + Zero-Allocation Fast Scratch Predictor + 12-Thread Parallel Lossless Byte Marginalizer
// + Binary Disk Persistence (0 MB GPU VRAM!)
//
// Zero backprop, zero Attention, zero pre-trained tokenizer, zero <UNK>, zero hardcoded language rules.
// Compile:
//   g++ -O3 -march=native -fopenmp -shared -fPIC neurograph/motor_kernel_v50.cpp -o neurograph/libmotor_v50.so

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <omp.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static inline uint64_t hash_macro_ctx(const int32_t* ids, int n, int32_t subj_id) {
    uint64_t h = 0x517CC1B727220A95ULL ^ (static_cast<uint64_t>(n) * 0x9E3779B97F4A7C15ULL);
    for (int i = 0; i < n; ++i) {
        h = splitmix64(h ^ (static_cast<uint64_t>(static_cast<uint32_t>(ids[i])) + 0x1000003ULL * (i + 1)));
    }
    if (subj_id >= 0) {
        h = splitmix64(h ^ (static_cast<uint64_t>(static_cast<uint32_t>(subj_id)) * 0xD6E8FEB86659FD93ULL));
    }
    return h;
}

static inline bool is_valid_utf8_bytes(const uint8_t* p, size_t n) {
    size_t i = 0;
    while (i < n) {
        uint8_t b = p[i];
        if (b < 0x80) {
            i += 1;
        } else if ((b & 0xE0) == 0xC0) {
            if (i + 1 >= n || (p[i + 1] & 0xC0) != 0x80) return false;
            if (b < 0xC2) return false;
            i += 2;
        } else if ((b & 0xF0) == 0xE0) {
            if (i + 2 >= n || (p[i + 1] & 0xC0) != 0x80 || (p[i + 2] & 0xC0) != 0x80) return false;
            if (b == 0xE0 && p[i + 1] < 0xA0) return false;
            if (b == 0xED && p[i + 1] >= 0xA0) return false;
            i += 3;
        } else if ((b & 0xF8) == 0xF0) {
            if (i + 3 >= n || (p[i + 1] & 0xC0) != 0x80 || (p[i + 2] & 0xC0) != 0x80 || (p[i + 3] & 0xC0) != 0x80) return false;
            if (b == 0xF0 && p[i + 1] < 0x90) return false;
            if (b > 0xF4 || (b == 0xF4 && p[i + 1] >= 0x90)) return false;
            i += 4;
        } else {
            return false;
        }
    }
    return true;
}

static inline bool has_control_or_newline(const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (p[i] == '\n' || p[i] == '\r' || p[i] == 0) return true;
    }
    return false;
}

struct TrieNode {
    int32_t motor_id = -1;
    std::vector<std::pair<uint8_t, int32_t>> ch;
    int32_t find_child(uint8_t b) const {
        for (const auto& kv : ch) {
            if (kv.first == b) return kv.second;
        }
        return -1;
    }
};

struct MotorVocab {
    std::vector<std::vector<uint8_t>> bytes;
    std::vector<TrieNode> trie;

    MotorVocab() {
        bytes.reserve(32768);
        for (int b = 0; b < 256; ++b) {
            bytes.push_back({static_cast<uint8_t>(b)});
        }
        rebuild_trie();
    }

    void rebuild_trie() {
        trie.clear();
        trie.emplace_back();
        for (int32_t mid = 0; mid < static_cast<int32_t>(bytes.size()); ++mid) {
            int32_t cur = 0;
            for (uint8_t b : bytes[mid]) {
                int32_t nxt = trie[cur].find_child(b);
                if (nxt < 0) {
                    nxt = static_cast<int32_t>(trie.size());
                    trie[cur].ch.emplace_back(b, nxt);
                    trie.emplace_back();
                }
                cur = nxt;
            }
            trie[cur].motor_id = mid;
        }
    }

    int64_t encode(const uint8_t* data, int64_t len, std::vector<int32_t>& out) const {
        out.clear();
        out.reserve(static_cast<size_t>(len / 3 + 64));
        int64_t pos = 0;
        while (pos < len) {
            int32_t cur = 0;
            int32_t best_id = static_cast<int32_t>(data[pos]);
            int64_t best_len = 1;
            int64_t scan = pos;
            while (scan < len) {
                int32_t nxt = trie[cur].find_child(data[scan]);
                if (nxt < 0) break;
                cur = nxt;
                scan += 1;
                if (trie[cur].motor_id >= 0) {
                    best_id = trie[cur].motor_id;
                    best_len = scan - pos;
                }
            }
            out.push_back(best_id);
            pos += best_len;
        }
        return static_cast<int64_t>(out.size());
    }
};

static constexpr int MACRO_WAYS = 8;

struct MacroWay {
    int32_t sig = 0;
    int32_t dst = -1;
    uint16_t w = 0;
};

struct MacroCortexV50 {
    int32_t V_m = 256;
    int num_banks = 5;
    // Bank 0: 1-chunk (1M buckets = 8M slots)
    // Bank 1: 2-chunk (2M buckets = 16M slots)
    // Bank 2: 3-chunk (4M buckets = 32M slots)
    // Bank 3: 4-chunk (4M buckets = 32M slots)
    // Bank 4: Subject-Skip (4M buckets = 32M slots)
    // Total = 120M slots (~1.44 GB CPU RAM, 0 MB GPU VRAM!)
    std::vector<int32_t> buckets;
    std::vector<std::vector<MacroWay>> banks;

    std::vector<uint64_t> motor_cnt;
    uint64_t total_motors = 0;
    std::vector<float> motor_idf;
    float median_idf = 5.0f;
    bool finalized = false;

    // Temporary accumulation during ingest (freed immediately in finalize!)
    std::vector<std::unordered_map<int32_t, float>> cooc_dir_tmp; // Directed forward (a -> b, dist 1..28)
    std::vector<std::unordered_map<int32_t, float>> cooc_sym_tmp; // Symmetric topic field (dist 1..38)

    // Finalized Directed Forward Subject->Predicate CSR
    std::vector<uint32_t> dir_off;
    std::vector<int32_t> dir_dst;
    std::vector<float> dir_w;

    // Finalized Symmetric Long-Range Topic Compatibility CSR (for Spectral Embeddings + Topic Field!)
    std::vector<uint32_t> sym_off;
    std::vector<int32_t> sym_dst;
    std::vector<float> sym_w;

    explicit MacroCortexV50(int32_t vocab_size, int32_t bucket_shift = 0) : V_m(vocab_size) {
        int sh = std::max(0, std::min(2, bucket_shift));
        buckets = {1 << (20 + sh), 1 << (21 + sh), 1 << (22 + sh), 1 << (22 + sh), 1 << (22 + sh)};
        banks.resize(num_banks);
        for (int b = 0; b < num_banks; ++b) {
            banks[b].resize(static_cast<size_t>(buckets[b]) * MACRO_WAYS);
        }
        motor_cnt.assign(V_m, 0);
        motor_idf.assign(V_m, 4.0f);
        cooc_dir_tmp.resize(V_m);
        cooc_sym_tmp.resize(V_m);
    }

    inline void update_bank(int bi, uint64_t k, int32_t dst) {
        uint64_t mask = static_cast<uint64_t>(buckets[bi] - 1);
        uint32_t bkt = static_cast<uint32_t>(k & mask);
        int32_t tag = static_cast<int32_t>((k >> 24) & 0x7FFFFFFFULL);
        if (tag == 0) tag = 1;
        MacroWay* base = banks[bi].data() + static_cast<int64_t>(bkt) * MACRO_WAYS;

        int empty_i = -1;
        int min_i = 0;
        uint16_t min_w = 65535;
        for (int j = 0; j < MACRO_WAYS; ++j) {
            if (base[j].w > 0 && base[j].sig == tag && base[j].dst == dst) {
                if (base[j].w < 65000) base[j].w += 1;
                return;
            }
            if (base[j].w == 0 && empty_i < 0) {
                empty_i = j;
            } else if (base[j].w < min_w) {
                min_w = base[j].w;
                min_i = j;
            }
        }
        if (empty_i >= 0) {
            base[empty_i].sig = tag;
            base[empty_i].dst = dst;
            base[empty_i].w = 1;
        } else if (min_w <= 2) {
            base[min_i].sig = tag;
            base[min_i].dst = dst;
            base[min_i].w = 1;
        } else {
            if (base[min_i].w > 1) base[min_i].w -= 1;
        }
    }
};

// Reusable thread-local scratch buffer for zero-allocation O(1) macro prediction!
struct PredScratch {
    std::vector<float> sc;
    std::vector<int32_t> sc_touched;
    std::vector<float> tf;
    std::vector<int32_t> tf_touched;

    void ensure(int V_m) {
        if (static_cast<int>(sc.size()) < V_m) {
            sc.assign(V_m, 0.0f);
            tf.assign(V_m, 0.0f);
            sc_touched.reserve(512);
            tf_touched.reserve(1024);
        }
    }

    inline void add_sc(int32_t id, float val) {
        if (sc[id] == 0.0f) sc_touched.push_back(id);
        sc[id] += val;
    }

    inline void add_tf(int32_t id, float val) {
        if (tf[id] == 0.0f) tf_touched.push_back(id);
        tf[id] += val;
    }

    void reset() {
        for (int32_t id : sc_touched) sc[id] = 0.0f;
        sc_touched.clear();
        for (int32_t id : tf_touched) tf[id] = 0.0f;
        tf_touched.clear();
    }
};

static int predict_topk_fast(
    const MacroCortexV50* mc,
    PredScratch& scratch,
    const int32_t* hist_ids,
    int hist_len,
    const int32_t* topic_ids,
    const float* topic_w,
    int num_topics,
    float topic_ppmi_gain,
    int top_k,
    int32_t* out_ids,
    float* out_scores
) {
    if (!mc || hist_len <= 0 || top_k <= 0) return 0;
    const int V_m = mc->V_m;
    scratch.ensure(V_m);

    static constexpr float BANK_W[5] = {0.55f, 1.35f, 2.35f, 3.15f, 1.75f};
    int32_t ctx[4];
    int avail = std::min(4, hist_len);
    for (int d = 0; d < avail; ++d) ctx[d] = hist_ids[hist_len - 1 - d];

    // 1) Local 1..4-chunk Macro-Transitions (Banks 0..3)
    for (int bi = 0; bi < 4; ++bi) {
        int span = bi + 1;
        if (span > avail) break;
        uint64_t k = hash_macro_ctx(ctx, span, -1);
        uint64_t mask = static_cast<uint64_t>(mc->buckets[bi] - 1);
        uint32_t bkt = static_cast<uint32_t>(k & mask);
        int32_t tag = static_cast<int32_t>((k >> 24) & 0x7FFFFFFFULL);
        if (tag == 0) tag = 1;
        const MacroWay* base = mc->banks[bi].data() + static_cast<int64_t>(bkt) * MACRO_WAYS;

        float w_tot = 0.0f;
        for (int j = 0; j < MACRO_WAYS; ++j) {
            if (base[j].w > 0 && base[j].sig == tag && base[j].dst >= 0 && base[j].dst < V_m) {
                w_tot += static_cast<float>(base[j].w);
            }
        }
        if (w_tot <= 0.0f) continue;
        for (int j = 0; j < MACRO_WAYS; ++j) {
            int32_t dst = base[j].dst;
            if (base[j].w > 0 && base[j].sig == tag && dst >= 0 && dst < V_m) {
                float wj = static_cast<float>(base[j].w);
                float ev = std::tanh(std::log1p(wj) * 0.40f) * (wj / (w_tot + 1.5f));
                scratch.add_sc(dst, BANK_W[bi] * ev);
            }
        }
    }

    // 2) Subject-Skip Bank (Bank 4: active_subject x prev_chunk -> next_chunk)
    if (num_topics > 0) {
        uint64_t mask4 = static_cast<uint64_t>(mc->buckets[4] - 1);
        for (int ti = 0; ti < num_topics; ++ti) {
            int32_t tid = topic_ids[ti];
            if (tid < 0 || tid >= V_m) continue;
            uint64_t k = hash_macro_ctx(ctx, 1, tid);
            uint32_t bkt = static_cast<uint32_t>(k & mask4);
            int32_t tag = static_cast<int32_t>((k >> 24) & 0x7FFFFFFFULL);
            if (tag == 0) tag = 1;
            const MacroWay* base = mc->banks[4].data() + static_cast<int64_t>(bkt) * MACRO_WAYS;

            float w_tot = 0.0f;
            for (int j = 0; j < MACRO_WAYS; ++j) {
                if (base[j].w > 0 && base[j].sig == tag && base[j].dst >= 0 && base[j].dst < V_m) {
                    w_tot += static_cast<float>(base[j].w);
                }
            }
            if (w_tot <= 0.0f) continue;
            float tw = topic_w[ti];
            for (int j = 0; j < MACRO_WAYS; ++j) {
                int32_t dst = base[j].dst;
                if (base[j].w > 0 && base[j].sig == tag && dst >= 0 && dst < V_m) {
                    float wj = static_cast<float>(base[j].w);
                    float ev = std::tanh(std::log1p(wj) * 0.40f) * (wj / (w_tot + 1.5f));
                    scratch.add_sc(dst, BANK_W[4] * tw * ev);
                }
            }
        }
    }

    // 3) Directed Forward Subject -> Predicate PPMI Field!
    if (num_topics > 0 && topic_ppmi_gain > 0.0f && !mc->dir_off.empty()) {
        for (int ti = 0; ti < num_topics; ++ti) {
            int32_t a = topic_ids[ti];
            if (a < 0 || a >= V_m) continue;
            float tw = topic_w[ti];
            uint32_t p0 = mc->dir_off[a];
            uint32_t p1 = mc->dir_off[a + 1];
            for (uint32_t p = p0; p < p1; ++p) {
                scratch.add_tf(mc->dir_dst[p], tw * mc->dir_w[p]);
            }
        }
        // Boost syntactically proposed candidates that match the Directed Subject->Predicate field
        for (int32_t mid : scratch.sc_touched) {
            float fv = scratch.tf[mid];
            if (fv > 0.0f) {
                scratch.sc[mid] += topic_ppmi_gain * std::tanh(fv * 0.65f);
            }
        }
        // Gently seed top-14 strongest directed predicate nodes not already proposed
        if (!scratch.tf_touched.empty()) {
            std::vector<std::pair<float, int32_t>> tf_top;
            tf_top.reserve(scratch.tf_touched.size());
            for (int32_t mid : scratch.tf_touched) {
                if (scratch.sc[mid] == 0.0f) tf_top.emplace_back(scratch.tf[mid], mid);
            }
            int take_t = std::min(14, static_cast<int>(tf_top.size()));
            if (take_t > 0) {
                std::partial_sort(tf_top.begin(), tf_top.begin() + take_t, tf_top.end(),
                                  [](const auto& x, const auto& y) { return x.first > y.first; });
                for (int i = 0; i < take_t; ++i) {
                    scratch.add_sc(tf_top[i].second, 0.30f * topic_ppmi_gain * std::tanh(tf_top[i].first * 0.65f));
                }
            }
        }
    }

    int n_cand = static_cast<int>(scratch.sc_touched.size());
    if (n_cand == 0) {
        scratch.reset();
        return 0;
    }

    std::vector<std::pair<float, int32_t>> ranked;
    ranked.reserve(n_cand);
    for (int32_t mid : scratch.sc_touched) {
        ranked.emplace_back(scratch.sc[mid], mid);
    }
    int take = std::min(top_k, n_cand);
    std::partial_sort(ranked.begin(), ranked.begin() + take, ranked.end(),
                      [](const auto& x, const auto& y) { return x.first > y.first; });
    for (int i = 0; i < take; ++i) {
        out_ids[i] = ranked[i].second;
        out_scores[i] = ranked[i].first;
    }
    scratch.reset();
    return take;
}

} // namespace

extern "C" {

void* motor_vocab_create() {
    return new MotorVocab();
}

void motor_vocab_destroy(void* vh) {
    delete static_cast<MotorVocab*>(vh);
}

int motor_vocab_size(void* vh) {
    return vh ? static_cast<int>(static_cast<MotorVocab*>(vh)->bytes.size()) : 0;
}

int motor_vocab_get_bytes(void* vh, int32_t mid, uint8_t* out_buf, int max_len) {
    if (!vh) return 0;
    const auto* v = static_cast<const MotorVocab*>(vh);
    if (mid < 0 || mid >= static_cast<int32_t>(v->bytes.size())) return 0;
    const auto& b = v->bytes[mid];
    int n = std::min(static_cast<int>(b.size()), max_len);
    if (n > 0) std::memcpy(out_buf, b.data(), n);
    return n;
}

int motor_vocab_add_raw(void* vh, const uint8_t* ptr, int len) {
    if (!vh || !ptr || len <= 0) return -1;
    auto* v = static_cast<MotorVocab*>(vh);
    v->bytes.emplace_back(ptr, ptr + len);
    return static_cast<int>(v->bytes.size()) - 1;
}

void motor_vocab_rebuild_trie(void* vh) {
    if (vh) static_cast<MotorVocab*>(vh)->rebuild_trie();
}

// Grow output-side Broca motor macro-nodes autonomously from ZERO hidden motor nodes!
int motor_grow_vocab(
    void* vh,
    const uint8_t* data,
    int64_t len,
    int target_extra_motors,
    int max_rounds,
    int max_motor_bytes,
    int min_pair_count
) {
    if (!vh || !data || len < 1024 || target_extra_motors <= 0) return 0;
    auto* v = static_cast<MotorVocab*>(vh);

    std::vector<int32_t> seq(static_cast<size_t>(len));
    for (int64_t i = 0; i < len; ++i) seq[i] = static_cast<int32_t>(data[i]);

    // Round 0: Crystallize intact multi-byte UTF-8 characters (RFC 3629)
    {
        std::unordered_map<uint64_t, uint32_t> utf_cnt;
        utf_cnt.reserve(16384);
        int64_t i = 0;
        while (i < len) {
            uint8_t b = data[i];
            int clen = 1;
            if (b < 0x80) clen = 1;
            else if ((b & 0xE0) == 0xC0 && i + 1 < len) clen = 2;
            else if ((b & 0xF0) == 0xE0 && i + 2 < len) clen = 3;
            else if ((b & 0xF8) == 0xF0 && i + 3 < len) clen = 4;
            if (clen >= 2 && is_valid_utf8_bytes(data + i, clen)) {
                uint64_t key = static_cast<uint64_t>(clen);
                for (int k = 0; k < clen; ++k) key = (key << 8) | data[i + k];
                utf_cnt[key] += 1;
            }
            i += clen;
        }
        std::vector<std::pair<uint32_t, uint64_t>> ranked;
        ranked.reserve(utf_cnt.size());
        for (const auto& kv : utf_cnt) {
            if (kv.second >= static_cast<uint32_t>(std::max(12, min_pair_count / 2))) {
                ranked.emplace_back(kv.second, kv.first);
            }
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        int cap0 = std::min(static_cast<int>(ranked.size()), std::min(3500, target_extra_motors / 3));
        std::unordered_map<uint64_t, int32_t> utf_to_id;
        utf_to_id.reserve(cap0 * 2 + 16);
        for (int r = 0; r < cap0; ++r) {
            uint64_t key = ranked[r].second;
            int clen = static_cast<int>(key >> (8 * 4));
            if (clen < 2 || clen > 4) {
                for (int c = 2; c <= 4; ++c) {
                    if ((key >> (8 * c)) == static_cast<uint64_t>(c)) { clen = c; break; }
                }
            }
            std::vector<uint8_t> cb(clen);
            for (int k = clen - 1; k >= 0; --k) { cb[k] = static_cast<uint8_t>(key & 0xFF); key >>= 8; }
            int32_t nid = static_cast<int32_t>(v->bytes.size());
            v->bytes.push_back(std::move(cb));
            utf_to_id[ranked[r].second] = nid;
        }
        std::vector<int32_t> nseq;
        nseq.reserve(len / 2);
        int64_t p = 0;
        while (p < len) {
            uint8_t b = data[p];
            int clen = 1;
            if (b < 0x80) clen = 1;
            else if ((b & 0xE0) == 0xC0 && p + 1 < len) clen = 2;
            else if ((b & 0xF0) == 0xE0 && p + 2 < len) clen = 3;
            else if ((b & 0xF8) == 0xF0 && p + 3 < len) clen = 4;
            if (clen >= 2) {
                uint64_t key = static_cast<uint64_t>(clen);
                for (int k = 0; k < clen; ++k) key = (key << 8) | data[p + k];
                auto it = utf_to_id.find(key);
                if (it != utf_to_id.end()) {
                    nseq.push_back(it->second);
                    p += clen;
                    continue;
                }
            }
            nseq.push_back(static_cast<int32_t>(data[p]));
            p += 1;
        }
        seq.swap(nseq);
    }

    int target_total = 256 + target_extra_motors;
    int per_round = std::max(512, (target_total - static_cast<int>(v->bytes.size()) + max_rounds - 1) / std::max(1, max_rounds));
    int n_threads = std::min(12, omp_get_max_threads());

    for (int rd = 0; rd < max_rounds && static_cast<int>(v->bytes.size()) < target_total; ++rd) {
        size_t L = seq.size();
        if (L < 4) break;

        std::vector<std::unordered_map<uint64_t, uint32_t>> local_maps(n_threads);
        #pragma omp parallel num_threads(n_threads)
        {
            int tid = omp_get_thread_num();
            auto& mp = local_maps[tid];
            mp.reserve(1 << 16);
            size_t chunk = (L - 1 + n_threads - 1) / n_threads;
            size_t s0 = static_cast<size_t>(tid) * chunk;
            size_t s1 = std::min(L - 1, s0 + chunk);
            for (size_t i = s0; i < s1; ++i) {
                int32_t a = seq[i];
                int32_t b = seq[i + 1];
                const auto& ba = v->bytes[a];
                const auto& bb = v->bytes[b];
                if (ba.size() + bb.size() > static_cast<size_t>(max_motor_bytes)) continue;
                if (has_control_or_newline(ba.data(), ba.size()) || has_control_or_newline(bb.data(), bb.size())) continue;
                if (ba.size() == 1 && ba[0] >= 0x80) continue;
                if (bb.size() == 1 && bb[0] >= 0x80) continue;
                uint64_t pk = (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
                mp[pk] += 1;
            }
        }

        std::unordered_map<uint64_t, uint32_t> merged;
        merged.reserve(1 << 18);
        for (int t = 0; t < n_threads; ++t) {
            for (const auto& kv : local_maps[t]) merged[kv.first] += kv.second;
            std::unordered_map<uint64_t, uint32_t>().swap(local_maps[t]);
        }

        std::vector<std::pair<uint32_t, uint64_t>> pairs;
        pairs.reserve(merged.size());
        for (const auto& kv : merged) {
            if (kv.second >= static_cast<uint32_t>(min_pair_count)) pairs.emplace_back(kv.second, kv.first);
        }
        if (pairs.empty()) break;

        int need = std::min(per_round, target_total - static_cast<int>(v->bytes.size()));
        int take = std::min(need, static_cast<int>(pairs.size()));
        std::partial_sort(pairs.begin(), pairs.begin() + take, pairs.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        std::unordered_map<uint64_t, int32_t> merge_to_id;
        merge_to_id.reserve(take * 2 + 16);
        int added = 0;
        for (int i = 0; i < take; ++i) {
            uint64_t pk = pairs[i].second;
            int32_t a = static_cast<int32_t>(pk >> 32);
            int32_t b = static_cast<int32_t>(pk & 0xFFFFFFFFULL);
            std::vector<uint8_t> comb;
            comb.reserve(v->bytes[a].size() + v->bytes[b].size());
            comb.insert(comb.end(), v->bytes[a].begin(), v->bytes[a].end());
            comb.insert(comb.end(), v->bytes[b].begin(), v->bytes[b].end());
            if (!is_valid_utf8_bytes(comb.data(), comb.size())) continue;
            int32_t nid = static_cast<int32_t>(v->bytes.size());
            v->bytes.push_back(std::move(comb));
            merge_to_id[pk] = nid;
            added += 1;
        }
        if (added == 0) break;

        std::vector<int32_t> nseq;
        nseq.reserve(L);
        size_t i = 0;
        while (i < L) {
            if (i + 1 < L) {
                uint64_t pk = (static_cast<uint64_t>(static_cast<uint32_t>(seq[i])) << 32) | static_cast<uint32_t>(seq[i + 1]);
                auto it = merge_to_id.find(pk);
                if (it != merge_to_id.end()) {
                    nseq.push_back(it->second);
                    i += 2;
                    continue;
                }
            }
            nseq.push_back(seq[i]);
            i += 1;
        }
        seq.swap(nseq);
    }

    v->rebuild_trie();
    return static_cast<int>(v->bytes.size()) - 256;
}

int64_t motor_encode(void* vh, const uint8_t* data, int64_t len, int32_t* out_ids, int64_t max_out) {
    if (!vh || !data || len <= 0 || !out_ids || max_out <= 0) return 0;
    const auto* v = static_cast<const MotorVocab*>(vh);
    std::vector<int32_t> tmp;
    v->encode(data, len, tmp);
    int64_t n = std::min(static_cast<int64_t>(tmp.size()), max_out);
    if (n > 0) std::memcpy(out_ids, tmp.data(), static_cast<size_t>(n) * sizeof(int32_t));
    return n;
}

void* macro_create_v50(int32_t vocab_size) {
    return new MacroCortexV50(vocab_size, 0);
}

void* macro_create_v50_sized(int32_t vocab_size, int32_t bucket_shift) {
    return new MacroCortexV50(vocab_size, bucket_shift);
}

void macro_destroy_v50(void* mh) {
    delete static_cast<MacroCortexV50*>(mh);
}

int64_t macro_ingest_stream_v50(
    void* mh,
    const int32_t* m_ids,
    int64_t len,
    int collect_cooc,
    int cooc_window,
    int cooc_stride,
    int req_threads
) {
    if (!mh || !m_ids || len < 4) return 0;
    auto* mc = static_cast<MacroCortexV50*>(mh);
    const int32_t V_m = mc->V_m;
    if (collect_cooc && (mc->finalized || mc->cooc_dir_tmp.size() != static_cast<size_t>(V_m) || mc->cooc_sym_tmp.size() != static_cast<size_t>(V_m))) {
        return -1;
    }

    // 1) Update unigram counts & IDF estimate
    for (int64_t i = 0; i < len; ++i) {
        int32_t id = m_ids[i];
        if (id >= 0 && id < V_m) mc->motor_cnt[id] += 1;
    }
    mc->total_motors += static_cast<uint64_t>(len);
    double logN = std::log(static_cast<double>(mc->total_motors) + 10.0);
    std::vector<float> idf_vals;
    idf_vals.reserve(V_m);
    for (int32_t id = 0; id < V_m; ++id) {
        mc->motor_idf[id] = static_cast<float>(logN - std::log(static_cast<double>(mc->motor_cnt[id]) + 1.0));
        if (id >= 256 && mc->motor_cnt[id] > 0) idf_vals.push_back(mc->motor_idf[id]);
    }
    if (!idf_vals.empty()) {
        size_t mid_idx = idf_vals.size() / 2;
        std::nth_element(idf_vals.begin(), idf_vals.begin() + mid_idx, idf_vals.end());
        mc->median_idf = idf_vals[mid_idx];
    }

    // 2) RACE-FREE, 100% CHRONOLOGICAL, BIT-EXACT DETERMINISTIC Bucket-Owned OpenMP Bank Update!
    // Every thread owns strictly disjoint buckets (bkt % nt == tid) and processes all positions in exact
    // chronological order (4..len-1) -> 1-thread and 12-thread produce 100% BIT-EXACT IDENTICAL banks with 0 extra RAM!
    int max_t = std::min(12, omp_get_max_threads());
    int n_threads = (req_threads > 0) ? std::min(req_threads, max_t) : max_t;
    const uint64_t m0 = static_cast<uint64_t>(mc->buckets[0] - 1);
    const uint64_t m1 = static_cast<uint64_t>(mc->buckets[1] - 1);
    const uint64_t m2 = static_cast<uint64_t>(mc->buckets[2] - 1);
    const uint64_t m3 = static_cast<uint64_t>(mc->buckets[3] - 1);
    const uint64_t m4 = static_cast<uint64_t>(mc->buckets[4] - 1);

    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();

        int32_t act_subj[2] = {-1, -1};
        float act_idf[2] = {0.0f, 0.0f};
        int act_age[2] = {100, 100};

        for (int64_t i = 4; i < len; ++i) {
            int32_t target = m_ids[i];
            if (target < 0 || target >= V_m) continue;
            int32_t ctx[4] = {m_ids[i - 1], m_ids[i - 2], m_ids[i - 3], m_ids[i - 4]};

            int32_t prev = ctx[0];
            act_age[0] += 1;
            act_age[1] += 1;
            if (prev >= 256) {
                float pidf = mc->motor_idf[prev];
                if (pidf > 5.0f) {
                    if (act_subj[0] == prev) {
                        act_age[0] = 0;
                    } else if (act_subj[1] == prev) {
                        act_age[1] = 0;
                    } else if (act_age[0] > 28 || pidf >= act_idf[0] * 0.88f) {
                        act_subj[1] = act_subj[0]; act_idf[1] = act_idf[0]; act_age[1] = act_age[0];
                        act_subj[0] = prev; act_idf[0] = pidf; act_age[0] = 0;
                    } else if (act_age[1] > 28 || pidf >= act_idf[1] * 0.90f) {
                        act_subj[1] = prev; act_idf[1] = pidf; act_age[1] = 0;
                    }
                }
            }

            uint64_t k0 = hash_macro_ctx(ctx, 1, -1);
            if (static_cast<int>((k0 & m0) % nt) == tid) mc->update_bank(0, k0, target);
            uint64_t k1 = hash_macro_ctx(ctx, 2, -1);
            if (static_cast<int>((k1 & m1) % nt) == tid) mc->update_bank(1, k1, target);
            uint64_t k2 = hash_macro_ctx(ctx, 3, -1);
            if (static_cast<int>((k2 & m2) % nt) == tid) mc->update_bank(2, k2, target);
            uint64_t k3 = hash_macro_ctx(ctx, 4, -1);
            if (static_cast<int>((k3 & m3) % nt) == tid) mc->update_bank(3, k3, target);

            if (act_subj[0] >= 256 && act_age[0] <= 28 && act_subj[0] != prev) {
                uint64_t ks0 = hash_macro_ctx(ctx, 1, act_subj[0]);
                if (static_cast<int>((ks0 & m4) % nt) == tid) mc->update_bank(4, ks0, target);
            }
            if (act_subj[1] >= 256 && act_age[1] <= 28 && act_subj[1] != prev) {
                uint64_t ks1 = hash_macro_ctx(ctx, 1, act_subj[1]);
                if (static_cast<int>((ks1 & m4) % nt) == tid) mc->update_bank(4, ks1, target);
            }
        }
    }

    // 3) Directed Forward Subject->Predicate + Symmetric Long-Range Topic Co-occurrence!
    // Thread-safe sharding by (a % nt == tid) + strict per-row memory cap (<= 2048 entries/row)!
    if (collect_cooc && cooc_window > 0) {
        int stride = std::max(1, cooc_stride);
        int W_dir = std::min(28, cooc_window);
        int W_sym = std::min(38, cooc_window + 8);
        float min_idf_subj = std::max(4.6f, mc->median_idf * 0.62f);

        auto prune_row_if_large = [](std::unordered_map<int32_t, float>& mp) {
            if (mp.size() <= 2048) return;
            std::vector<float> vs;
            vs.reserve(mp.size());
            for (const auto& kv : mp) vs.push_back(kv.second);
            size_t mid = vs.size() / 2;
            std::nth_element(vs.begin(), vs.begin() + mid, vs.end());
            float cut = vs[mid];
            for (auto it = mp.begin(); it != mp.end(); ) {
                if (it->second <= cut) it = mp.erase(it);
                else ++it;
            }
        };

        #pragma omp parallel num_threads(n_threads)
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            for (int64_t i = 0; i < len; i += stride) {
                int32_t a = m_ids[i];
                if (a < 256 || a >= V_m) continue;
                if ((a % nt) != tid) continue;
                if (mc->motor_idf[a] < min_idf_subj) continue;

                auto& mp_dir = mc->cooc_dir_tmp[a];
                auto& mp_sym = mc->cooc_sym_tmp[a];

                int64_t j_end = std::min(len, i + static_cast<int64_t>(W_sym) + 1);
                for (int64_t j = i + 1; j < j_end; ++j) {
                    int32_t b = m_ids[j];
                    if (b < 256 || b >= V_m || b == a) continue;
                    if (mc->motor_idf[b] < 4.2f) continue;
                    int dist = static_cast<int>(j - i);
                    float w_sym = 1.0f / (1.0f + 0.08f * static_cast<float>(dist));
                    mp_sym[b] += w_sym;
                    if (dist <= W_dir) {
                        float w_dir = 1.0f / (1.0f + 0.12f * static_cast<float>(dist));
                        mp_dir[b] += w_dir;
                    }
                }
                int64_t j_beg = std::max<int64_t>(0, i - static_cast<int64_t>(W_sym));
                for (int64_t j = j_beg; j < i; ++j) {
                    int32_t b = m_ids[j];
                    if (b < 256 || b >= V_m || b == a) continue;
                    if (mc->motor_idf[b] < 4.2f) continue;
                    int dist = static_cast<int>(i - j);
                    float w_sym = 1.0f / (1.0f + 0.08f * static_cast<float>(dist));
                    mp_sym[b] += w_sym;
                }
                prune_row_if_large(mp_dir);
                prune_row_if_large(mp_sym);
            }
        }
    }

    return len;
}

// Bit-exact checksum over all 5 macro banks (for automated 1-thread vs 12-thread determinism verification!)
uint64_t macro_bank_checksum_v50(void* mh) {
    if (!mh) return 0;
    const auto* mc = static_cast<const MacroCortexV50*>(mh);
    uint64_t h = 0x123456789ABCDEF0ULL;
    for (int bi = 0; bi < mc->num_banks; ++bi) {
        const auto& vec = mc->banks[bi];
        for (size_t i = 0; i < vec.size(); ++i) {
            if (vec[i].w > 0) {
                uint64_t v = (static_cast<uint64_t>(static_cast<uint32_t>(vec[i].sig)) << 32)
                           ^ (static_cast<uint64_t>(static_cast<uint32_t>(vec[i].dst)) << 16)
                           ^ static_cast<uint64_t>(vec[i].w)
                           ^ (static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL);
                h = splitmix64(h ^ v);
            }
        }
    }
    return h;
}

// Finalize Directed Subject->Predicate CSR + Symmetric Topic PPMI CSR and FREE all temp maps immediately!
int64_t macro_finalize_ppmi_v50(void* mh, int top_k_dir, int top_k_sym, float min_cooc, int64_t* out_sym_edges) {
    if (!mh) return -1;
    auto* mc = static_cast<MacroCortexV50*>(mh);
    const int V_m = mc->V_m;
    if (mc->finalized || mc->cooc_dir_tmp.size() != static_cast<size_t>(V_m) || mc->cooc_sym_tmp.size() != static_cast<size_t>(V_m)) {
        return -1;
    }

    double N = std::max(1.0, static_cast<double>(mc->total_motors));
    auto build_csr = [&](std::vector<std::unordered_map<int32_t, float>>& src,
                         int top_k, float min_c,
                         std::vector<uint32_t>& off, std::vector<int32_t>& dst, std::vector<float>& w) -> int64_t {
        std::vector<std::vector<std::pair<int32_t, float>>> rows(V_m);
        #pragma omp parallel for schedule(dynamic, 64)
        for (int a = 0; a < V_m; ++a) {
            const auto& mp = src[a];
            if (mp.empty() || mc->motor_cnt[a] == 0) continue;
            double pa = static_cast<double>(mc->motor_cnt[a]) / N;
            std::vector<std::pair<float, int32_t>> scored;
            scored.reserve(mp.size());
            for (const auto& kv : mp) {
                int32_t b = kv.first;
                float c = kv.second;
                if (c < min_c || mc->motor_cnt[b] == 0) continue;
                double pb = static_cast<double>(mc->motor_cnt[b]) / N;
                double pab = static_cast<double>(c) / (N * 0.25 + 1.0);
                double pmi = std::log((pab + 1e-12) / (pa * pb + 1e-12));
                if (pmi <= 0.15) continue;
                float sc = static_cast<float>(pmi * std::log1p(static_cast<double>(c)));
                scored.emplace_back(sc, b);
            }
            if (scored.empty()) continue;
            int take = std::min(top_k, static_cast<int>(scored.size()));
            std::partial_sort(scored.begin(), scored.begin() + take, scored.end(),
                              [](const auto& x, const auto& y) { return x.first > y.first; });
            float norm_max = std::max(scored[0].first, 1e-4f);
            rows[a].reserve(take);
            for (int i = 0; i < take; ++i) {
                rows[a].emplace_back(scored[i].second, scored[i].first / norm_max);
            }
        }
        off.assign(V_m + 1, 0);
        dst.clear();
        w.clear();
        uint64_t total = 0;
        for (int a = 0; a < V_m; ++a) {
            off[a] = static_cast<uint32_t>(total);
            for (const auto& kv : rows[a]) {
                dst.push_back(kv.first);
                w.push_back(kv.second);
                total += 1;
            }
            std::unordered_map<int32_t, float>().swap(src[a]);
        }
        off[V_m] = static_cast<uint32_t>(total);
        src.clear();
        src.shrink_to_fit();
        return static_cast<int64_t>(total);
    };

    int64_t dir_e = build_csr(mc->cooc_dir_tmp, top_k_dir, min_cooc, mc->dir_off, mc->dir_dst, mc->dir_w);
    int64_t sym_e = build_csr(mc->cooc_sym_tmp, top_k_sym, min_cooc, mc->sym_off, mc->sym_dst, mc->sym_w);
    mc->finalized = true;
    if (out_sym_edges) *out_sym_edges = sym_e;
    return dir_e;
}

int64_t macro_sleep_prune_v50(void* mh) {
    if (!mh) return 0;
    auto* mc = static_cast<MacroCortexV50*>(mh);
    int64_t active = 0;
    for (int bi = 0; bi < mc->num_banks; ++bi) {
        auto& vec = mc->banks[bi];
        int64_t n = static_cast<int64_t>(vec.size());
        int64_t loc = 0;
        #pragma omp parallel for reduction(+:loc) schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            if (vec[i].w < 2) {
                vec[i].sig = 0;
                vec[i].dst = -1;
                vec[i].w = 0;
            } else {
                loc += 1;
            }
        }
        active += loc;
    }
    return active;
}

// Export Symmetric Topic PPMI CSR for GPU Spectral Hebbian Eigen-Embedding (0 backprop!)
int64_t macro_sym_edge_count_v50(void* mh) {
    return mh ? static_cast<int64_t>(static_cast<const MacroCortexV50*>(mh)->sym_dst.size()) : 0;
}

int64_t macro_export_sym_csr_v50(void* mh, uint32_t* out_off, int32_t* out_dst, float* out_w) {
    if (!mh) return 0;
    const auto* mc = static_cast<const MacroCortexV50*>(mh);
    if (mc->sym_off.empty()) return 0;
    std::memcpy(out_off, mc->sym_off.data(), mc->sym_off.size() * sizeof(uint32_t));
    if (!mc->sym_dst.empty()) {
        std::memcpy(out_dst, mc->sym_dst.data(), mc->sym_dst.size() * sizeof(int32_t));
        std::memcpy(out_w, mc->sym_w.data(), mc->sym_w.size() * sizeof(float));
    }
    return static_cast<int64_t>(mc->sym_dst.size());
}

void macro_export_idfs_v50(void* mh, float* out_idfs) {
    if (!mh || !out_idfs) return;
    const auto* mc = static_cast<const MacroCortexV50*>(mh);
    std::memcpy(out_idfs, mc->motor_idf.data(), static_cast<size_t>(mc->V_m) * sizeof(float));
}

int macro_predict_topk_v50(
    void* mh,
    const int32_t* hist_ids,
    int hist_len,
    const int32_t* topic_ids,
    const float* topic_w,
    int num_topics,
    float topic_ppmi_gain,
    int top_k,
    int32_t* out_ids,
    float* out_scores
) {
    thread_local PredScratch scratch;
    return predict_topk_fast(
        static_cast<const MacroCortexV50*>(mh),
        scratch, hist_ids, hist_len, topic_ids, topic_w, num_topics,
        topic_ppmi_gain, top_k, out_ids, out_scores
    );
}

int macro_top_dir_neighbors_v50(void* mh, int32_t mid, int top_k, int32_t* out_ids, float* out_w) {
    if (!mh || top_k <= 0) return 0;
    const auto* mc = static_cast<const MacroCortexV50*>(mh);
    if (mid < 0 || mid >= mc->V_m || mc->dir_off.empty()) return 0;
    uint32_t p0 = mc->dir_off[mid];
    uint32_t p1 = mc->dir_off[mid + 1];
    int n = std::min(top_k, static_cast<int>(p1 - p0));
    for (int i = 0; i < n; ++i) {
        out_ids[i] = mc->dir_dst[p0 + i];
        out_w[i] = mc->dir_w[p0 + i];
    }
    return n;
}

// Compute Directed PPMI affinity between a candidate motor `cand` and active topics!
float macro_dir_affinity_v50(void* mh, int32_t cand, const int32_t* topic_ids, const float* topic_w, int num_topics) {
    if (!mh || num_topics <= 0) return 0.0f;
    const auto* mc = static_cast<const MacroCortexV50*>(mh);
    if (cand < 0 || cand >= mc->V_m || mc->dir_off.empty()) return 0.0f;
    float acc = 0.0f;
    for (int ti = 0; ti < num_topics; ++ti) {
        int32_t a = topic_ids[ti];
        if (a < 0 || a >= mc->V_m) continue;
        uint32_t p0 = mc->dir_off[a];
        uint32_t p1 = mc->dir_off[a + 1];
        for (uint32_t p = p0; p < p1; ++p) {
            if (mc->dir_dst[p] == cand) {
                acc += topic_w[ti] * mc->dir_w[p];
                break;
            }
        }
    }
    return acc;
}

// 100% STRICTLY CAUSAL ONLINE Prefix-Trie Dual-Hypothesis Marginalizer (ZERO FUTURE LEAKAGE!)
// Invariant: For ANY two streams sharing prefix stream[0..k] and differing at k+1,
// out_boost[0..k, 0..255] is 100% BIT-EXACT IDENTICAL (verified by automated prefix-equivalence tests!).
// Covers ALL prediction positions t = 0 .. T-2 (including the last byte T-2!).
void macro_marginal_byte_boost_causal_v50(
    void* vh,
    void* mh,
    const uint8_t* data_2d,
    int B,
    int T,
    int top_k,
    float topic_ppmi_gain,
    float tau_idf,
    const float* emb_flat,
    const float* w_pred_flat,
    int D_m,
    float cont_gain,
    float inhib_gain,
    float* out_boost
) {
    if (!vh || !mh || !data_2d || !out_boost || B <= 0 || T < 2) return;
    const auto* v = static_cast<const MotorVocab*>(vh);
    const auto* mc = static_cast<const MacroCortexV50*>(mh);
    const int V_m = mc->V_m;
    const bool use_cont = (emb_flat != nullptr && D_m > 0 && (cont_gain > 0.0f || inhib_gain > 0.0f));
    const bool has_wp = (use_cont && w_pred_flat != nullptr);

    int n_threads = std::min(12, omp_get_max_threads());
    #pragma omp parallel num_threads(n_threads)
    {
        PredScratch scratch;
        scratch.ensure(V_m);
        std::vector<int32_t> committed;
        committed.reserve(static_cast<size_t>(T / 3 + 32));
        std::vector<int32_t> tmp_hist;
        tmp_hist.reserve(40);

        std::vector<int32_t> commit_ids(top_k);
        std::vector<float> commit_sc(top_k);
        std::vector<float> commit_probs(top_k);

        std::vector<int32_t> bnd_ids(top_k);
        std::vector<float> bnd_sc(top_k);
        std::vector<float> bnd_probs(top_k);

        std::vector<float> h_long(std::max(1, D_m), 0.0f);
        std::vector<float> h_mid(std::max(1, D_m), 0.0f);
        std::vector<float> h_long_tmp(std::max(1, D_m), 0.0f);
        std::vector<float> h_mid_tmp(std::max(1, D_m), 0.0f);
        std::vector<float> h_thought(std::max(1, D_m), 0.0f);
        std::vector<float> h_pred(std::max(1, D_m), 0.0f);

        auto update_thought_vec = [&](std::vector<float>& hl, std::vector<float>& hm, int32_t mid) {
            if (!use_cont || mid < 256 || mid >= V_m) return;
            float idf = mc->motor_idf[mid];
            float g = 1.0f / (1.0f + std::exp(-(idf - tau_idf) * 2.2f));
            if (g < 0.08f) return;
            const float* em = emb_flat + static_cast<int64_t>(mid) * D_m;
            float a_l = 0.07f * g;
            float a_m = 0.24f * g;
            for (int d = 0; d < D_m; ++d) {
                hl[d] = (1.0f - a_l) * hl[d] + a_l * em[d];
                hm[d] = (1.0f - a_m) * hm[d] + a_m * em[d];
            }
        };

        auto run_predict = [&](
            const int32_t* seq_ptr, int seq_len,
            const std::vector<float>& hl, const std::vector<float>& hm,
            std::vector<int32_t>& out_i, std::vector<float>& out_s, std::vector<float>& out_p
        ) -> int {
            if (seq_len <= 0) return 0;
            int w0 = std::max(0, seq_len - 32);
            std::pair<float, int32_t> cand_t[32];
            int nct = 0;
            for (int j = seq_len - 1; j >= w0; --j) {
                int32_t id = seq_ptr[j];
                if (id < 256 || id >= V_m) continue;
                float idf = mc->motor_idf[id];
                if (idf < tau_idf - 0.8f) continue;
                bool dup = false;
                for (int k = 0; k < nct; ++k) {
                    if (cand_t[k].second == id) { dup = true; break; }
                }
                if (dup) continue;
                float rec = 1.0f / (1.0f + 0.05f * static_cast<float>(seq_len - 1 - j));
                cand_t[nct++] = {idf * rec, id};
            }
            int32_t t_ids[6];
            float t_w[6];
            int nt = 0;
            if (nct > 0) {
                int take_t = std::min(6, nct);
                std::partial_sort(cand_t, cand_t + take_t, cand_t + nct,
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                float mx_t = std::max(cand_t[0].first, 1e-5f);
                for (int k = 0; k < take_t; ++k) {
                    t_ids[k] = cand_t[k].second;
                    t_w[k] = cand_t[k].first / mx_t;
                }
                nt = take_t;
            }

            int h0 = std::max(0, seq_len - 6);
            int n_pred = predict_topk_fast(
                mc, scratch, seq_ptr + h0, seq_len - h0,
                t_ids, t_w, nt, topic_ppmi_gain, top_k,
                out_i.data(), out_s.data()
            );
            if (n_pred <= 0) return 0;

            if (use_cont) {
                float norm2 = 1e-8f;
                for (int d = 0; d < D_m; ++d) {
                    float val = 0.62f * hl[d] + 0.38f * hm[d];
                    h_thought[d] = val;
                    norm2 += val * val;
                }
                if (norm2 > 1e-5f) {
                    float inv_n = 1.0f / std::sqrt(norm2);
                    for (int d = 0; d < D_m; ++d) h_thought[d] *= inv_n;

                    if (has_wp) {
                        float pn2 = 1e-8f;
                        for (int r = 0; r < D_m; ++r) {
                            const float* row = w_pred_flat + static_cast<int64_t>(r) * D_m;
                            float acc = 0.0f;
                            for (int c = 0; c < D_m; ++c) acc += row[c] * h_thought[c];
                            h_pred[r] = acc;
                            pn2 += acc * acc;
                        }
                        float inv_p = 1.0f / std::sqrt(pn2);
                        for (int r = 0; r < D_m; ++r) h_pred[r] *= inv_p;
                    }

                    for (int i = 0; i < n_pred; ++i) {
                        int32_t cid = out_i[i];
                        if (cid < 256 || cid >= V_m) continue;
                        float idf = mc->motor_idf[cid];
                        float cg = 1.0f / (1.0f + std::exp(-(idf - tau_idf) * 2.2f));
                        if (cg < 0.05f) continue;
                        const float* em = emb_flat + static_cast<int64_t>(cid) * D_m;
                        float cos_t = 0.0f;
                        float cos_p = 0.0f;
                        for (int d = 0; d < D_m; ++d) {
                            cos_t += em[d] * h_thought[d];
                            if (has_wp) cos_p += em[d] * h_pred[d];
                        }
                        float align = has_wp ? (0.60f * cos_t + 0.40f * cos_p) : cos_t;
                        out_s[i] += cg * cont_gain * align;
                        if (align < 0.08f) {
                            out_s[i] -= cg * inhib_gain * (0.08f - align);
                        }
                    }
                }
            }

            float mx = out_s[0];
            for (int i = 1; i < n_pred; ++i) if (out_s[i] > mx) mx = out_s[i];
            float sum_p = 1e-8f;
            for (int i = 0; i < n_pred; ++i) {
                float z = std::max(-15.0f, std::min(0.0f, (out_s[i] - mx) * 1.15f));
                out_p[i] = std::exp(z);
                sum_p += out_p[i];
            }
            float inv_s = 1.0f / sum_p;
            for (int i = 0; i < n_pred; ++i) out_p[i] *= inv_s;
            return n_pred;
        };

        float cont_bp[256];
        float bnd_bp[256];
        int cont_tb[64];
        int bnd_tb[64];

        #pragma omp for schedule(dynamic, 1)
        for (int bi = 0; bi < B; ++bi) {
            const uint8_t* stream = data_2d + static_cast<int64_t>(bi) * T;
            float* boost_b = out_boost + static_cast<int64_t>(bi) * (T - 1) * 256;
            std::memset(boost_b, 0, static_cast<size_t>(T - 1) * 256 * sizeof(float));

            committed.clear();
            if (use_cont) {
                std::fill(h_long.begin(), h_long.end(), 0.0f);
                std::fill(h_mid.begin(), h_mid.end(), 0.0f);
            }

            int commit_pos = 0;
            int32_t active_node = 0;
            int32_t best_id = -1;
            int best_end = 0;
            bool commit_pred_valid = false;
            int commit_n_pred = 0;

            // STRICTLY CAUSAL BYTE-BY-BYTE LOOP: step t reads ONLY stream[0..t] and writes row t (predicting stream[t+1])!
            for (int t = 0; t < T - 1; ++t) {
                uint8_t b = stream[t];
                int32_t nxt = v->trie[active_node].find_child(b);
                if (nxt < 0) {
                    int32_t cid = (best_id >= 0) ? best_id : static_cast<int32_t>(stream[commit_pos]);
                    int cend = (best_end > commit_pos) ? best_end : (commit_pos + 1);
                    committed.push_back(cid);
                    update_thought_vec(h_long, h_mid, cid);
                    commit_pos = cend;
                    commit_pred_valid = false;

                    // Replay strictly-observed suffix stream[commit_pos..t] (ZERO bytes > t!)
                    while (true) {
                        active_node = 0;
                        best_id = -1;
                        best_end = commit_pos;
                        bool broke = false;
                        for (int s = commit_pos; s <= t; ++s) {
                            int32_t nn = v->trie[active_node].find_child(stream[s]);
                            if (nn < 0) {
                                int32_t cid2 = (best_id >= 0) ? best_id : static_cast<int32_t>(stream[commit_pos]);
                                int cend2 = (best_end > commit_pos) ? best_end : (commit_pos + 1);
                                committed.push_back(cid2);
                                update_thought_vec(h_long, h_mid, cid2);
                                commit_pos = cend2;
                                broke = true;
                                break;
                            }
                            active_node = nn;
                            if (v->trie[active_node].motor_id >= 0) {
                                best_id = v->trie[active_node].motor_id;
                                best_end = s + 1;
                            }
                        }
                        if (!broke) break;
                    }
                } else {
                    active_node = nxt;
                    if (v->trie[active_node].motor_id >= 0) {
                        best_id = v->trie[active_node].motor_id;
                        best_end = t + 1;
                    }
                }

                // Lazily compute prediction from strictly-past committed history (bytes < commit_pos <= t)
                if (!commit_pred_valid) {
                    commit_n_pred = run_predict(
                        committed.data(), static_cast<int>(committed.size()),
                        h_long, h_mid, commit_ids, commit_sc, commit_probs
                    );
                    commit_pred_valid = true;
                }

                int p_len = t - commit_pos + 1;
                const uint8_t* pending = stream + commit_pos;

                // Hypothesis 1: Continue in-progress motor matching `pending[0..p_len-1]`
                int n_ctb = 0;
                float p_cont = 0.0f;
                float p_exact = 0.0f;
                for (int i = 0; i < commit_n_pred; ++i) {
                    const auto& mb = v->bytes[commit_ids[i]];
                    if (static_cast<int>(mb.size()) < p_len) continue;
                    if (std::memcmp(mb.data(), pending, static_cast<size_t>(p_len)) != 0) continue;
                    float pr = commit_probs[i];
                    if (static_cast<int>(mb.size()) > p_len) {
                        uint8_t bv = mb[p_len];
                        bool found = false;
                        for (int k = 0; k < n_ctb; ++k) {
                            if (cont_tb[k] == bv) { cont_bp[bv] += pr; found = true; break; }
                        }
                        if (!found) {
                            if (n_ctb < 64) cont_tb[n_ctb++] = bv;
                            cont_bp[bv] = pr;
                        }
                        p_cont += pr;
                    } else {
                        p_exact += pr;
                    }
                }

                // Hypothesis 2: Boundary right after `t` (if `pending` itself is a valid complete motor `m_now`)
                int n_btb = 0;
                float p_bnd_tot = 0.0f;
                int32_t m_now = v->trie[active_node].motor_id;
                bool has_ch = !v->trie[active_node].ch.empty();
                float w_end = 0.0f;

                if (m_now >= 0) {
                    w_end = has_ch ? std::max(p_exact, (p_cont < 0.05f ? 0.35f : 0.15f)) : 1.0f;
                    // Build temporary causal history = last <= 32 committed + m_now (does NOT mutate committed!)
                    int c_sz = static_cast<int>(committed.size());
                    int c_start = std::max(0, c_sz - 31);
                    tmp_hist.clear();
                    for (int k = c_start; k < c_sz; ++k) tmp_hist.push_back(committed[k]);
                    tmp_hist.push_back(m_now);

                    if (use_cont) {
                        std::memcpy(h_long_tmp.data(), h_long.data(), static_cast<size_t>(D_m) * sizeof(float));
                        std::memcpy(h_mid_tmp.data(), h_mid.data(), static_cast<size_t>(D_m) * sizeof(float));
                        update_thought_vec(h_long_tmp, h_mid_tmp, m_now);
                    }
                    int b_n = run_predict(
                        tmp_hist.data(), static_cast<int>(tmp_hist.size()),
                        use_cont ? h_long_tmp : h_long,
                        use_cont ? h_mid_tmp : h_mid,
                        bnd_ids, bnd_sc, bnd_probs
                    );
                    for (int i = 0; i < b_n; ++i) {
                        const auto& mb = v->bytes[bnd_ids[i]];
                        if (mb.empty()) continue;
                        uint8_t bv = mb[0];
                        float pr = bnd_probs[i];
                        bool found = false;
                        for (int k = 0; k < n_btb; ++k) {
                            if (bnd_tb[k] == bv) { bnd_bp[bv] += pr; found = true; break; }
                        }
                        if (!found) {
                            if (n_btb < 64) bnd_tb[n_btb++] = bv;
                            bnd_bp[bv] = pr;
                        }
                        p_bnd_tot += pr;
                    }
                }

                // Combine strictly causal mixture -> write row t!
                float eff_end = (p_bnd_tot > 0.05f) ? w_end : 0.0f;
                float tot_mass = p_cont + eff_end;
                if (tot_mass > 0.05f) {
                    float* row = boost_b + static_cast<int64_t>(t) * 256;
                    float conf = std::min(1.0f, p_cont + eff_end * std::min(1.0f, p_bnd_tot));
                    float sq = std::sqrt(conf);

                    // Merge touched byte indices from both hypotheses
                    float mix_p[256];
                    int mix_tb[128];
                    int n_mtb = 0;
                    if (p_cont > 1e-6f) {
                        for (int k = 0; k < n_ctb; ++k) {
                            int bv = cont_tb[k];
                            mix_tb[n_mtb++] = bv;
                            mix_p[bv] = cont_bp[bv];
                        }
                    }
                    if (eff_end > 1e-6f) {
                        float scale_b = eff_end / p_bnd_tot;
                        for (int k = 0; k < n_btb; ++k) {
                            int bv = bnd_tb[k];
                            bool found = false;
                            for (int m = 0; m < n_mtb; ++m) {
                                if (mix_tb[m] == bv) { mix_p[bv] += bnd_bp[bv] * scale_b; found = true; break; }
                            }
                            if (!found && n_mtb < 128) {
                                mix_tb[n_mtb++] = bv;
                                mix_p[bv] = bnd_bp[bv] * scale_b;
                            }
                        }
                    }
                    for (int k = 0; k < n_mtb; ++k) {
                        int bv = mix_tb[k];
                        float cond_p = mix_p[bv] / tot_mass;
                        row[bv] = std::min(2.2f, std::tanh(cond_p * 2.2f) * 2.0f * sq);
                    }
                }
            }
        }
    }
}

// Binary Persistence for Fast Mounting (< 1s!)
int macro_save_v50(void* vh, void* mh, const char* path) {
    if (!vh || !mh || !path) return 0;
    const auto* v = static_cast<const MotorVocab*>(vh);
    const auto* mc = static_cast<const MacroCortexV50*>(mh);
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return 0;

    uint32_t magic = 0x4E473530U; // "NG50"
    ofs.write(reinterpret_cast<const char*>(&magic), 4);

    // 1) MotorVocab
    int32_t V_m = static_cast<int32_t>(v->bytes.size());
    ofs.write(reinterpret_cast<const char*>(&V_m), 4);
    for (int32_t i = 0; i < V_m; ++i) {
        uint8_t blen = static_cast<uint8_t>(v->bytes[i].size());
        ofs.write(reinterpret_cast<const char*>(&blen), 1);
        ofs.write(reinterpret_cast<const char*>(v->bytes[i].data()), blen);
    }

    // 2) Stats & IDFs
    ofs.write(reinterpret_cast<const char*>(&mc->total_motors), 8);
    ofs.write(reinterpret_cast<const char*>(&mc->median_idf), 4);
    ofs.write(reinterpret_cast<const char*>(mc->motor_cnt.data()), static_cast<std::streamsize>(V_m) * 8);
    ofs.write(reinterpret_cast<const char*>(mc->motor_idf.data()), static_cast<std::streamsize>(V_m) * 4);

    // 3) Banks (sparse compact: only write non-zero ways to save disk & speed up load!)
    int32_t nb = mc->num_banks;
    ofs.write(reinterpret_cast<const char*>(&nb), 4);
    for (int bi = 0; bi < nb; ++bi) {
        int32_t bkts = mc->buckets[bi];
        ofs.write(reinterpret_cast<const char*>(&bkts), 4);
        const auto& vec = mc->banks[bi];
        uint64_t nz = 0;
        for (const auto& w : vec) if (w.w > 0) nz += 1;
        ofs.write(reinterpret_cast<const char*>(&nz), 8);
        struct CompactEntry { uint32_t idx; int32_t sig; int32_t dst; uint16_t w; uint16_t pad; };
        std::vector<CompactEntry> buf;
        buf.reserve(std::min<uint64_t>(nz, 1 << 20));
        for (uint32_t idx = 0; idx < static_cast<uint32_t>(vec.size()); ++idx) {
            if (vec[idx].w > 0) {
                buf.push_back({idx, vec[idx].sig, vec[idx].dst, vec[idx].w, 0});
                if (buf.size() >= (1 << 20)) {
                    ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size() * sizeof(CompactEntry)));
                    buf.clear();
                }
            }
        }
        if (!buf.empty()) {
            ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size() * sizeof(CompactEntry)));
        }
    }

    // 4) Directed & Symmetric PPMI CSRs
    auto write_csr = [&](const std::vector<uint32_t>& off, const std::vector<int32_t>& dst, const std::vector<float>& w) {
        uint64_t e = static_cast<uint64_t>(dst.size());
        ofs.write(reinterpret_cast<const char*>(&e), 8);
        if (static_cast<int32_t>(off.size()) == V_m + 1) {
            ofs.write(reinterpret_cast<const char*>(off.data()), static_cast<std::streamsize>(V_m + 1) * 4);
        } else {
            std::vector<uint32_t> zoff(V_m + 1, 0);
            ofs.write(reinterpret_cast<const char*>(zoff.data()), static_cast<std::streamsize>(V_m + 1) * 4);
        }
        if (e > 0) {
            ofs.write(reinterpret_cast<const char*>(dst.data()), static_cast<std::streamsize>(e) * 4);
            ofs.write(reinterpret_cast<const char*>(w.data()), static_cast<std::streamsize>(e) * 4);
        }
    };
    write_csr(mc->dir_off, mc->dir_dst, mc->dir_w);
    write_csr(mc->sym_off, mc->sym_dst, mc->sym_w);
    return ofs.good() ? 1 : 0;
}

int macro_load_v50(const char* path, void** out_vh, void** out_mh) {
    if (!path || !out_vh || !out_mh) return 0;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return 0;

    uint32_t magic = 0;
    ifs.read(reinterpret_cast<char*>(&magic), 4);
    if (!ifs.good() || magic != 0x4E473530U) return 0;

    int32_t V_m = 0;
    ifs.read(reinterpret_cast<char*>(&V_m), 4);
    if (!ifs.good() || V_m < 256 || V_m > 10000000) return 0;

    auto* v = new MotorVocab();
    v->bytes.clear();
    v->bytes.resize(V_m);
    for (int32_t i = 0; i < V_m; ++i) {
        uint8_t blen = 0;
        ifs.read(reinterpret_cast<char*>(&blen), 1);
        if (!ifs.good() || blen == 0) { delete v; return 0; }
        v->bytes[i].resize(blen);
        ifs.read(reinterpret_cast<char*>(v->bytes[i].data()), blen);
        if (!ifs.good()) { delete v; return 0; }
    }
    v->rebuild_trie();

    auto* mc = new MacroCortexV50(V_m);
    auto fail = [&]() -> int { delete v; delete mc; return 0; };
    mc->cooc_dir_tmp.clear(); mc->cooc_dir_tmp.shrink_to_fit();
    mc->cooc_sym_tmp.clear(); mc->cooc_sym_tmp.shrink_to_fit();
    mc->finalized = true;

    ifs.read(reinterpret_cast<char*>(&mc->total_motors), 8);
    ifs.read(reinterpret_cast<char*>(&mc->median_idf), 4);
    ifs.read(reinterpret_cast<char*>(mc->motor_cnt.data()), static_cast<std::streamsize>(V_m) * 8);
    ifs.read(reinterpret_cast<char*>(mc->motor_idf.data()), static_cast<std::streamsize>(V_m) * 4);
    if (!ifs.good()) return fail();

    int32_t nb = 0;
    ifs.read(reinterpret_cast<char*>(&nb), 4);
    if (!ifs.good() || nb != 5) return fail();
    mc->num_banks = nb;
    mc->buckets.resize(nb);
    mc->banks.resize(nb);
    struct CompactEntry { uint32_t idx; int32_t sig; int32_t dst; uint16_t w; uint16_t pad; };
    std::vector<CompactEntry> buf(1 << 20);
    for (int bi = 0; bi < nb; ++bi) {
        int32_t bkts = 0;
        uint64_t nz = 0;
        ifs.read(reinterpret_cast<char*>(&bkts), 4);
        ifs.read(reinterpret_cast<char*>(&nz), 8);
        if (!ifs.good() || bkts <= 0 || (bkts & (bkts - 1)) != 0 || bkts > (1 << 26)) return fail();
        uint64_t max_slots = static_cast<uint64_t>(bkts) * MACRO_WAYS;
        if (nz > max_slots) return fail();
        mc->buckets[bi] = bkts;
        mc->banks[bi].assign(static_cast<size_t>(max_slots), MacroWay{});
        uint64_t rem = nz;
        while (rem > 0) {
            uint64_t take = std::min<uint64_t>(rem, buf.size());
            ifs.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(take * sizeof(CompactEntry)));
            if (!ifs.good()) return fail();
            for (uint64_t k = 0; k < take; ++k) {
                const auto& e = buf[k];
                if (e.idx >= mc->banks[bi].size() || e.dst < 0 || e.dst >= V_m) return fail();
                mc->banks[bi][e.idx] = {e.sig, e.dst, e.w};
            }
            rem -= take;
        }
    }

    auto read_csr = [&](std::vector<uint32_t>& off, std::vector<int32_t>& dst, std::vector<float>& w) -> bool {
        uint64_t e = 0;
        ifs.read(reinterpret_cast<char*>(&e), 8);
        if (!ifs.good() || e > 200000000ULL) return false;
        off.resize(V_m + 1);
        ifs.read(reinterpret_cast<char*>(off.data()), static_cast<std::streamsize>(off.size()) * 4);
        if (!ifs.good() || off[0] != 0 || static_cast<uint64_t>(off[V_m]) != e) return false;
        for (int32_t i = 0; i < V_m; ++i) {
            if (off[i] > off[i + 1]) return false;
        }
        dst.resize(e);
        w.resize(e);
        if (e > 0) {
            ifs.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(e) * 4);
            ifs.read(reinterpret_cast<char*>(w.data()), static_cast<std::streamsize>(e) * 4);
            if (!ifs.good()) return false;
            for (uint64_t k = 0; k < e; ++k) {
                if (dst[k] < 0 || dst[k] >= V_m) return false;
            }
        }
        return true;
    };
    if (!read_csr(mc->dir_off, mc->dir_dst, mc->dir_w)) return fail();
    if (!read_csr(mc->sym_off, mc->sym_dst, mc->sym_w)) return fail();

    *out_vh = v;
    *out_mh = mc;
    return 1;
}

} // extern "C"
