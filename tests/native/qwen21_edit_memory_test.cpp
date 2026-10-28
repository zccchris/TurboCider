#include "../../native/models/qwen21/memory_policy.hpp"
#include <cassert>

int main() {
    using namespace tc;
    using namespace tc::qwen21;
    const DeviceInfo qualified{"Apple M5 Pro", 24ull << 30};
    const DeviceInfo other{"Apple M5 Pro", 48ull << 30};
    Request r;
    r.model = "qwen-image-2.1"; r.operation = "image.edit";
    r.residency = "component_staged"; r.audio = false; r.steps = 25;
    r.inputs.push_back({.kind="image", .role="reference", .path="not-read.png"});
    r.qwen21_dit_cache_explicit = true;
    assert(layer_staged_edit(r, qualified));
    assert(!layer_staged_edit(r, other));
    assert(!qualified_staged_optimizations(r, qualified));
    assert(!layer_staged_hybrid_t2i(r, qualified));
    r.prompt_enhancer_path = "/persisted-but-disabled-pe";
    assert(layer_staged_edit(r, qualified));
    for (const auto *cache : {"off", "conservative", "balanced", "fast"}) {
        r.qwen21_dit_cache = cache;
        for (int count : {1, 2, 3}) {
            r.inputs.resize(count, r.inputs.front());
            for (int size : {512, 1024}) {
                r.qwen21_reference_size = size;
                r.allow_approximation = size == 512 || std::string_view(cache) != "off";
                assert(layer_staged_edit(r, qualified));
                const uint64_t tokens = uint64_t(count) * (size / 16) * (size / 16);
                const auto estimate = staged_edit_memory_estimate(r, tokens);
                assert(estimate >= (16ull << 30) && estimate + (4ull << 30) <= qualified.physical_memory);
                const auto rounded = staged_edit_memory_estimate(r, tokens * 2);
                assert(rounded >= estimate);
                if (count == 3 && size == 1024)
                    assert(rounded + (4ull << 30) > qualified.physical_memory);
            }
        }
    }
    auto reject = [&](const Request &candidate) { assert(!layer_staged_edit(candidate, qualified)); };
    auto copy = r; copy.width = 768; reject(copy);
    copy = r; copy.residency = "resident"; reject(copy);
    copy = r; copy.inputs.push_back(r.inputs.front()); reject(copy);
    copy = r; copy.inputs.clear(); reject(copy);
    copy = r; copy.prompt_enhance = true; reject(copy);
    copy = r; copy.loras.push_back({.path="adapter.safetensors"}); reject(copy);
    copy = r; copy.qwen21_dit_w8a8 = true; reject(copy);
    copy = r; copy.qwen21_denoiser_backend = "native"; reject(copy);
    copy = r; copy.execution = "gpu_ane"; reject(copy);
    copy = r; copy.encoder_ane_manifest = "encoder.json"; reject(copy);
    Request hybrid = r;
    hybrid.qwen21_reference_size = 512; hybrid.execution = "gpu_ane";
    hybrid.qwen21_w8a8 = true; hybrid.allow_approximation = true;
    hybrid.ane_manifest = "verified-at-execution.json";
    for (int count : {1, 2, 3}) {
        hybrid.inputs.resize(count, r.inputs.front());
        for (int steps : {20, 25, 40}) for (const auto *mode : {"auto", "base_fused"})
            for (const auto *cache : {"off", "conservative", "balanced", "fast"})
                for (bool full_dit : {false, true}) {
                    hybrid.steps = steps; hybrid.hybrid_mlp_mode = mode;
                    hybrid.qwen21_dit_cache = cache; hybrid.qwen21_dit_w8a8 = full_dit;
                    assert(layer_staged_hybrid_edit(hybrid, qualified));
                    assert(!layer_staged_hybrid_edit(hybrid, other));
                    assert(!layer_staged_edit(hybrid, qualified));
                    assert(!qualified_staged_optimizations(hybrid, qualified));
                    const auto options = staged_optimization_options(hybrid, qualified);
                    assert(options.qk_norm_rope && options.dit_w8a8 == full_dit);
                    assert(options.attention_bits == (full_dit ? 8 : 0));
                    assert(options.suffix_bits == (full_dit ? 8 : 0));
                    const auto estimate = staged_hybrid_edit_memory_estimate(hybrid, count * 1024);
                    assert(estimate + (4ull << 30) <= qualified.physical_memory);
                    assert(staged_hybrid_edit_memory_estimate(hybrid, count * 1152) >= estimate);
                }
    }
    assert(staged_hybrid_edit_memory_estimate(hybrid, 3 * 4096) + (4ull << 30) > qualified.physical_memory);
    auto reject_hybrid = [&](const Request &candidate) { assert(!layer_staged_hybrid_edit(candidate, qualified)); };
    copy = hybrid; copy.qwen21_reference_size = 256; reject_hybrid(copy);
    copy = hybrid; copy.qwen21_reference_size = 1024; reject_hybrid(copy);
    copy = hybrid; copy.inputs.clear(); reject_hybrid(copy);
    copy = hybrid; copy.inputs.push_back(hybrid.inputs.front()); reject_hybrid(copy);
    copy = hybrid; copy.steps = 19; reject_hybrid(copy);
    copy = hybrid; copy.steps = 41; reject_hybrid(copy);
    copy = hybrid; copy.qwen21_w8a8 = false; reject_hybrid(copy);
    copy = hybrid; copy.qwen21_dit_smoothquant_convrot = true; copy.qwen21_dit_w8a8 = true;
    assert(layer_staged_hybrid_edit(copy, qualified));
    assert(staged_optimization_options(copy, qualified).dit_w8a8);
    copy.qwen21_dit_w8a8 = false;
    bool rejected_sq = false;
    try { (void)staged_optimization_options(copy, qualified); }
    catch (const std::exception &) { rejected_sq = true; }
    assert(rejected_sq);
    copy = hybrid; copy.hybrid_mlp_mode = "runtime"; reject_hybrid(copy);
    copy = hybrid; copy.prompt_enhance = true; reject_hybrid(copy);
    copy = hybrid; copy.width = 768; reject_hybrid(copy);
    copy = hybrid; copy.qwen21_gpu_w8a16 = true; reject_hybrid(copy);
    copy = hybrid; copy.qwen21_gpu_full_ffn_blocks = {3, 5, 7}; reject_hybrid(copy);
    copy = hybrid; copy.loras.push_back({.path="adapter.safetensors"}); reject_hybrid(copy);
    copy = hybrid; copy.encoder_ane_manifest = "encoder.json"; reject_hybrid(copy);
    setenv("TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS", "1", 1);
    reject_hybrid(hybrid);
    unsetenv("TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS");
    setenv("TURBOCIDER_QWEN21_STAGED_FAST_DEFAULTS", "0", 1);
    hybrid.qwen21_dit_w8a8 = false;
    assert(layer_staged_hybrid_edit(hybrid, qualified));
    assert(!staged_optimization_options(hybrid, qualified).qk_norm_rope);
    unsetenv("TURBOCIDER_QWEN21_STAGED_FAST_DEFAULTS");
    return 0;
}
