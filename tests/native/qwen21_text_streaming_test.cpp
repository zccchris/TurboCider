#include "../../native/models/qwen21/text_encoder.hpp"
#include "../../native/models/qwen21/transformer.hpp"
#include "../../native/models/qwen21/cider_w8a8.hpp"
#include "../../native/runtime/streaming/source_lease.hpp"
#include <mlx/random.h>
#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 2, "expected fixture directory");
        tc::configure_streams();
        tc::mx::set_cache_limit(0);
        tc::qwen21::TextConfig config;
        config.layers = 6; config.heads = 4; config.kv_heads = 2;
        config.head_dim = 16; config.mrope_sections = {4, 2, 2};
        auto positions = tc::mx::broadcast_to(tc::mx::reshape(tc::mx::arange(7, tc::mx::int32), {1, 7}), {3, 7});
        int seed = 0;
        auto random = [&](tc::mx::Shape shape) {
            return tc::mx::astype(tc::mx::random::normal(shape, tc::mx::float32,
                tc::mx::random::key(++seed)) * .03f, tc::mx::bfloat16);
        };
        for (const std::string stem : {"model.", "model.language_model."}) {
            std::unordered_map<std::string, tc::Tensor> data;
            data.emplace(stem + "embed_tokens.weight", random({32, 64}));
            data.emplace(stem + "norm.weight", tc::mx::ones({64}, tc::mx::bfloat16));
            for (int i = 0; i < config.layers; ++i) {
                auto p = stem + "layers." + std::to_string(i);
                for (auto name : {"input_layernorm", "post_attention_layernorm"})
                    data.emplace(p + "." + name + ".weight", tc::mx::ones({64}, tc::mx::bfloat16));
                for (auto name : {"q_norm", "k_norm"})
                    data.emplace(p + ".self_attn." + name + ".weight", tc::mx::ones({16}, tc::mx::bfloat16));
                data.emplace(p + ".self_attn.q_proj.weight", random({64, 64}));
                data.emplace(p + ".self_attn.k_proj.weight", random({32, 64}));
                data.emplace(p + ".self_attn.v_proj.weight", random({32, 64}));
                data.emplace(p + ".self_attn.o_proj.weight", random({64, 64}));
                data.emplace(p + ".mlp.gate_proj.weight", random({128, 64}));
                data.emplace(p + ".mlp.up_proj.weight", random({128, 64}));
                data.emplace(p + ".mlp.down_proj.weight", random({64, 128}));
            }
            auto path = std::filesystem::path(argv[1]) / (stem + "safetensors");
            tc::mx::save_safetensors(path.string(), data);
            data.clear();
            auto embeddings = random({1, 7, 64});
            std::vector<tc::Tensor> deepstack{random({1, 7, 64}), random({1, 7, 64})};
            for (bool final_norm : {false, true}) {
                config.final_norm = final_norm;
                tc::Weights resident; resident.load_file(path);
                tc::qwen21::TextEncoder reference(resident, config);
                std::atomic<bool> cancelled{false};
                auto expected = reference.encode_embeddings(embeddings, positions, 5, {}, cancelled, deepstack);
                tc::mx::eval(expected);
                resident.clear();
                auto run = [&](bool cancel_midway) {
                    tc::Weights streamed; streamed.load_file(path);
                    tc::qwen21::TextEncoder encoder(streamed, config);
                    streamed.erase(stem + "embed_tokens.weight");
                    int released = 0;
                    auto release = [&](const std::string &prefix) {
                        tc::require(prefix == stem + "layers." + std::to_string(released) + ".", "layer release order changed");
                        streamed.erase_prefix(prefix);
                        tc::require(!streamed.has(prefix + "self_attn.q_proj.weight"), "consumed layer remains resident");
                        ++released;
                        if (cancel_midway && released == 2) cancelled.store(true);
                    };
                    try {
                        auto actual = encoder.encode_embeddings(embeddings, positions, 5, {}, cancelled, deepstack, release);
                        tc::mx::eval(actual);
                        tc::require(!cancel_midway && released == config.layers, "streaming coverage incomplete");
                        tc::require(tc::mx::array_equal(expected, actual).item<bool>(), "streamed BF16 output differs");
                    } catch (const tc::Cancelled &) {
                        tc::require(cancel_midway && released == 2, "unexpected streaming cancellation");
                    }
                };
                run(false);
                run(true);
                cancelled.store(false);
                run(false); // cancellation must permit a clean reload/retry
            }
        }
        // Exercise the actual compiled DiT blocks over prefill and two decode
        // passes. Restoring lazy weights must preserve the prefix cache and
        // never execute a graph captured against already-consumed arrays.
        tc::qwen21::TransformerConfig dit_config;
        dit_config.layers = 4; dit_config.heads = 2; dit_config.head_dim = 8;
        dit_config.channels = 4; dit_config.context_dim = 12;
        dit_config.rope_axes = {2, 2, 4};
        std::unordered_map<std::string, tc::Tensor> data;
        for (const auto &[name, shape] : std::vector<std::pair<std::string, tc::mx::Shape>>{
                {"time_text_embed.timestep_embedder.linear_1", {16, 256}},
                {"time_text_embed.timestep_embedder.linear_2", {16, 16}},
                {"modulation.1", {64, 16}}, {"img_in", {16, 4}},
                {"txt_in.in_layer", {24, 12}}, {"txt_in.out_layer", {16, 24}},
                {"norm_out.linear", {16, 16}}, {"proj_out", {4, 16}}})
            data.emplace(name + ".weight", random(shape));
        data.emplace("txt_in.text_norm.weight", random({12}));
        for (int i = 0; i < dit_config.layers; ++i) {
            auto p = "transformer_blocks." + std::to_string(i);
            for (auto name : {"to_q", "to_k", "to_v", "to_out.0"})
                data.emplace(p + ".attn." + name + ".weight", random({16, 16}));
            for (auto name : {"norm_q", "norm_k"})
                data.emplace(p + ".attn." + name + ".weight", tc::mx::ones({8}, tc::mx::bfloat16));
            data.emplace(p + ".img_mlp.gate_up.weight", random({64, 16}));
            data.emplace(p + ".img_mlp.out.weight", random({16, 32}));
        }
        auto dit_path = std::filesystem::path(argv[1]) / "dit.safetensors";
        tc::mx::save_safetensors(dit_path.string(), data);
        data.clear();
        auto initial = random({1, 4, 4}), text = random({1, 3, 12});
        for (int reference_count : {0, 1, 3}) {
            std::vector<tc::qwen21::ReferenceLatents> references;
            for (int i = 0; i < reference_count; ++i)
                references.push_back({random({1, 2 * (i + 1), 4}), {2, i + 1, i}});
            std::atomic<bool> cancelled{false};
            const tc::Event quiet = [](const std::string &, int, int) {};
            tc::Weights resident; resident.load_file(dit_path);
            tc::qwen21::Transformer reference(resident, dit_config);
            tc::streaming::SourceFileIdentity source;
            source.logical_id = "dit"; source.path = dit_path;
            auto lease = tc::streaming::SourceLease::capture({source});
            tc::Weights streamed; streamed.load_lease(lease, {"dit"}, quiet, cancelled);
            tc::qwen21::Transformer model(streamed, dit_config);
            int released = 0;
            model.set_layer_release([&](const std::string &prefix) {
                tc::require(prefix == "transformer_blocks." + std::to_string(released % dit_config.layers) + ".",
                            "DiT layer release order changed");
                streamed.erase_prefix(prefix); ++released;
            });
            auto actual = initial, expected = initial;
            for (int step = 0; step < 6; ++step) {
                if (step) {
                    lease->revalidate_after_drain();
                    streamed.clear();
                    streamed.load_lease(lease, {"dit"}, quiet, cancelled);
                }
                float sigma = 1.f - step * .15f;
                expected = expected + reference.forward(expected, text, sigma, 2, 2, true, nullptr, references) * tc::Tensor(-.15f, tc::mx::bfloat16);
                tc::mx::eval(expected);
                actual = actual + model.forward(actual, text, sigma, 2, 2, true, nullptr, references) * tc::Tensor(-.15f, tc::mx::bfloat16);
                tc::mx::eval(actual);
                tc::require(tc::mx::array_equal(expected, actual).item<bool>(), "streamed DiT trajectory differs");
                tc::require(model.cached_layers() == size_t(dit_config.layers), "streaming lost prefix KV");
                tc::require(model.prefix_matches(text, 2, 2, references), "streaming lost reference geometry");
                tc::require(released == (step + 1) * dit_config.layers, "streamed DiT skipped a layer");
            }
        }
        // Public edit cache presets must preserve the resident cache schedule
        // while each executed layer is retired. Skipped layers remain lazy
        // and are discarded before the next descriptor-backed reload.
        auto cache_config = dit_config; cache_config.layers = 12;
        auto cache_weights = tc::mx::load_safetensors(dit_path.string()).first;
        std::vector<std::pair<std::string, tc::Tensor>> extra;
        for (const auto &[key, value] : cache_weights)
            if (key.starts_with("transformer_blocks.0."))
                for (int i = dit_config.layers; i < cache_config.layers; ++i)
                    extra.emplace_back("transformer_blocks." + std::to_string(i) + "." +
                        key.substr(std::string("transformer_blocks.0.").size()), value);
        for (auto &[key, value] : extra) cache_weights.emplace(key, value);
        extra.clear();
        auto cache_path = std::filesystem::path(argv[1]) / "cache-dit.safetensors";
        tc::mx::save_safetensors(cache_path.string(), cache_weights); cache_weights.clear();
        for (int reference_count : {1, 3}) for (bool split : {false, true}) {
            std::vector<tc::qwen21::ReferenceLatents> references;
            for (int i = 0; i < reference_count; ++i)
                references.push_back({random({1, 2 * (i + 1), 4}), {2, i + 1, i}});
            std::atomic<bool> cancelled{false};
            const tc::Event quiet = [](const std::string &, int, int) {};
            tc::Weights resident; resident.load_file(cache_path);
            tc::qwen21::Transformer reference(resident, cache_config);
            auto lease = tc::streaming::SourceLease::capture({{
                .logical_id="dit", .path=cache_path}});
            tc::Weights streamed; streamed.load_lease(lease, {"dit"}, quiet, cancelled);
            tc::qwen21::Transformer model(streamed, cache_config);
            reference.configure_db_cache(true, .25f, 20, 4);
            model.configure_db_cache(true, .25f, 20, 4);
            int calls = 0;
            bool fail_split = false;
            if (split) {
                // The frozen ANE callback receives only target rows. Use a
                // deterministic approximate FFN to isolate callback/prefix and
                // layer-retirement parity without loading Core ML or a model.
                auto feed = [](int block, const tc::Tensor &input) {
                    tc::require(input.shape() == tc::mx::Shape{1, 4, 16},
                                "edit reference rows leaked into fixed-shape decode callback");
                    return input * tc::Tensor(.125f, input.dtype()) +
                        tc::Tensor(float(block + 1) * .0001f, input.dtype());
                };
                reference.set_decode_mlp(feed);
                model.set_decode_mlp([&, feed](int block, const tc::Tensor &input) {
                    ++calls;
                    if (fail_split && block == 3) throw tc::Cancelled();
                    return feed(block, input);
                });
            }
            int released = 0;
            model.set_layer_release([&](const std::string &prefix) {
                tc::require(prefix == "transformer_blocks." + std::to_string(released) + ".",
                            "cached edit release order changed");
                streamed.erase_prefix(prefix); ++released;
            });
            for (int step = 0; step < 20; ++step) {
                if (step) {
                    lease->revalidate_after_drain(); streamed.clear();
                    streamed.load_lease(lease, {"dit"}, quiet, cancelled);
                }
                released = 0; calls = 0;
                reference.set_db_cache_step(step); model.set_db_cache_step(step);
                if (split && step == 3) {
                    fail_split = true;
                    bool caught = false;
                    try { tc::mx::eval(model.forward(initial, text, .5f, 2, 2, true, nullptr, references)); }
                    catch (const tc::Cancelled &) { caught = true; }
                    tc::require(caught && calls == 4 && released == 3,
                                "split edit did not cancel after retiring completed layers");
                    fail_split = false;
                    lease->revalidate_after_drain(); streamed.clear();
                    streamed.load_lease(lease, {"dit"}, quiet, cancelled);
                    released = 0; calls = 0;
                    model.set_db_cache_step(step);
                }
                const auto before = model.db_cached_steps();
                auto expected = reference.forward(initial, text, .5f, 2, 2, true, nullptr, references);
                tc::mx::eval(expected);
                auto actual = model.forward(initial, text, .5f, 2, 2, true, nullptr, references);
                tc::mx::eval(actual);
                tc::require(tc::mx::array_equal(expected, actual).item<bool>(), "streamed edit DBCache output differs");
                tc::require(model.db_cached_steps() == reference.db_cached_steps(), "streamed edit cache schedule differs");
                const bool skipped = model.db_cached_steps() > before;
                tc::require(released == (skipped ? 8 : cache_config.layers), "streamed cache did not retire the executed layers");
                tc::require(model.cached_layers() == size_t(cache_config.layers), "cached edit lost reference prefix");
                tc::require(model.prefix_matches(text, 2, 2, references), "split edit lost reference identity/geometry");
                tc::require(calls == (split && step ? released : 0),
                            "split edit prefill/decode/cache callback coverage differs");
            }
            tc::require(model.db_cached_steps() > 0, "streamed edit fixture never reused DBCache");
        }
        // Execute the complete Cider prefill -> quantized attention decode
        // path, including the production 32-layer coverage assertion. Earlier
        // split-only fixtures missed failure to configure the Transformer:
        // a quantized suffix can run while its attention bank remains empty.
        auto quant_config = dit_config;
        quant_config.layers = 32; quant_config.heads = 2; quant_config.head_dim = 64;
        quant_config.rope_axes = {16, 16, 32};
        std::unordered_map<std::string, tc::Tensor> quant_weights;
        for (const auto &[name, shape] : std::vector<std::pair<std::string, tc::mx::Shape>>{
                {"time_text_embed.timestep_embedder.linear_1", {128, 256}},
                {"time_text_embed.timestep_embedder.linear_2", {128, 128}},
                {"modulation.1", {512, 128}}, {"img_in", {128, 4}},
                {"txt_in.in_layer", {256, 12}}, {"txt_in.out_layer", {128, 256}},
                {"norm_out.linear", {128, 128}}, {"proj_out", {4, 128}}})
            quant_weights.emplace(name + ".weight", random(shape));
        quant_weights.emplace("txt_in.text_norm.weight", random({12}));
        for (int i = 0; i < quant_config.layers; ++i) {
            const auto p = "transformer_blocks." + std::to_string(i);
            for (const auto *name : {"to_q", "to_k", "to_v", "to_out.0"})
                quant_weights.emplace(p + ".attn." + name + ".weight", random({128, 128}));
            for (const auto *name : {"norm_q", "norm_k"})
                quant_weights.emplace(p + ".attn." + name + ".weight", tc::mx::ones({64}, tc::mx::bfloat16));
            quant_weights.emplace(p + ".img_mlp.gate_up.weight", random({512, 128}));
            quant_weights.emplace(p + ".img_mlp.out.weight", random({128, 256}));
        }
        const auto quant_path = std::filesystem::path(argv[1]) / "quant-dit.safetensors";
        tc::mx::save_safetensors(quant_path.string(), quant_weights); quant_weights.clear();
        auto cider = std::make_shared<tc::qwen21::CiderCache>(
            std::filesystem::canonical(argv[1]) / "cider", std::string(64, 'a'), std::function<void()>{});
        for (int reference_count : {1, 3}) for (bool cached : {false, true}) {
            std::vector<tc::qwen21::ReferenceLatents> references;
            for (int i = 0; i < reference_count; ++i)
                references.push_back({random({1, 2 * (i + 1), 4}), {2, i + 1, i}});
            std::atomic<bool> cancelled{false};
            const tc::Event quiet = [](const std::string &, int, int) {};
            tc::Weights resident; resident.load_file(quant_path);
            tc::qwen21::Transformer reference(resident, quant_config);
            auto lease = tc::streaming::SourceLease::capture({{
                .logical_id="dit", .path=quant_path}});
            tc::Weights streamed; streamed.load_lease(lease, {"dit"}, quiet, cancelled);
            tc::qwen21::Transformer model(streamed, quant_config);
            // The real Q/K fused kernel is tested at its fixed 4096 geometry
            // separately. This fixture uses the same dynamic Cider/attention
            // path with 128-wide projections and all 32 layers.
            reference.configure_staged_optimizations(8, false, cider);
            model.configure_staged_optimizations(8, false, cider);
            reference.set_layer_release([](const std::string &) {});
            int released = 0, calls = 0;
            bool fail_decode = false;
            model.set_layer_release([&](const std::string &prefix) {
                tc::require(prefix == "transformer_blocks." + std::to_string(released) + ".",
                            "W8A8 reference edit released layers out of order");
                streamed.erase_prefix(prefix); ++released;
            });
            auto feed = [](int, const tc::Tensor &input) {
                tc::require(input.shape() == tc::mx::Shape{1, 4, 128},
                            "W8A8 edit reference rows entered fixed-row callback");
                return input * tc::Tensor(.125f, input.dtype());
            };
            reference.set_decode_mlp(feed);
            model.set_decode_mlp([&](int block, const tc::Tensor &input) {
                ++calls;
                if (fail_decode && block == 3) throw tc::Cancelled();
                return feed(block, input);
            });
            if (cached) {
                reference.configure_db_cache(true, .25f, 20, 4);
                model.configure_db_cache(true, .25f, 20, 4);
            }
            for (int step = 0; step < 20; ++step) {
                if (step) {
                    lease->revalidate_after_drain(); streamed.clear();
                    streamed.load_lease(lease, {"dit"}, quiet, cancelled);
                }
                released = 0; calls = 0;
                reference.set_db_cache_step(step); model.set_db_cache_step(step);
                const auto before = model.db_cached_steps();
                auto expected = reference.forward(initial, text, .5f, 2, 2, true, nullptr, references);
                tc::mx::eval(expected);
                auto actual = model.forward(initial, text, .5f, 2, 2, true, nullptr, references);
                tc::mx::eval(actual);
                tc::require(tc::mx::array_equal(expected, actual).item<bool>(),
                            "W8A8 reference prefill/decode differs after dense layer release");
                tc::require(model.prefix_matches(text, 2, 2, references), "W8A8 edit lost reference prefix KV");
                const bool skipped = model.db_cached_steps() > before;
                tc::require(released == (skipped ? 8 : 32) && calls == (step ? released : 0),
                            "W8A8 edit cache/layer/callback coverage differs");
                tc::require(model.staged_attention_cached_layers() == (step ? 32 : 0),
                            "W8A8 edit attention cache did not cover all 32 decode layers");
            }
            if (cached) tc::require(model.db_cached_steps() > 0, "W8A8 edit fixture never skipped cache steps");
            tc::require(model.staged_attention_cache_bytes() > 0, "W8A8 edit did not own its packed attention bank");
            model.reset();
            tc::require(model.staged_attention_cached_layers() == 0 && model.cached_layers() == 0 &&
                            model.staged_attention_cache_bytes() == 0,
                        "W8A8 edit reset retained packed attention or reference prefix");
            auto reload = [&] {
                lease->revalidate_after_drain(); streamed.clear();
                streamed.load_lease(lease, {"dit"}, quiet, cancelled);
                released = 0; calls = 0;
            };
            reload(); model.set_db_cache_step(0);
            tc::mx::eval(model.forward(initial, text, .5f, 2, 2, true, nullptr, references));
            reload(); model.set_db_cache_step(1); fail_decode = true;
            bool caught = false;
            try { tc::mx::eval(model.forward(initial, text, .5f, 2, 2, true, nullptr, references)); }
            catch (const tc::Cancelled &) { caught = true; }
            tc::require(caught && calls == 4 && released == 3 &&
                            model.staged_attention_cached_layers() == 0 && model.cached_layers() == 0 &&
                            model.staged_attention_cache_bytes() == 0,
                        "W8A8 edit cancellation retained partial attention or reference prefix banks");
            fail_decode = false; reference.reset();
            for (int step = 0; step < 2; ++step) {
                reload(); reference.set_db_cache_step(step); model.set_db_cache_step(step);
                auto expected = reference.forward(initial, text, .5f, 2, 2, true, nullptr, references);
                tc::mx::eval(expected);
                auto actual = model.forward(initial, text, .5f, 2, 2, true, nullptr, references);
                tc::mx::eval(actual);
                tc::require(tc::mx::array_equal(expected, actual).item<bool>() &&
                                model.staged_attention_cached_layers() == (step ? 32 : 0),
                            "W8A8 reference edit retry did not rebuild complete identical caches");
            }
        }
        std::cout << "PASS Qwen21 text streaming: exact BF16, both key layouts, raw/final norm, deepstack, padding, cancellation/retry; DiT prefill/decode, six-step generation/edit trajectories, 1/3 references, prefix KV, staged edit DBCache, fixed-row split callbacks, cancellation/retry and full DiT W8A8 edit 32-layer cache coverage\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
