# ONNX variants (H3 method track)

Built with onnxruntime-genai 0.18.0 model builder (CPU EP) from the same pinned HF revision as the GGUFs. Local files only.

| Model | Variant | Builder args | Data size (MiB) | Bits/weight (file) | SHA256 model.onnx.data (first 16) | Smoke |
|---|---|---|---|---|---|---|
| qwen3-0.6b | ONNX int8_cpu_b32 | `-p int8 -e cpu --extra_options block_size=32` | 1,253 | 13.989 | `c459ecb2951a9ce9` | T=1 sample coherent; 16-bit probe parse 10/10; seeds reproduce |
| qwen3-0.6b | ONNX int4_rtn_cpu_b32 | `-p int4 -e cpu --extra_options block_size=32 algo_config=rtn` | 376 | 4.194 | `7ab4e61b0a8ab6c0` | garbled factual answer; 16-bit probe parse 4/10 (spaces inserted) |

Caveats for H3: the ONNX INT8 recipe leaves embeddings and the LM head in fp32 (about 13.3 bits/weight overall vs 8.5 for GGUF Q8_0), the INT4 RTN recipe is harsher than Q4_K_M, and the runtime differs from llama-server. Method and runtime are confounded; results will say so.
