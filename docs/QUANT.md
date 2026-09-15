# Quantizing the surgical model without losing what you just gained

## Order of operations

1. Edit first (`swiftlab edit` / `transfer` / merged LoRA). Quantization must see the final
   tensors; a rank-one edit applied to a quantized model re-introduces rounding error on
   exactly the rows you care about.
2. Roll out the **edited** BF16 model on the `calib` split (`swiftlab rollout --arm edited`).
3. `swiftlab quant` builds the calibration set from those traces: prompt + short thinking +
   answer, correct traces preferred, mixed with any plain text you pass, never from `eval`.
4. Run the generated `quant_gguf.sh` (llama.cpp) and/or `quant_w4a16.py` (llmcompressor).
5. Serve each quant and add it as an arm in `swiftlab eval`. A quant that reverts to long
   thinking shows up as a low `overspend_removed` and a high `think_mean`.

## Why in-distribution calibration matters here

Importance matrices and GPTQ Hessians are estimated from calibration activations. Generic
WikiText calibration under-weights the exact activation directions the edit modified
(the reasoning-loop subspace), so the quantizer is free to spend its error budget there.
Calibrating on the edited model's own reasoning traces puts that subspace in the Hessian.
This is also the fix for the "overthinking errors in PTQ" that started the Swift project:
loops that are rare in BF16 become frequent in a naive Q4 because the quant noise lands in
the subspace that gates re-verification.

## GGUF (llama.cpp)

```
convert_hf_to_gguf.py <edited-bf16> --outtype bf16
llama-imatrix   -m bf16.gguf -f calib.txt -c 8192 --chunks 400 -o imatrix.gguf
llama-quantize  --imatrix imatrix.gguf bf16.gguf out-Q4_K_M.gguf Q4_K_M
llama-perplexity -m bf16.gguf     -f heldout.txt --kl-divergence-base kld-base.bin
llama-perplexity -m out-Q4_K_M.gguf --kl-divergence-base kld-base.bin --kl-divergence
```

Read `Mean KLD`, `99.0% KLD`, `Same top p` (top-1 agreement). Rules of thumb for a
27B reasoning model: mean KLD < 0.02 and top-1 agreement > 95% for Q5_K_M/Q6_K; Q4_K_M
typically 0.03–0.06. Anything above ~0.1 mean KLD will visibly re-introduce loops.
`swiftlab.quant.parse_kl_output` parses that summary.

Recommended ladder for a 27B: `Q4_K_M` (~16 GB), `Q5_K_M` (~19 GB), `Q6_K` (~22 GB),
`Q8_0` (~29 GB). Use `--output-tensor-type q8_0 --token-embedding-type q8_0` when going
below Q4_K_M; the embedding and output rows are where the marker tokens live.

## W4A16 / NVFP4 (vLLM)

`quant_w4a16.py` uses llmcompressor GPTQ with `dampening_frac=0.01`, `ignore=["lm_head"]`,
512 samples × 4096 tokens. For NVFP4 change `scheme="NVFP4"`; for FP8 use the
dynamic per-token FP8 modifier (no calibration needed, but still verify with the eval).
Serve with `vllm serve <out> --reasoning-parser qwen3`.

## Sanity checks before publishing a quant

* paired eval: accuracy delta CI contains 0, `needed_cut` ≈ 0, `truncated` count unchanged
* thinking-token reduction within a few points of the BF16 edited model
* KL vs BF16 as above; compare against the community quant of the *unedited* model to
  confirm the edit survived (it should have lower `think_mean` at the same KL)
* speculative-decoding heads (e.g. DFlash2) trained on the base still work; acceptance
  drops if the token distribution moved too far, which is another KL proxy
