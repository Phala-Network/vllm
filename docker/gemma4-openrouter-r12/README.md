# Gemma4 OpenRouter r12 overlay

This image extends the digest-pinned r11 image with the validated Gemma4 MTP4
runtime assets:

- calibrated static FP8 E4M3 KV scales for the four MTP draft positions;
- Outlines handling for a JSON tail and bonus EOS accepted in one speculative
  scheduler step;
- regression tests for speculative validation and rollback state.

The production serving configuration must keep `VLLM_STATIC_KV_SCALE_PATH`
pointing to `/opt/vllm/kv-scales/gemma4-static-kv-scales-mtp.json` and enable
MTP with four speculative tokens.
