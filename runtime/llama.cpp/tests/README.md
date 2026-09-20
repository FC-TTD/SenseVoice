# Regression tests

Run each runtime tool on a fixed clip (`sample.wav`, ~6 s) and diff the output against
the frozen golden in `golden/`. Catches regressions in the ggml graphs, the FSMN-VAD
state machine, the CIF predictor and CTC decode.

## Run
```bash
# from runtime/llama.cpp/, after building (cmake --build build):
./tests/run_regression.sh              # VAD (tiny model auto-fetched) + any tool whose GGUF is already local
RUN_FULL=1 ./tests/run_regression.sh   # also download the ASR GGUFs from Hugging Face and test every tool

# model-free server CLI, naming, and resource-limit checks:
BIN_DIR="$PWD/build/bin" ./tests/run_server_contract.sh

# REST formats, SSE, malformed/limited WebSockets, and realtime transcription:
BIN_DIR="$PWD/build/bin" MODEL_GGUF=/path/to/sensevoice.gguf \
  VAD_GGUF=/path/to/fsmn-vad.gguf ./tests/run_server_smoke.sh
```
`BIN_DIR` / `MODELS_DIR` override where binaries and GGUFs are found. Exit code is
non-zero if any test fails; tools with no binary or no model are skipped.

## Golden
Captured on Linux x86-64 (the reference platform) with the f16 GGUFs published at
`FunAudioLLM/*-GGUF`. Update a golden file only with a deliberate, reviewed output change.
