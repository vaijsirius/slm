# minilm — a small GPT-style language model, built from scratch

A byte-level **BPE tokenizer**, a **Transformer language model** (RoPE, pre-norm RMSNorm, SwiGLU, causal attention),
the **training loop** and a **text generator** — all written by hand in PyTorch (no `tiktoken`/`tokenizers`/
`sentencepiece`, no `nn.Transformer`/`nn.MultiheadAttention`/Hugging Face models). It is trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and writes short children's stories.

* Writeup with answers to Questions 1–11, plots and numbers: **[report/WRITEUP.md](report/WRITEUP.md)**
* Model: 4 layers · d_model 512 · 16 heads · d_ff 1,344 · vocab 10,000 · context 256 → **22,696,448 parameters**

## Generate a story (the one command)

Needs only Python ≥ 3.9 and the packages in `requirements.txt`; no GPU, no data, no training code.

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # (CPU wheel: add --index-url https://download.pytorch.org/whl/cpu)
python inference.py --model-dir release --prompt "Once upon a time" \
    --max-new-tokens 256 --temperature 0.8 --top-p 0.9 --seed 42 --device cpu
```

It prints the story and then one line such as `Generated 136 tokens in 0.80s (170.6 tokens/s)` (it stops early if the model emits `<|endoftext|>`).
The same `--seed` gives the same story. `release/` holds exactly four files: `model.pt` (weights only),
`config.json`, `vocab.json`, `merges.txt`.

## Repository layout

```
minilm/
  tokenizer/   bpe.py (train_bpe), tokenizer.py (Tokenizer, vocab/merges files)
  model/       layers.py (RoPE, CausalSelfAttention, FeedForward, hand-written attention), transformer.py
  train/       core.py (loss, AdamW groups, LR schedule, get_batch, train_step, checkpoints)
  generate.py  temperature / top-p sampling, optional KV cache
train.py       training script (all settings via flags or --config; --resume; --overfit-single-batch)
inference.py   run the packaged model from release/
release/       model.pt, config.json, vocab.json, merges.txt
tests/         pytest suite (tokenizer, BPE, attention/RoPE/FFN, model, training pieces, generation)
scripts/       train_tokenizer, encode_dataset, run_checks, run_pilots, make_plots, make_release, analysis & benchmarks
notebooks/     colab_full_run.ipynb – runbook for the full-spec run on a free T4 (only calls the .py files)
report/        WRITEUP.md, figures/, analysis/ (raw outputs behind every number), stories.md
artifacts/     the trained tokenizer (vocab.json, merges.txt) + its training stats
```

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest            # ~1–2 min on a laptop CPU
```

## Reproduce everything

```bash
# 0. data: TinyStoriesV2-GPT4-train.txt and -valid.txt from the roneneldan/TinyStories dataset on Hugging Face
# 1. tokenizer (10k vocab) and the encoded uint16 datasets
python scripts/train_tokenizer.py --input data/TinyStoriesV2-GPT4-train.txt --out-dir artifacts/tokenizer --vocab-size 10000
python scripts/encode_dataset.py  --input data/TinyStoriesV2-GPT4-valid.txt --out data/valid.npy --tokenizer-dir artifacts/tokenizer
python scripts/encode_dataset.py  --input data/TinyStoriesV2-GPT4-train.txt --out data/train.npy --tokenizer-dir artifacts/tokenizer
# 2. correctness checks + learning-rate pilots (batch 32 / 300 steps = the assignment's setup)
python scripts/run_checks.py --train-data data/train.npy --val-data data/valid.npy --runs-dir runs
python scripts/run_pilots.py --train-data data/train.npy --val-data data/valid.npy --runs-dir runs --steps 300 --batch-size 32 \
       --lrs 1e-4 3e-4 1e-3 3e-3 1e-2
# 3. the main run: 5,000 steps x 32 x 256 tokens (fp16 on a T4; add --compile)
python train.py --train-data data/train.npy --val-data data/valid.npy --out-dir runs/main --lr <best> --max-steps 5000 --batch-size 32
python train.py ... --resume          # continue after a disconnect
# 4. package, plots, stories
python scripts/make_release.py --checkpoint runs/main/final_model.pt --tokenizer-dir artifacts/tokenizer --release-dir release
python scripts/make_plots.py --runs-dir runs --out-dir report/figures
python scripts/sample_stories.py --model-dir release --out report/stories.md
```

`notebooks/colab_full_run.ipynb` strings these together for Colab/Kaggle (it saves tokenizer, encoded data and
checkpoints to Drive / `/kaggle/working`).

## What was actually run for the submitted numbers

The submitted run was made on a laptop **CPU (no GPU)**, so the main run is a *scaled-down* version of the
assignment's 5,000-step run — see the "Honest scope" section at the top of the writeup for the exact settings
(steps, batch size, tokens) and what that means for the loss.

## Docker (optional)

```bash
docker build -t minilm .
docker run --rm minilm                       # same defaults as the command above
docker run --rm minilm --prompt "One day, a little dog" --seed 7
```
