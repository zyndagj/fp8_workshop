"""Tokenize TinyStories into flat uint16 token files.

TinyStories is a corpus of short, simple children's stories written with a small
vocabulary. It is a good fit for this workshop because a GPT-2-sized model starts
producing *readable, complete stories* after only a few minutes of training, so the
"talk to your model" step is actually rewarding at every rung of the ladder.

The raw text is downloaded by the Makefile (`make data`), which uses curl -- the files
are served straight from Hugging Face over plain HTTP, so there is no login, no access
token, and no `datasets` dependency. This script only does the tokenizing.

Input (in --out-dir, default ./data): the two TinyStoriesV2-GPT4-*.txt files.

Output (in --out-dir, default ./data):
    train.bin   uint16 GPT-2 BPE token ids, one flat stream
    val.bin     same, from the held-out split
    meta.json   token counts and the settings used to build them

Usage:
    make data                                 # downloads, then runs this: ~300M train tokens
    python prepare_data.py --max-tokens 50e6  # quicker, if you are short on time or disk
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from helpers import VOCAB_SIZE, get_tokenizer

# The "V2-GPT4" files are the higher-quality subset (GPT-4 generated only). The Makefile
# downloads these names; keep the two in step if you ever change the dataset.
FILES = {
    "train": "TinyStoriesV2-GPT4-train.txt",
    "val": "TinyStoriesV2-GPT4-valid.txt",
}
STORY_SEP = "<|endoftext|>"

# In a terminal we redraw one status line with \r. Piped to a file or captured by a
# notebook cell that would leave thousands of progress lines behind, so keep quiet.
LIVE = sys.stderr.isatty()



def build_split(txt_path, bin_path, max_tokens, enc, batch_size=4096):
    """Stream the raw text file, tokenize in batches, append to a flat uint16 file.

    Batching is what makes this fast: `encode_batch` hands the whole batch to the Rust
    tokenizer, which spreads it across threads. Streaming in batches rather than reading
    the file at once is what keeps memory flat -- the training text is 2.2 GB.
    """
    print(f"  tokenizing -> {os.path.basename(bin_path)}")
    total, start = 0, time.time()

    def stories():
        """Yield one story at a time so we never hold the 2 GB file in memory."""
        buf = ""
        with open(txt_path, encoding="utf-8", errors="ignore") as f:
            while block := f.read(1 << 22):
                buf += block
                *complete, buf = buf.split(STORY_SEP)
                yield from complete
        if buf.strip():
            yield buf

    def batches():
        """Group non-empty stories into lists of `batch_size`."""
        batch = []
        for story in stories():
            story = story.strip()
            if not story:
                continue
            batch.append(story)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    with open(bin_path, "wb") as out:
        for batch in batches():
            # The EOT token is how the model learns where a story stops. Without it, the
            # model would run every story together into one endless stream and never
            # learn to finish.
            ids = [tok for story_ids in enc.encode_batch(batch)
                   for tok in story_ids + [enc.eot_token]]
            if max_tokens and total + len(ids) > max_tokens:
                ids = ids[: max_tokens - total]
            # uint16 is enough because GPT-2's vocabulary is 50257 < 65536, and it
            # halves the size of the token files on disk versus int32.
            out.write(np.asarray(ids, dtype=np.uint16).tobytes())
            total += len(ids)
            if LIVE:
                rate = total / 1e6 / (time.time() - start)
                print(f"\r    {total/1e6:7.1f}M tokens  ({rate:.1f}M tok/s)", end="", flush=True)
            if max_tokens and total >= max_tokens:
                break
    print(f"\r    {total/1e6:7.1f}M tokens  done in {time.time()-start:.0f}s        ")
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="data")
    p.add_argument("--max-tokens", type=float, default=300e6,
                   help="cap on training tokens (0 = use all ~540M)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    enc = get_tokenizer(args.out_dir)
    meta = {"dataset": "TinyStories V2 (GPT-4)", "encoding": "gpt2",
            "vocab_size": VOCAB_SIZE}

    for split, fname in FILES.items():
        print(f"{split}:")
        txt = os.path.join(args.out_dir, fname)
        if not os.path.exists(txt):
            sys.exit(f"{txt} not found -- run `make data` from the repository root, which "
                     f"downloads the raw text and then runs this script.")
        # The validation split is only used to measure loss, so it stays small.
        cap = int(args.max_tokens) if split == "train" else 0
        meta[f"{split}_tokens"] = build_split(
            txt, os.path.join(args.out_dir, f"{split}.bin"), cap, enc
        )

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nready: {meta['train_tokens']/1e6:.0f}M train / "
          f"{meta['val_tokens']/1e6:.0f}M val tokens in {args.out_dir}/")


if __name__ == "__main__":
    sys.exit(main())
