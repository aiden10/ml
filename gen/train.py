import argparse
from pathlib import Path

import numpy as np
import torch

from checkpoint import DEFAULT_PATH, load_checkpoint, restore, save_checkpoint
from data import get_corpus, get_vocab, sample_batch
from lm import Transformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--resume",
        nargs="?",
        const=DEFAULT_PATH,
        type=Path,
        default=None,
        help="resume from a checkpoint (defaults to the latest one)",
    )
    parser.add_argument("--max-corpus-tokens", type=int, default=2_000_000)
    return parser.parse_args()


def main():
    args = parse_args()

    vocab = get_vocab()
    corpus = get_corpus(max_tokens=args.max_corpus_tokens)
    print(f"vocab {len(vocab)} tokens, corpus {len(corpus)} tokens")

    model = Transformer(embed_dim=args.embed_dim, vocab=vocab)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    step = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume)
        step = restore(model, optimizer, checkpoint)
        print(f"resumed from {args.resume} at step {step}")

    rng = np.random.default_rng(args.seed)
    running_loss = 0.0

    for step in range(step + 1, args.steps + 1):
        inputs, targets = sample_batch(
            corpus, args.batch_size, args.context_len, rng
        )
        loss = model.train_step(
            torch.from_numpy(inputs), torch.from_numpy(targets), optimizer
        )
        running_loss += loss

        if step % args.log_every == 0:
            print(f"step {step:5d} loss {running_loss / args.log_every:.4f}")
            running_loss = 0.0

        if step % args.save_every == 0 or step == args.steps:
            path = save_checkpoint(
                model, optimizer, step, args.checkpoint, loss=loss
            )
            print(f"saved {path}")


if __name__ == "__main__":
    main()
