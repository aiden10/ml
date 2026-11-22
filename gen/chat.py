import argparse
from pathlib import Path
import sys

from checkpoint import DEFAULT_PATH, load_checkpoint, restore
from data import build_stoi, decode, encode, get_vocab, tokenize
from lm import Transformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        nargs="?",
        const=DEFAULT_PATH,
        type=Path,
        default=DEFAULT_PATH,
        help="checkpoint to load (defaults to gen/checkpoints/lm.pt)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--context-len", type=int, default=128)
    return parser.parse_args()


def load_model(path: Path, vocab: list[str]) -> Transformer:
    checkpoint = load_checkpoint(path)
    if checkpoint["vocab_size"] != len(vocab):
        raise SystemExit(
            f"checkpoint was trained on {checkpoint['vocab_size']} tokens, "
            f"vocab.json has {len(vocab)}"
        )

    model = Transformer(embed_dim=checkpoint["embed_dim"], vocab=vocab)
    restore(model, None, checkpoint)
    model.eval()
    print(f"loaded {path} (step {checkpoint.get('step', 0)})")
    return model


def main():
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    vocab = get_vocab()
    stoi = build_stoi(vocab)
    model = load_model(args.checkpoint, vocab)
    print("type text to continue it, or 'quit' to exit\n")

    while True:
        try:
            prompt = input("you: ")
        except (EOFError, KeyboardInterrupt):
            break

        if prompt.strip().lower() in {"quit", "exit"}:
            break

        ids = encode(tokenize(prompt), stoi)
        if not ids:
            continue

        out_ids = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            max_context=args.context_len,
        )
        generated = out_ids[len(ids) :]
        print(f"bot: {decode(generated, vocab)}\n")


if __name__ == "__main__":
    main()
