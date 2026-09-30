"""
Runs the whole training pipeline, in order, on CPU:

  1. Part 2 / Phase A  - pretrain the transformer on IMDB review text
  2. Part 2 / Phase B  - fine-tune it as a sentiment classifier (prints accuracy)
  3. Part 1            - train the Shakespeare text generator (optional)

Every stage saves a checkpoint and resumes from it, so if you stop (Ctrl+C) you
can re-run the same command and it continues where it left off.

Usage:
    python run_all.py                    # everything, default steps
    python run_all.py --skip_part1       # only the classifier (what you need for accuracy)
    python run_all.py --finetune_steps 2000
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(label, args):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
    start = time.time()
    result = subprocess.run([sys.executable] + args, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n{label} failed (exit code {result.returncode}). Fix the error above, "
              f"then re-run — completed stages resume from their checkpoints.")
        sys.exit(result.returncode)
    print(f"\n{label} finished in {(time.time() - start) / 60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_steps", type=int, default=2000)
    parser.add_argument("--finetune_steps", type=int, default=1200)
    parser.add_argument("--part1_steps", type=int, default=2000)
    parser.add_argument("--skip_part1", action="store_true")
    args = parser.parse_args()

    run("Phase A: pretraining on IMDB text",
        ["src/pretrain_imdb_lm.py", "--target_total", str(args.pretrain_steps)])
    run("Phase B: fine-tuning the sentiment classifier",
        ["src/finetune_imdb_classifier.py", "--target_total", str(args.finetune_steps)])

    if not args.skip_part1:
        run("Part 1: training the Shakespeare text generator",
            ["src/train.py", "--target_total", str(args.part1_steps)])

    print("\nAll done. Try the demo with:  streamlit run app.py")


if __name__ == "__main__":
    main()
