import os
import re
import sys
import time
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional

"""
Auto-tune HSD tau annealing epochs for discrete mode by running multiple trainings
and parsing logs to select the best configuration based on validation NDCG@10.

- Anneals `--hsd_tau` from a high temperature (e.g., 5.0) down to `--hsd_tau_final` (e.g., 0.5)
- Uses epoch-level linear schedule: `--hsd_tau_schedule linear --hsd_tau_anneal_scope epoch`
- Searches `--hsd_tau_anneal_epochs` in a provided integer range (default 10..20)
- Requires `--log` to capture metrics; reads `./log/{dataset}/{model_name}/{now_str}/*.txt`

Usage examples:
python experiments/auto_tune_hsd_tau.py \
  --dataset amazon \
  --model_name llm4cdsr \
  --start 10 --end 20 --step 2 \
  --hsd_tau 5.0 --hsd_tau_final 0.5 \
  --extra "--inter_file cloth_sport --llm_emb_file itm_emb_np --user_emb_file usr_profile_emb --local_emb --global_emb --freeze_emb --selector_lambda 0.1 --selector_insert_mode left --train_batch_size 128 --max_len 200 --num_train_epochs 200 --gpu_id 0 --num_workers 4 --log --patience 20 --ts_user 12 --ts_item 13"

Notes:
- This script runs trainings sequentially; ensure enough time and resources.
- The selection metric is `NDCG@10` parsed from Trainer logs (best epoch summary).
- Output summary CSV: `./log/{dataset}/autotune_hsd_tau.csv`.
"""

LOG_ROOT = Path('./log')
PROJECT_ROOT = Path('.')

BEST_EPOCH_RE = re.compile(r"The best epoch is (\d+)")
BEST_RES_RE = re.compile(r"The best results are NDCG@10: ([0-9.]+), HR@10: ([0-9.]+)")


def list_run_dirs(dataset: str, model_name: str) -> List[Path]:
    base = LOG_ROOT / dataset / model_name
    if not base.exists():
        return []
    return [p for p in base.iterdir() if p.is_dir() and p.name != 'default']


def find_new_run_dir(before: List[Path], dataset: str, model_name: str) -> Optional[Path]:
    after = list_run_dirs(dataset, model_name)
    before_set = {p.name for p in before}
    candidates = [p for p in after if p.name not in before_set]
    if candidates:
        # return the most recent by mtime
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    # fallback: latest dir in after
    if after:
        after.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return after[0]
    return None


def find_log_file(run_dir: Path) -> Optional[Path]:
    if run_dir is None or not run_dir.exists():
        return None
    # Typical file name: bs{batch}_lr{lr}.txt
    txt_files = list(run_dir.glob('**/*.txt'))
    if not txt_files:
        txt_files = list(run_dir.glob('*.txt'))
    if not txt_files:
        return None
    # pick the largest file (contains full logs)
    txt_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return txt_files[0]


def parse_best_metrics(log_path: Path) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    if log_path is None or not log_path.exists():
        return None, None, None
    best_epoch = None
    best_ndcg = None
    best_hr = None
    with log_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m1 = BEST_EPOCH_RE.search(line)
            if m1:
                try:
                    best_epoch = int(m1.group(1))
                except Exception:
                    pass
            m2 = BEST_RES_RE.search(line)
            if m2:
                try:
                    best_ndcg = float(m2.group(1))
                    best_hr = float(m2.group(2))
                except Exception:
                    pass
    return best_epoch, best_ndcg, best_hr


def run_training(args_list: List[str], cwd: Optional[Path] = None) -> int:
    proc = subprocess.run(args_list, cwd=str(cwd) if cwd else None)
    return proc.returncode


def write_summary_csv(dataset: str, rows: List[Dict]):
    out_dir = LOG_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'autotune_hsd_tau.csv'
    # write headers if new
    headers = [
        'dataset','model_name','now_str','hsd_tau','hsd_tau_final','schedule','scope',
        'anneal_epochs','best_epoch','best_ndcg','best_hr','seed','extra'
    ]
    import csv
    file_exists = out_csv.exists()
    with out_csv.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--model_name', type=str, default='llm4cdsr')
    parser.add_argument('--start', type=int, default=10, help='min anneal epochs')
    parser.add_argument('--end', type=int, default=20, help='max anneal epochs (inclusive)')
    parser.add_argument('--step', type=int, default=2, help='anneal epochs step')
    parser.add_argument('--hsd_tau', type=float, default=5.0)
    parser.add_argument('--hsd_tau_final', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--python', type=str, default=sys.executable, help='python executable to run main.py')
    parser.add_argument('--extra', type=str, default='', help='extra CLI args string to append to main.py')
    args = parser.parse_args()

    # Build base CLI
    base_cli = [
        args.python, 'main.py',
        '--dataset', args.dataset,
        '--model_name', args.model_name,
        '--hsd_enable',
        '--hsd_mode', 'discrete',
        '--hsd_tau', str(args.hsd_tau),
        '--hsd_tau_final', str(args.hsd_tau_final),
        '--hsd_tau_schedule', 'linear',
        '--hsd_tau_anneal_scope', 'epoch',
        '--gpu_id', str(args.gpu_id),
        '--seed', str(args.seed),
        '--log'
    ]

    # Append extra args if provided
    if args.extra:
        # naive split by spaces; advanced users can quote as needed
        base_cli.extend(args.extra.strip().split())

    results = []
    best_overall = None  # (ndcg, hr, epoch, anneal_epochs, now_str)

    for ae in range(args.start, args.end + 1, args.step):
        before_dirs = list_run_dirs(args.dataset, args.model_name)
        cli = base_cli + ['--hsd_tau_anneal_epochs', str(ae)]
        print(f"[AutoTune] Running anneal_epochs={ae} ...")
        rc = run_training(cli, cwd=PROJECT_ROOT)
        if rc != 0:
            print(f"[AutoTune] Run failed for anneal_epochs={ae} (rc={rc}). Skipping.")
            continue
        run_dir = find_new_run_dir(before_dirs, args.dataset, args.model_name)
        log_file = find_log_file(run_dir)
        best_epoch, best_ndcg, best_hr = parse_best_metrics(log_file)
        now_str = run_dir.name if run_dir else ''
        row = {
            'dataset': args.dataset,
            'model_name': args.model_name,
            'now_str': now_str,
            'hsd_tau': args.hsd_tau,
            'hsd_tau_final': args.hsd_tau_final,
            'schedule': 'linear',
            'scope': 'epoch',
            'anneal_epochs': ae,
            'best_epoch': best_epoch if best_epoch is not None else -1,
            'best_ndcg': best_ndcg if best_ndcg is not None else -1.0,
            'best_hr': best_hr if best_hr is not None else -1.0,
            'seed': args.seed,
            'extra': args.extra,
        }
        results.append(row)
        print(f"[AutoTune] anneal_epochs={ae} -> best_ndcg={row['best_ndcg']}, best_hr={row['best_hr']}, best_epoch={row['best_epoch']} \n  log={log_file}")
        if best_ndcg is not None:
            if best_overall is None or best_ndcg > best_overall[0]:
                best_overall = (best_ndcg, best_hr or -1.0, best_epoch or -1, ae, now_str)

    # write summary
    write_summary_csv(args.dataset, results)

    # report best
    if best_overall is None:
        print("[AutoTune] No successful runs to select from.")
    else:
        ndcg, hr, epoch, ae, now_str = best_overall
        print(f"[AutoTune] Selected anneal_epochs={ae} with NDCG@10={ndcg} (HR@10={hr}, best_epoch={epoch}, run={now_str})")


if __name__ == '__main__':
    main()