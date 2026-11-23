"""Cross-validate a model variant on the IDRiD training set only.

The test set is never touched, so this is what model choices are based on.
Each fold's result is saved as it finishes, and --fold runs a single fold.

Example:
    python cv.py --data-dir data/IDRiD --features pretrained --encoding amplitude \
        --loss cross_entropy --maxiter 1000 --cache-dir data/cache --out results/cv/amplitude_ce
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

import train


def main(argv=None) -> dict | None:
    parser = train.build_parser()
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold", type=int, default=None, help="run only this fold (0-based)")
    args = train.finish_args(parser.parse_args(argv))
    if args.features != "pretrained":
        raise SystemExit("cv.py supports --features pretrained (the autoencoder would see the validation fold)")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    feats, _, labels, _ = train.extract_features(args, cache_dir)
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for k, (tr, va) in enumerate(folds.split(feats, labels)):
        if args.fold is not None and k != args.fold:
            continue
        x_tr, x_va = train.prepare_inputs(feats[tr], feats[va], args)
        summary, *_ = train.fit_and_evaluate(
            x_tr, labels[tr], x_va, labels[va], args, feats[tr], feats[va], verbose=False
        )
        (out / f"fold_{k}.json").write_text(json.dumps(summary, indent=2))
        q = summary["test_results"][args.model]
        print(f"fold {k}: {args.model} balanced acc {q['balanced_accuracy']:.3f}, kappa {q['quadratic_kappa']}",
              flush=True)

    done = sorted(out.glob("fold_*.json"))
    if len(done) < args.folds:
        print(f"{len(done)}/{args.folds} folds done")
        return None
    runs = [json.loads(p.read_text())["test_results"] for p in done]
    agg = {
        model: {
            metric: {
                "mean": float(np.mean([r[model][metric] or 0.0 for r in runs])),
                "std": float(np.std([r[model][metric] or 0.0 for r in runs])),
            }
            for metric in runs[0][model]
        }
        for model in runs[0]
    }
    (out / "cv_summary.json").write_text(json.dumps(agg, indent=2))
    for model, m in agg.items():
        print(f"  {model:<26} bal_acc {m['balanced_accuracy']['mean']:.3f} ± {m['balanced_accuracy']['std']:.3f}"
              f"   qwk {m['quadratic_kappa']['mean']:.3f} ± {m['quadratic_kappa']['std']:.3f}")
    return agg


if __name__ == "__main__":
    main()
