"""Command line entry point: ``asp train | predict | serve``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .pipeline import MODELS


def _train(args) -> int:
    from .train import TrainConfig, run

    cfg = TrainConfig(
        data=args.data,
        out=args.out,
        include_curricular=args.include_curricular,
        models=args.models,
        cv_folds=args.cv_folds,
        seed=args.seed,
        z_limit=None if args.no_outlier_filter else args.z_limit,
    )
    report = run(cfg)
    from .train import results_table

    print(results_table(report))
    return 0


def _predict(args) -> int:
    import pandas as pd

    from .predict import Predictor

    predictor = Predictor.load(args.model)
    df = pd.read_csv(args.input, sep=None, engine="python", encoding="utf-8-sig")
    for row in predictor.predict(df):
        print(json.dumps(row))
    return 0


def _serve(args) -> int:
    import os

    import uvicorn

    os.environ["ASP_MODEL"] = args.model
    uvicorn.run("academic_success.serve:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="asp", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train, select and evaluate models")
    t.add_argument("--data", help="UCI CSV; omit to use synthetic data with the same schema")
    t.add_argument("--out", default="artifacts")
    t.add_argument(
        "--include-curricular",
        action="store_true",
        help="keep the 'Curricular units' semester columns",
    )
    t.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    t.add_argument("--cv-folds", type=int, default=5)
    t.add_argument("--z-limit", type=float, default=4.0)
    t.add_argument("--no-outlier-filter", action="store_true")
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=_train)

    p = sub.add_parser("predict", help="score a CSV of students, one JSON line per row")
    p.add_argument("input")
    p.add_argument("--model", default="artifacts/model.joblib")
    p.set_defaults(func=_predict)

    s = sub.add_parser("serve", help="run the HTTP inference API")
    s.add_argument("--model", default="artifacts/model.joblib")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_serve)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
