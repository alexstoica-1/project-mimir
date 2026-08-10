"""Assign an MLflow alias to a registered model version."""

from __future__ import annotations

import argparse

import mlflow


def main() -> int:
    """Set an MLflow alias, defaulting to the selected global LightGBM model."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="mimir-lightgbm-global")
    parser.add_argument("--version", default="1")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.MlflowClient().set_registered_model_alias(args.model_name, args.alias, args.version)
    print(f"Assigned alias {args.alias!r} to {args.model_name} version {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
