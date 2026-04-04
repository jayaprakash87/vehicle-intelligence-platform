"""CLI smoke tests — verify each command runs without crashing."""

import pytest
from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()


def test_simulate(tmp_path):
    result = runner.invoke(app, [
        "simulate",
        "--config", "configs/default.yaml",
        "--output", str(tmp_path),
        "--format", "parquet",
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "telemetry.parquet").exists()
    assert (tmp_path / "labels.parquet").exists()


def test_pipeline(tmp_path):
    result = runner.invoke(app, [
        "pipeline",
        "--config", "configs/default.yaml",
        "--output", str(tmp_path),
        "--format", "parquet",
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "telemetry.parquet").exists()
    assert (tmp_path / "scored.parquet").exists()


def test_train_after_simulate(tmp_path):
    """Train requires data — simulate first, then train."""
    runner.invoke(app, [
        "simulate",
        "--config", "configs/default.yaml",
        "--output", str(tmp_path),
        "--format", "parquet",
    ])
    result = runner.invoke(app, [
        "train",
        "--config", "configs/default.yaml",
        "--data", str(tmp_path / "telemetry.parquet"),
    ])
    assert result.exit_code == 0, result.output


def test_infer_after_train(tmp_path):
    """Infer should work with or without a trained model."""
    runner.invoke(app, [
        "simulate",
        "--config", "configs/default.yaml",
        "--output", str(tmp_path),
    ])
    result = runner.invoke(app, [
        "infer",
        "--config", "configs/default.yaml",
        "--data", str(tmp_path / "telemetry.parquet"),
        "--output", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output


def test_edge_max_iter(tmp_path):
    """Edge command with --max-iter should finish quickly."""
    runner.invoke(app, [
        "simulate",
        "--config", "configs/default.yaml",
        "--output", str(tmp_path),
    ])
    result = runner.invoke(app, [
        "edge",
        "--config", "configs/default.yaml",
        "--data", str(tmp_path / "telemetry.parquet"),
        "--output", str(tmp_path),
        "--max-iter", "3",
    ])
    assert result.exit_code == 0, result.output


def test_simulate_csv(tmp_path):
    result = runner.invoke(app, [
        "simulate",
        "--output", str(tmp_path),
        "--format", "csv",
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "telemetry.csv").exists()
