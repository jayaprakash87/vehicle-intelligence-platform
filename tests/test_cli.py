"""CLI smoke tests — verify each command runs without crashing."""

from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()


def _run_dir(tmp_path):
    """Return the first (and only) run-ID subdirectory inside tmp_path."""
    dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(dirs) == 1, f"Expected exactly one run dir, found {dirs}"
    return dirs[0]


def test_simulate(tmp_path):
    result = runner.invoke(app, [
        "simulate",
        "--config", "configs/default.yaml",
        "--output", str(tmp_path),
        "--format", "parquet",
    ])
    assert result.exit_code == 0, result.output
    rd = _run_dir(tmp_path)
    assert (rd / "telemetry.parquet").exists()
    assert (rd / "labels.parquet").exists()


def test_pipeline(tmp_path):
    result = runner.invoke(app, [
        "pipeline",
        "--config", "configs/default.yaml",
        "--output", str(tmp_path),
        "--format", "parquet",
    ])
    assert result.exit_code == 0, result.output
    rd = _run_dir(tmp_path)
    assert (rd / "telemetry.parquet").exists()
    assert (rd / "scored.parquet").exists()


def test_train_after_simulate(tmp_path):
    """Train requires data — simulate first, then train."""
    runner.invoke(app, [
        "simulate",
        "--config", "configs/default.yaml",
        "--output", str(tmp_path),
        "--format", "parquet",
    ])
    rd = _run_dir(tmp_path)
    result = runner.invoke(app, [
        "train",
        "--config", "configs/default.yaml",
        "--data", str(rd / "telemetry.parquet"),
    ])
    assert result.exit_code == 0, result.output


def test_infer_after_train(tmp_path):
    """Infer should work with or without a trained model."""
    runner.invoke(app, [
        "simulate",
        "--config", "configs/default.yaml",
        "--output", str(tmp_path / "sim"),
    ])
    sim_rd = _run_dir(tmp_path / "sim")
    result = runner.invoke(app, [
        "infer",
        "--config", "configs/default.yaml",
        "--data", str(sim_rd / "telemetry.parquet"),
        "--output", str(tmp_path / "infer"),
    ])
    assert result.exit_code == 0, result.output


def test_edge_max_iter(tmp_path):
    """Edge command with --max-iter should finish quickly."""
    runner.invoke(app, [
        "simulate",
        "--config", "configs/default.yaml",
        "--output", str(tmp_path / "sim"),
    ])
    sim_rd = _run_dir(tmp_path / "sim")
    result = runner.invoke(app, [
        "edge",
        "--config", "configs/default.yaml",
        "--data", str(sim_rd / "telemetry.parquet"),
        "--output", str(tmp_path / "edge"),
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
    rd = _run_dir(tmp_path)
    assert (rd / "telemetry.csv").exists()


def test_missing_data_file_graceful_error(tmp_path):
    """Referencing a nonexistent data file should give a clean error, not a traceback."""
    result = runner.invoke(app, [
        "train",
        "--data", str(tmp_path / "does_not_exist.parquet"),
    ])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_bad_config_file_graceful_error(tmp_path):
    """A malformed config should give a clean error."""
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("simulation:\n  duration_s: not_a_number\n")
    result = runner.invoke(app, [
        "simulate",
        "--config", str(bad_cfg),
        "--output", str(tmp_path),
    ])
    assert result.exit_code == 1
    assert "error" in result.output.lower()


def test_run_id_in_output_path(tmp_path):
    """Each run should create a unique timestamped subdirectory."""
    result1 = runner.invoke(app, ["simulate", "--output", str(tmp_path)])
    result2 = runner.invoke(app, ["simulate", "--output", str(tmp_path)])
    assert result1.exit_code == 0
    assert result2.exit_code == 0
    dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(dirs) == 2, f"Expected 2 run dirs, got {dirs}"
