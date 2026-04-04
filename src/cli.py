"""CLI entry point for the Vehicle Intelligence Platform.

Commands:
  simulate   — Generate synthetic telemetry
  train      — Train anomaly detection model
  infer      — Run inference on telemetry data
  edge       — Run the edge runtime loop
  pipeline   — Full pipeline: simulate → train → infer
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="vip", help="Vehicle Intelligence Platform CLI")
console = Console()


@app.command()
def simulate(
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="Config file path"),
    output_dir: str = typer.Option("output", "--output", "-o", help="Output directory"),
    fmt: str = typer.Option("parquet", "--format", "-f", help="Output format: parquet|csv|json"),
) -> None:
    """Generate synthetic eFuse telemetry data."""
    from src.config.models import load_config, default_config
    from src.simulation.generator import TelemetryGenerator
    from src.ingestion.normalizer import Normalizer
    from src.storage.writer import StorageWriter, StorageConfig

    cfg = load_config(config) if Path(config).exists() else default_config()
    cfg.storage.output_dir = output_dir
    cfg.storage.format = fmt

    gen = TelemetryGenerator(cfg.simulation)
    telem_df, labels_df = gen.generate()

    norm = Normalizer()
    telem_df = norm.normalize(telem_df)

    writer = StorageWriter(cfg.storage)
    writer.write_telemetry(telem_df)
    writer.write_labels(labels_df)

    console.print(f"[green]Generated {len(telem_df)} telemetry rows, {len(labels_df)} labels[/green]")


@app.command()
def train(
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    data_path: str = typer.Option("output/telemetry.parquet", "--data", "-d"),
) -> None:
    """Train the anomaly detection model on generated (or provided) telemetry."""
    import pandas as pd
    from src.config.models import load_config, default_config
    from src.features.engine import FeatureEngine
    from src.models.anomaly import AnomalyDetector

    cfg = load_config(config) if Path(config).exists() else default_config()

    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=["timestamp"])

    engine = FeatureEngine(cfg.features)
    feat_df = engine.compute(df)

    detector = AnomalyDetector(cfg.model)
    detector.train(feat_df)
    model_path = detector.save()

    console.print(f"[green]Model trained on {len(feat_df)} samples → saved to {model_path}[/green]")


@app.command()
def infer(
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    data_path: str = typer.Option("output/telemetry.parquet", "--data", "-d"),
    output_dir: str = typer.Option("output", "--output", "-o"),
    fmt: str = typer.Option("parquet", "--format", "-f"),
) -> None:
    """Run inference on telemetry data (batch mode)."""
    import pandas as pd
    from src.config.models import load_config, default_config
    from src.inference.pipeline import InferencePipeline
    from src.models.anomaly import AnomalyDetector
    from src.storage.writer import StorageWriter

    cfg = load_config(config) if Path(config).exists() else default_config()
    cfg.storage.output_dir = output_dir
    cfg.storage.format = fmt

    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=["timestamp"])

    # Load trained model if available
    detector = AnomalyDetector(cfg.model)
    model_file = Path(cfg.model.model_dir) / "anomaly_detector.joblib"
    if model_file.exists():
        detector.load()
        console.print("[dim]Loaded trained anomaly model[/dim]")
    else:
        console.print("[yellow]No trained model found — using heuristic scoring[/yellow]")

    pipeline = InferencePipeline(cfg.features, cfg.model, detector)
    scored = pipeline.run_batch(df)

    writer = StorageWriter(cfg.storage)
    writer.write_scored(scored)

    # Print summary
    _print_summary(scored)


@app.command()
def edge(
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    data_path: str = typer.Option("output/telemetry.parquet", "--data", "-d"),
    output_dir: str = typer.Option("output", "--output", "-o"),
    max_iter: int = typer.Option(0, "--max-iter", help="Max batches (0=unlimited)"),
) -> None:
    """Run the edge runtime loop over telemetry data."""
    import pandas as pd
    from src.config.models import load_config, default_config
    from src.edge.runtime import EdgeRuntime
    from src.inference.pipeline import InferencePipeline
    from src.models.anomaly import AnomalyDetector
    from src.storage.writer import StorageWriter
    from src.transport.mock_can import DataFrameTransport

    cfg = load_config(config) if Path(config).exists() else default_config()
    cfg.storage.output_dir = output_dir

    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=["timestamp"])

    detector = AnomalyDetector(cfg.model)
    model_file = Path(cfg.model.model_dir) / "anomaly_detector.joblib"
    if model_file.exists():
        detector.load()

    transport = DataFrameTransport(df)
    pipeline = InferencePipeline(cfg.features, cfg.model, detector)
    writer = StorageWriter(cfg.storage)
    runtime = EdgeRuntime(transport, pipeline, cfg.edge, writer)

    mi = max_iter if max_iter > 0 else None
    alerts = runtime.run(max_iterations=mi)

    console.print(f"\n[bold]Edge run complete — {len(alerts)} alerts[/bold]")
    for a in alerts[:10]:
        console.print(f"  {a['timestamp']}  {a['channel_id']}  {a['fault']}  score={a['score']:.2f}")
    if len(alerts) > 10:
        console.print(f"  ... and {len(alerts) - 10} more")


@app.command()
def pipeline(
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    output_dir: str = typer.Option("output", "--output", "-o"),
    fmt: str = typer.Option("parquet", "--format", "-f"),
) -> None:
    """Full pipeline: simulate → train → infer — one command demo."""
    import pandas as pd
    from src.config.models import load_config, default_config
    from src.simulation.generator import TelemetryGenerator
    from src.ingestion.normalizer import Normalizer
    from src.features.engine import FeatureEngine
    from src.models.anomaly import AnomalyDetector
    from src.inference.pipeline import InferencePipeline
    from src.storage.writer import StorageWriter

    cfg = load_config(config) if Path(config).exists() else default_config()
    cfg.storage.output_dir = output_dir
    cfg.storage.format = fmt

    console.rule("[bold blue]1. Simulate")
    gen = TelemetryGenerator(cfg.simulation)
    telem_df, labels_df = gen.generate()
    norm = Normalizer()
    telem_df = norm.normalize(telem_df)
    console.print(f"  {len(telem_df)} telemetry rows, {len(labels_df)} labels")

    console.rule("[bold blue]2. Feature Engineering")
    engine = FeatureEngine(cfg.features)
    feat_df = engine.compute(telem_df)
    console.print(f"  {len(feat_df.columns)} columns after feature computation")

    console.rule("[bold blue]3. Train Anomaly Model")
    detector = AnomalyDetector(cfg.model)
    detector.train(feat_df)
    detector.save()
    console.print("  Model trained and saved")

    console.rule("[bold blue]4. Inference")
    pipe = InferencePipeline(cfg.features, cfg.model, detector)
    scored = pipe.run_batch(telem_df)
    console.print(f"  {scored['is_anomaly'].sum()} anomalies detected out of {len(scored)} rows")

    console.rule("[bold blue]5. Persist")
    writer = StorageWriter(cfg.storage)
    writer.write_telemetry(telem_df)
    writer.write_labels(labels_df)
    writer.write_scored(scored)
    console.print(f"  Output written to {output_dir}/")

    console.rule("[bold blue]Summary")
    _print_summary(scored)


def _print_summary(scored: "pd.DataFrame") -> None:
    """Print a rich table summarizing inference results."""
    import pandas as pd

    table = Table(title="Inference Summary by Channel")
    table.add_column("Channel")
    table.add_column("Rows", justify="right")
    table.add_column("Anomalies", justify="right")
    table.add_column("Top Fault")
    table.add_column("Max Score", justify="right")

    for ch_id, grp in scored.groupby("channel_id"):
        n_anom = int(grp.get("is_anomaly", pd.Series(dtype=bool)).sum())
        max_score = float(grp.get("anomaly_score", pd.Series([0.0])).max())
        faults = grp.get("predicted_fault", pd.Series(dtype=str))
        top_fault = faults[faults != "none"].mode().iloc[0] if (faults != "none").any() else "none"
        table.add_row(str(ch_id), str(len(grp)), str(n_anom), top_fault, f"{max_score:.3f}")

    console.print(table)


if __name__ == "__main__":
    app()
