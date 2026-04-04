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
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from src.utils.logging import configure_logging

if TYPE_CHECKING:
    import pandas as pd

app = typer.Typer(name="vip", help="Vehicle Intelligence Platform CLI")
console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup(
    config: str,
    output_dir: str | None = None,
    fmt: str | None = None,
    json_log: bool = False,
):
    """Common setup: configure logging, load config, stamp output dir with run ID."""
    from src.config.models import load_config, default_config

    run_id = configure_logging(json_format=json_log)

    cfg_path = Path(config)
    if not cfg_path.exists():
        if config != "configs/default.yaml":
            _abort(f"Config file not found: {config}")
        cfg = default_config()
    else:
        try:
            cfg = load_config(config)
        except Exception as exc:
            _abort(f"Failed to parse config '{config}': {exc}")

    # Stamp output directory with run ID so runs never overwrite each other
    if output_dir:
        cfg.storage.output_dir = str(Path(output_dir) / run_id)
    else:
        cfg.storage.output_dir = str(Path(cfg.storage.output_dir) / run_id)
    if fmt:
        cfg.storage.format = fmt

    Path(cfg.storage.output_dir).mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]run_id={run_id}  output={cfg.storage.output_dir}[/dim]")
    return cfg


def _load_data(data_path: str) -> pd.DataFrame:
    """Load telemetry from parquet or CSV with a clear error on missing file."""
    import pandas as pd

    p = Path(data_path)
    if not p.exists():
        _abort(f"Data file not found: {data_path}")
    try:
        if data_path.endswith(".parquet"):
            return pd.read_parquet(data_path)
        return pd.read_csv(data_path, parse_dates=["timestamp"])
    except Exception as exc:
        _abort(f"Failed to read '{data_path}': {exc}")


def _abort(msg: str) -> None:
    """Print a rich error and exit with code 1."""
    console.print(f"[bold red]Error:[/bold red] {msg}")
    raise typer.Exit(code=1)


@app.command()
def simulate(
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="Config file path"),
    output_dir: str = typer.Option("output", "--output", "-o", help="Output directory"),
    fmt: str = typer.Option("parquet", "--format", "-f", help="Output format: parquet|csv|json"),
    json_log: bool = typer.Option(False, "--json-log", help="Emit structured JSON logs"),
) -> None:
    """Generate synthetic eFuse telemetry data."""
    from src.simulation.generator import TelemetryGenerator
    from src.ingestion.normalizer import Normalizer
    from src.storage.writer import StorageWriter

    cfg = _setup(config, output_dir, fmt, json_log)

    gen = TelemetryGenerator(cfg.simulation)
    telem_df, labels_df = gen.generate()

    norm = Normalizer(cfg.normalizer)
    telem_df = norm.normalize(telem_df)

    writer = StorageWriter(cfg.storage)
    writer.write_telemetry(telem_df)
    writer.write_labels(labels_df)

    console.print(
        f"[green]Generated {len(telem_df)} telemetry rows, {len(labels_df)} labels[/green]"
    )


@app.command()
def replay(
    data_path: str = typer.Argument(
        ..., help="Path to measurement file (.mf4, .mdf, .csv, .parquet)"
    ),
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    output_dir: str = typer.Option("output", "--output", "-o"),
    fmt: str = typer.Option("parquet", "--format", "-f"),
    channel_id: str = typer.Option(
        "ch_01", "--channel", help="Default channel ID when file has no channel column"
    ),
    map_current: str = typer.Option("", "--map-current", help="Source signal name for current_a"),
    map_voltage: str = typer.Option("", "--map-voltage", help="Source signal name for voltage_v"),
    map_temperature: str = typer.Option(
        "", "--map-temperature", help="Source signal name for temperature_c"
    ),
    map_timestamp: str = typer.Option(
        "", "--map-timestamp", help="Source signal name for timestamp"
    ),
    json_log: bool = typer.Option(False, "--json-log", help="Emit structured JSON logs"),
) -> None:
    """Replay real measurement data (MDF4/CSV/Parquet) through the VIP pipeline."""
    from src.features.engine import FeatureEngine
    from src.inference.pipeline import InferencePipeline
    from src.ingestion.normalizer import Normalizer
    from src.ingestion.reader import ColumnMapping, MeasurementReader
    from src.models.anomaly import AnomalyDetector
    from src.storage.writer import StorageWriter

    cfg = _setup(config, output_dir, fmt, json_log)

    # Build column mapping from CLI overrides
    mapping_kwargs: dict[str, str] = {}
    if map_current:
        mapping_kwargs["current_a"] = map_current
    if map_voltage:
        mapping_kwargs["voltage_v"] = map_voltage
    if map_temperature:
        mapping_kwargs["temperature_c"] = map_temperature
    if map_timestamp:
        mapping_kwargs["timestamp"] = map_timestamp
    mapping = ColumnMapping(**mapping_kwargs)

    # Read measurement file
    reader = MeasurementReader(mapping=mapping, default_channel_id=channel_id)
    try:
        telem_df = reader.read(data_path)
    except Exception as exc:
        _abort(f"Failed to read measurement file: {exc}")
    console.print(f"  Loaded {len(telem_df)} rows from [bold]{data_path}[/bold]")

    # Normalize
    norm = Normalizer(cfg.normalizer)
    telem_df = norm.normalize(telem_df)

    # Features + train + infer
    engine = FeatureEngine(cfg.features)
    feat_df = engine.compute(telem_df)

    detector = AnomalyDetector(cfg.model)
    model_file = Path(cfg.model.model_dir) / "anomaly_detector.joblib"
    if model_file.exists():
        detector.load()
        console.print("[dim]Loaded pre-trained model[/dim]")
    else:
        console.print("[dim]No pre-trained model — training on this data[/dim]")
        detector.train(feat_df)
        detector.save()

    pipeline = InferencePipeline(cfg.features, cfg.model, detector)
    scored = pipeline.run_batch(telem_df)

    # Persist
    writer = StorageWriter(cfg.storage)
    writer.write_telemetry(telem_df)
    writer.write_scored(scored)
    console.print(f"  Output written to {cfg.storage.output_dir}/")

    _print_summary(scored)


@app.command()
def train(
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    data_path: str = typer.Option("output/telemetry.parquet", "--data", "-d"),
    json_log: bool = typer.Option(False, "--json-log", help="Emit structured JSON logs"),
) -> None:
    """Train the anomaly detection model on generated (or provided) telemetry."""
    from src.features.engine import FeatureEngine
    from src.models.anomaly import AnomalyDetector

    cfg = _setup(config, json_log=json_log)
    df = _load_data(data_path)

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
    json_log: bool = typer.Option(False, "--json-log", help="Emit structured JSON logs"),
) -> None:
    """Run inference on telemetry data (batch mode)."""
    from src.inference.pipeline import InferencePipeline
    from src.models.anomaly import AnomalyDetector
    from src.storage.writer import StorageWriter

    cfg = _setup(config, output_dir, fmt, json_log)
    df = _load_data(data_path)

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
    mqtt: bool = typer.Option(False, "--mqtt", help="Enable MQTT alert publishing"),
    mqtt_broker: str = typer.Option("localhost", "--mqtt-broker", help="MQTT broker host"),
    mqtt_port: int = typer.Option(1883, "--mqtt-port", help="MQTT broker port"),
    mqtt_topic: str = typer.Option("vip/alerts", "--mqtt-topic", help="MQTT topic prefix"),
    json_log: bool = typer.Option(False, "--json-log", help="Emit structured JSON logs"),
) -> None:
    """Run the edge runtime loop over telemetry data."""
    from src.edge.runtime import EdgeRuntime
    from src.inference.pipeline import InferencePipeline
    from src.models.anomaly import AnomalyDetector
    from src.storage.writer import StorageWriter
    from src.transport.alert_sinks import AlertSinkBase, MqttAlertSink
    from src.transport.mock_can import DataFrameTransport

    cfg = _setup(config, output_dir, json_log=json_log)
    df = _load_data(data_path)

    detector = AnomalyDetector(cfg.model)
    model_file = Path(cfg.model.model_dir) / "anomaly_detector.joblib"
    if model_file.exists():
        detector.load()

    # Build alert sinks
    sinks: list[AlertSinkBase] = []
    mqtt_enabled = mqtt or cfg.mqtt.enabled
    if mqtt_enabled:
        host = mqtt_broker if mqtt else cfg.mqtt.broker_host
        port = mqtt_port if mqtt else cfg.mqtt.broker_port
        topic = mqtt_topic if mqtt else cfg.mqtt.topic_prefix
        sink = MqttAlertSink(
            broker_host=host,
            broker_port=port,
            topic_prefix=topic,
            client_id=cfg.mqtt.client_id,
            username=cfg.mqtt.username,
            password=cfg.mqtt.password,
            qos=cfg.mqtt.qos,
            tls=cfg.mqtt.tls,
        )
        sinks.append(sink)
        console.print(f"[dim]MQTT sink → {host}:{port}/{topic}[/dim]")

    transport = DataFrameTransport(df)
    pipeline = InferencePipeline(cfg.features, cfg.model, detector)
    writer = StorageWriter(cfg.storage)
    runtime = EdgeRuntime(transport, pipeline, cfg.edge, writer, alert_sinks=sinks)

    mi = max_iter if max_iter > 0 else None
    alerts = runtime.run(max_iterations=mi)

    console.print(f"\n[bold]Edge run complete — {len(alerts)} alerts[/bold]")
    for a in alerts[:10]:
        console.print(
            f"  {a['timestamp']}  {a['channel_id']}  {a['fault']}  score={a['score']:.2f}"
        )
    if len(alerts) > 10:
        console.print(f"  ... and {len(alerts) - 10} more")


@app.command()
def pipeline(
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    output_dir: str = typer.Option("output", "--output", "-o"),
    fmt: str = typer.Option("parquet", "--format", "-f"),
    json_log: bool = typer.Option(False, "--json-log", help="Emit structured JSON logs"),
) -> None:
    """Full pipeline: simulate → train → infer — one command demo."""
    from src.simulation.generator import TelemetryGenerator
    from src.ingestion.normalizer import Normalizer
    from src.features.engine import FeatureEngine
    from src.models.anomaly import AnomalyDetector
    from src.inference.pipeline import InferencePipeline
    from src.storage.writer import StorageWriter

    cfg = _setup(config, output_dir, fmt, json_log)

    console.rule("[bold blue]1. Simulate")
    gen = TelemetryGenerator(cfg.simulation)
    telem_df, labels_df = gen.generate()
    norm = Normalizer(cfg.normalizer)
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
    console.print(f"  Output written to {cfg.storage.output_dir}/")

    console.rule("[bold blue]Summary")
    _print_summary(scored)


def _print_summary(scored: pd.DataFrame) -> None:
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


@app.command()
def dashboard() -> None:
    """Launch the Streamlit monitoring dashboard."""
    import subprocess
    import sys

    app_path = Path(__file__).parent / "dashboard" / "app.py"
    if not app_path.exists():
        _abort(f"Dashboard app not found at {app_path}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=False)


if __name__ == "__main__":
    app()
