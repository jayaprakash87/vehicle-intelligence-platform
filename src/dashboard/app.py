"""VIP Dashboard — Streamlit-based channel health monitoring.

Launch:  streamlit run src/dashboard/app.py
        vip dashboard --data output/<run_id>/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure project root is on sys.path so imports work when streamlit runs this file directly
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config.catalog import build_channels, sedan_topology  # noqa: E402
from src.config.models import FeatureConfig, ModelConfig, NormalizerConfig, SimulationConfig  # noqa: E402
from src.features.engine import FeatureEngine  # noqa: E402
from src.inference.pipeline import InferencePipeline  # noqa: E402
from src.ingestion.normalizer import Normalizer  # noqa: E402
from src.models.anomaly import AnomalyDetector  # noqa: E402
from src.schemas.telemetry import (  # noqa: E402
    ChannelMeta,
    FaultInjection,
    FaultType,
    ProtectionEvent,
)
from src.simulation.generator import TelemetryGenerator  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VIP — Vehicle Intelligence Platform",
    page_icon="⚡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data source — disk or simulation
# ---------------------------------------------------------------------------

_DATA_DIR = os.environ.get("VIP_DATA_DIR", "")
_disk_mode = bool(_DATA_DIR)

st.sidebar.title("⚡ VIP Dashboard")
st.sidebar.markdown("---")

if _disk_mode:
    # ---------- Disk mode ----------
    st.sidebar.info(f"📂 Loading from:\n`{_DATA_DIR}`")
    run_btn = False
else:
    # ---------- Simulation mode ----------
    topology = st.sidebar.selectbox("Vehicle topology", ["sedan (52 ch)", "minimal (3 ch)"])
    duration_s = st.sidebar.slider("Duration (s)", 5, 120, 30, step=5)
    sample_ms = st.sidebar.selectbox("Sample interval (ms)", [50, 100, 200], index=1)
    seed = st.sidebar.number_input("Random seed", value=42, min_value=0, max_value=9999)

    st.sidebar.markdown("### Fault injection")
    inject_fault = st.sidebar.checkbox("Inject fault", value=True)
    if inject_fault:
        fault_type = st.sidebar.selectbox(
            "Fault type",
            [f.value for f in FaultType if f != FaultType.NONE],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        fault_channel_idx = st.sidebar.number_input("Channel index (0-based)", value=0, min_value=0)
        fault_start = st.sidebar.slider(
            "Fault start (s)", 1.0, float(duration_s - 2), float(duration_s // 4), step=0.5
        )
        fault_duration = st.sidebar.slider(
            "Fault duration (s)",
            1.0,
            float(duration_s - fault_start - 1),
            min(5.0, float(duration_s - fault_start - 1)),
            step=0.5,
        )
        fault_intensity = st.sidebar.slider("Intensity", 0.1, 1.0, 0.8, step=0.05)

    run_btn = st.sidebar.button("🚀 Run pipeline", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Cache the full pipeline run
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Running VIP pipeline…")
def run_pipeline(
    topo: str,
    dur: float,
    samp_ms: float,
    seed_val: int,
    inject: bool,
    f_type: str | None,
    f_ch_idx: int,
    f_start: float,
    f_dur: float,
    f_intensity: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate → normalise → features → train → infer.  Returns (raw, scored, labels)."""
    # Build channels
    if topo.startswith("sedan"):
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
    else:
        channels = [
            ChannelMeta(channel_id="ch_01", load_name="headlamp", nominal_current_a=6.0),
            ChannelMeta(channel_id="ch_02", load_name="defroster", nominal_current_a=12.0),
            ChannelMeta(channel_id="ch_03", load_name="seat_heater", nominal_current_a=8.0),
        ]

    faults: list[FaultInjection] = []
    if inject and f_type:
        ch_idx = min(f_ch_idx, len(channels) - 1)
        faults.append(
            FaultInjection(
                channel_id=channels[ch_idx].channel_id,
                fault_type=FaultType(f_type),
                start_s=f_start,
                duration_s=f_dur,
                intensity=f_intensity,
            )
        )

    sim_cfg = SimulationConfig(
        duration_s=dur,
        sample_interval_ms=samp_ms,
        channels=channels,
        fault_injections=faults,
        seed=seed_val,
    )

    # Generate
    gen = TelemetryGenerator(sim_cfg)
    raw_df, labels_df = gen.generate()

    # Normalise
    norm = Normalizer(NormalizerConfig())
    raw_df = norm.normalize(raw_df)

    # Features + train + infer
    feat_cfg = FeatureConfig(window_duration_s=2.0, min_duration_s=0.5)
    model_cfg = ModelConfig()
    engine = FeatureEngine(feat_cfg)
    feat_df = engine.compute(raw_df)

    detector = AnomalyDetector(model_cfg)
    detector.train(feat_df)

    pipeline = InferencePipeline(feat_cfg, model_cfg, detector)
    scored_df = pipeline.run_batch(raw_df)

    return raw_df, scored_df, labels_df


# ---------------------------------------------------------------------------
# Load data — from disk or simulation
# ---------------------------------------------------------------------------


def _load_from_disk(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load telemetry, scored, and labels from a pipeline output directory."""
    d = Path(data_dir)
    if not d.is_dir():
        st.error(f"Data directory not found: {data_dir}")
        st.stop()

    scored_path = d / "scored.parquet"
    if not scored_path.exists():
        # Try CSV fallback
        scored_path = d / "scored.csv"
    if not scored_path.exists():
        st.error(f"No scored.parquet or scored.csv in {data_dir}")
        st.stop()

    raw_path = d / "telemetry.parquet"
    if not raw_path.exists():
        raw_path = d / "telemetry.csv"

    labels_path = d / "labels.parquet"
    if not labels_path.exists():
        labels_path = d / "labels.csv"

    def _read(p: Path) -> pd.DataFrame:
        if not p.exists():
            return pd.DataFrame()
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p, parse_dates=["timestamp"])

    return _read(raw_path), _read(scored_path), _read(labels_path)


if _disk_mode:
    if "scored" not in st.session_state:
        raw, scored, labels = _load_from_disk(_DATA_DIR)
        st.session_state["raw"] = raw
        st.session_state["scored"] = scored
        st.session_state["labels"] = labels
else:
    if run_btn or "scored" not in st.session_state:
        f_type_val = fault_type if inject_fault else None
        f_start_val = fault_start if inject_fault else 1.0
        f_dur_val = fault_duration if inject_fault else 3.0
        f_int_val = fault_intensity if inject_fault else 0.8
        f_ch_val = fault_channel_idx if inject_fault else 0

        raw, scored, labels = run_pipeline(
            topology,
            duration_s,
            sample_ms,
            seed,
            inject_fault,
            f_type_val,
            f_ch_val,
            f_start_val,
            f_dur_val,
            f_int_val,
        )
        st.session_state["raw"] = raw
        st.session_state["scored"] = scored
        st.session_state["labels"] = labels

raw: pd.DataFrame = st.session_state["raw"]
scored: pd.DataFrame = st.session_state["scored"]
labels: pd.DataFrame = st.session_state["labels"]

# ---------------------------------------------------------------------------
# Header metrics
# ---------------------------------------------------------------------------

st.title("Vehicle Intelligence Platform")

n_channels = scored["channel_id"].nunique()
n_anomalies = int(scored["is_anomaly"].sum()) if "is_anomaly" in scored.columns else 0
n_faults = (
    int((scored["predicted_fault"] != FaultType.NONE.value).sum())
    if "predicted_fault" in scored.columns
    else 0
)
n_protection = 0
if "protection_event" in scored.columns:
    n_protection = int((scored["protection_event"] != ProtectionEvent.NONE.value).sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Channels", n_channels)
c2.metric("Total rows", f"{len(scored):,}")
c3.metric("Anomalies", n_anomalies)
c4.metric("Fault events", n_faults)
c5.metric("Protection trips", n_protection)

st.markdown("---")

# ---------------------------------------------------------------------------
# Channel selector
# ---------------------------------------------------------------------------

all_channels = sorted(scored["channel_id"].unique())
selected_channels = st.multiselect(
    "Select channels to inspect",
    all_channels,
    default=all_channels[:3] if len(all_channels) > 3 else all_channels,
)

if not selected_channels:
    st.warning("Select at least one channel.")
    st.stop()

view = scored[scored["channel_id"].isin(selected_channels)].copy()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab_signals, tab_anomaly, tab_faults, tab_protection, tab_summary = st.tabs(
    ["📈 Signals", "🔍 Anomalies", "⚠️ Faults", "🛡️ Protection Events", "📊 Summary"]
)

# ---- Signals tab ----
with tab_signals:
    st.subheader("Raw signals")

    fig_current = px.line(
        view,
        x="timestamp",
        y="current_a",
        color="channel_id",
        title="Current (A)",
        labels={"current_a": "Current (A)", "timestamp": "Time"},
    )
    fig_current.update_layout(height=350, margin=dict(t=40, b=30))
    st.plotly_chart(fig_current, use_container_width=True)

    col_v, col_t = st.columns(2)
    with col_v:
        fig_v = px.line(
            view,
            x="timestamp",
            y="voltage_v",
            color="channel_id",
            title="Voltage (V)",
        )
        fig_v.update_layout(height=300, margin=dict(t=40, b=30))
        st.plotly_chart(fig_v, use_container_width=True)
    with col_t:
        fig_t = px.line(
            view,
            x="timestamp",
            y="temperature_c",
            color="channel_id",
            title="Temperature (°C)",
        )
        fig_t.update_layout(height=300, margin=dict(t=40, b=30))
        st.plotly_chart(fig_t, use_container_width=True)

# ---- Anomalies tab ----
with tab_anomaly:
    st.subheader("Anomaly scores")

    if "anomaly_score" in view.columns:
        fig_anom = px.line(
            view,
            x="timestamp",
            y="anomaly_score",
            color="channel_id",
            title="Anomaly score over time",
        )
        fig_anom.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig_anom.update_layout(height=350, margin=dict(t=40, b=30))
        st.plotly_chart(fig_anom, use_container_width=True)

    if "spike_score" in view.columns:
        fig_spike = px.line(
            view,
            x="timestamp",
            y="spike_score",
            color="channel_id",
            title="Spike score",
        )
        fig_spike.update_layout(height=300, margin=dict(t=40, b=30))
        st.plotly_chart(fig_spike, use_container_width=True)

    if "is_anomaly" in view.columns:
        anom_rows = view[view["is_anomaly"]]
        st.write(f"**{len(anom_rows)} anomalous rows** in selected channels")
        if len(anom_rows) > 0:
            st.dataframe(
                anom_rows[
                    [
                        "timestamp",
                        "channel_id",
                        "anomaly_score",
                        "current_a",
                        "voltage_v",
                        "temperature_c",
                        "predicted_fault",
                    ]
                ].head(50),
                use_container_width=True,
            )

# ---- Faults tab ----
with tab_faults:
    st.subheader("Fault classification")

    if "predicted_fault" in view.columns:
        fault_view = view[view["predicted_fault"] != FaultType.NONE.value]

        if len(fault_view) > 0:
            # Fault distribution
            fault_counts = fault_view["predicted_fault"].value_counts().reset_index()
            fault_counts.columns = ["fault_type", "count"]
            fig_fdist = px.bar(
                fault_counts,
                x="fault_type",
                y="count",
                color="fault_type",
                title="Fault type distribution",
            )
            fig_fdist.update_layout(height=300, margin=dict(t=40, b=30), showlegend=False)
            st.plotly_chart(fig_fdist, use_container_width=True)

            # Fault timeline
            fig_ftime = px.scatter(
                fault_view,
                x="timestamp",
                y="channel_id",
                color="predicted_fault",
                size="fault_confidence",
                title="Fault timeline",
                labels={"predicted_fault": "Fault type"},
            )
            fig_ftime.update_layout(height=300, margin=dict(t=40, b=30))
            st.plotly_chart(fig_ftime, use_container_width=True)

            # Detail table
            display_cols = [
                "timestamp",
                "channel_id",
                "predicted_fault",
                "fault_confidence",
                "likely_causes",
            ]
            available = [c for c in display_cols if c in fault_view.columns]
            st.dataframe(fault_view[available].head(100), use_container_width=True)
        else:
            st.success("No faults detected in selected channels.")

# ---- Protection events tab ----
with tab_protection:
    st.subheader("Protection events")

    if "protection_event" in view.columns:
        pe_view = view[view["protection_event"] != ProtectionEvent.NONE.value]

        if len(pe_view) > 0:
            # Event type distribution
            pe_counts = pe_view["protection_event"].value_counts().reset_index()
            pe_counts.columns = ["event_type", "count"]
            fig_pe = px.pie(
                pe_counts,
                names="event_type",
                values="count",
                title="Protection event breakdown",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pe.update_layout(height=350, margin=dict(t=40, b=30))
            st.plotly_chart(fig_pe, use_container_width=True)

            # Timeline
            fig_pet = px.scatter(
                pe_view,
                x="timestamp",
                y="channel_id",
                color="protection_event",
                title="Protection event timeline",
                symbol="protection_event",
            )
            fig_pet.update_layout(height=300, margin=dict(t=40, b=30))
            st.plotly_chart(fig_pet, use_container_width=True)

            # Counts per channel
            pe_by_ch = (
                pe_view.groupby(["channel_id", "protection_event"]).size().reset_index(name="count")
            )
            fig_pech = px.bar(
                pe_by_ch,
                x="channel_id",
                y="count",
                color="protection_event",
                title="Protection events per channel",
                barmode="stack",
            )
            fig_pech.update_layout(height=300, margin=dict(t=40, b=30))
            st.plotly_chart(fig_pech, use_container_width=True)
        else:
            st.success("No protection events fired in selected channels.")
    else:
        st.info("Protection event data not available.")

# ---- Summary tab ----
with tab_summary:
    st.subheader("Per-channel summary")

    summary_rows = []
    for ch_id, grp in scored.groupby("channel_id"):
        n = len(grp)
        anom = int(grp["is_anomaly"].sum()) if "is_anomaly" in grp.columns else 0
        max_score = float(grp["anomaly_score"].max()) if "anomaly_score" in grp.columns else 0.0
        faults_col = grp.get("predicted_fault", pd.Series(dtype=str))
        fault_rows = faults_col[faults_col != FaultType.NONE.value]
        top_fault = fault_rows.mode().iloc[0] if len(fault_rows) > 0 else "none"
        pe_col = grp.get("protection_event", pd.Series(dtype=str))
        n_trips = int((pe_col != ProtectionEvent.NONE.value).sum()) if len(pe_col) > 0 else 0

        summary_rows.append(
            {
                "channel_id": ch_id,
                "rows": n,
                "anomalies": anom,
                "anomaly_rate": f"{anom / n * 100:.1f}%" if n > 0 else "0%",
                "max_score": f"{max_score:.3f}",
                "top_fault": top_fault,
                "protection_trips": n_trips,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, height=min(len(summary_df) * 35 + 40, 600))

    # Heatmap — anomaly rate per channel
    st.subheader("Channel health heatmap")
    if len(summary_df) > 0:
        heatmap_data = summary_df.copy()
        heatmap_data["anomaly_pct"] = (
            heatmap_data["anomalies"].astype(int) / heatmap_data["rows"].astype(int) * 100
        )
        heatmap_data = heatmap_data.sort_values("anomaly_pct", ascending=False).head(20)

        fig_heat = px.bar(
            heatmap_data,
            x="channel_id",
            y="anomaly_pct",
            color="anomaly_pct",
            color_continuous_scale="RdYlGn_r",
            title="Top 20 channels by anomaly rate (%)",
            labels={"anomaly_pct": "Anomaly %"},
        )
        fig_heat.update_layout(height=350, margin=dict(t=40, b=30))
        st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption("VIP v0.1 — Vehicle Intelligence Platform")
