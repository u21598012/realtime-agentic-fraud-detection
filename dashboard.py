import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Fraud Detection Overview", layout="wide")

DATA_PATH = os.environ.get("FRAUD_DASHBOARD_CSV", "output.csv")

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return df


def fraud_ratio(df: pd.DataFrame) -> tuple[float, float]:
    if "actual" not in df or df.empty:
        return 0.0, 0.0
    fraud_count = float((df["actual"] == 1).sum())
    real_count = float((df["actual"] == 0).sum())
    total = fraud_count + real_count
    if total == 0:
        return 0.0, 0.0
    return fraud_count / total, real_count / total


def most_common_trace(df: pd.DataFrame) -> str | None:
    if "trace" not in df or df["trace"].dropna().empty:
        return None
    return df["trace"].dropna().mode().iloc[0]


def sankey_for_trace(trace: str) -> go.Figure:
    steps = [s for s in trace.split("|") if s]
    if len(steps) < 2:
        return go.Figure()
    nodes = list(dict.fromkeys(steps))
    node_index = {name: idx for idx, name in enumerate(nodes)}
    source = []
    target = []
    value = []
    for i in range(len(steps) - 1):
        source.append(node_index[steps[i]])
        target.append(node_index[steps[i + 1]])
        value.append(1)
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(label=nodes, pad=12, thickness=16, color="#23395d"),
            link=dict(source=source, target=target, value=value, color="#7aa5d2"),
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    return fig


def confusion_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty or "actual" not in df or "predicted" not in df:
        return go.Figure()
    cm = df.pivot_table(index="actual", columns="predicted", aggfunc="size", fill_value=0)
    cm = cm.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale=["#e8edf2", "#23395d"],
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
    return fig


def payment_type_card(df: pd.DataFrame):
    if "type" not in df or df.empty:
        st.metric(label="Most common payment type", value="N/A")
        return
    counts = df["type"].value_counts()
    top_type = counts.index[0]
    pct = (counts.iloc[0] / counts.sum()) * 100 if counts.sum() else 0
    st.metric(label="Most common payment type", value=top_type, delta=f"{pct:0.1f}% of volume")


def payment_type_bar(df: pd.DataFrame) -> go.Figure:
    if "type" not in df or df.empty:
        return go.Figure()
    counts = df["type"].value_counts().reset_index()
    counts.columns = ["type", "count"]
    fig = px.bar(counts, x="type", y="count", text="count", color_discrete_sequence=["#23395d"])
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=280)
    fig.update_traces(textposition="outside")
    return fig


def style():
    st.markdown(
        """
        <style>
        body { background-color: #f7f9fb; }
        .main { background-color: #f7f9fb; }
        .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
        .metric-label { color: #506784; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    style()
    df = load_data(DATA_PATH)

    st.title("Fraud Detection Overview")
    st.caption(f"Data source: {DATA_PATH}")

    if df.empty:
        st.warning("No data available. Ensure output.csv has been generated.")
        return

    fraud_share, real_share = fraud_ratio(df)

    top_trace = most_common_trace(df)
    sankey_fig = sankey_for_trace(top_trace) if top_trace else go.Figure()

    heatmap_fig = confusion_heatmap(df)

    col1, col2, col3 = st.columns([1.2, 1.2, 1.2])
    with col1:
        st.metric("Fraud share", f"{fraud_share*100:0.1f}%")
    with col2:
        st.metric("Real share", f"{real_share*100:0.1f}%")
    with col3:
        payment_type_card(df)

    st.markdown("---")

    upper = st.columns([1.3, 1])
    with upper[0]:
        st.subheader("Actual vs Predicted")
        st.plotly_chart(heatmap_fig, use_container_width=True)
    with upper[1]:
        st.subheader("Most Common Trace")
        if top_trace:
            st.caption(top_trace)
            st.plotly_chart(sankey_fig, use_container_width=True)
        else:
            st.info("No trace data available.")

    st.markdown("---")

    lower = st.columns([1, 1])
    with lower[0]:
        st.subheader("Payment Types")
        st.plotly_chart(payment_type_bar(df), use_container_width=True)
    with lower[1]:
        st.subheader("Recent Transactions")
        st.dataframe(df.tail(10), use_container_width=True)

if __name__ == "__main__":
    main()
