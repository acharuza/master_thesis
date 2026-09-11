import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.config import TABULAR_FEATURES

LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

ECG_COLOUR = "#C94C4C"


def display_ecg(
    sample,
    metadata_row,
    ecg_index,
):
    waveform = sample["waveform"].numpy()

    st.header("Selected ECG")
    st.caption(f"ECG recording {ecg_index}")
    st.subheader("ECG waveform")

    fig = go.Figure()
    spacing = 8.0

    for i, lead_name in enumerate(LEAD_NAMES):
        signal = waveform[:, i]
        offset = (len(LEAD_NAMES) - 1 - i) * spacing
        signal_offset = signal + offset
        fig.add_trace(
            go.Scatter(
                x=list(range(waveform.shape[0])),
                y=signal_offset,
                mode="lines",
                line=dict(
                    width=1.2,
                    color=ECG_COLOUR,
                ),
                name=lead_name,
                customdata=signal,
                hovertemplate=(
                    f"<b>{lead_name}</b><br>"
                    "Sample: %{x}<br>"
                    "Amplitude: %{customdata:.3f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    tick_positions = [
        (len(LEAD_NAMES) - 1 - i) * spacing for i in range(len(LEAD_NAMES))
    ]

    fig.update_layout(
        height=1200,
        margin=dict(
            l=60,
            r=20,
            t=20,
            b=40,
        ),
        hovermode="x unified",
        colorway=[ECG_COLOUR],
        xaxis=dict(
            title="Sample",
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=tick_positions,
            ticktext=LEAD_NAMES,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks="",
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        theme="streamlit",
    )
    st.subheader("Clinical Features")

    feature_columns = st.columns(4)

    for i, feature_name in enumerate(TABULAR_FEATURES):
        value = metadata_row[feature_name]
        if feature_name == "sex":
            display_value = str(value).capitalize()
        elif pd.isna(value):
            display_value = "N/A"
        else:
            display_value = f"{float(value):.1f}"
        feature_columns[i % 4].metric(
            feature_name.replace(
                "-",
                " ",
            ).title(),
            display_value,
        )

    st.subheader("Additional Information")
    info_columns = st.columns(3)
    additional_info = {
        "Acquisition Year": metadata_row["acquisition_year"],
        "Location Setting": metadata_row["location_setting"],
        "Race / Ethnicity": metadata_row["race_ethnicity"],
    }

    for i, (
        label,
        value,
    ) in enumerate(additional_info.items()):
        if pd.isna(value):
            display_value = "N/A"
        else:
            display_value = (
                str(value)
                .replace(
                    "_",
                    " ",
                )
                .title()
            )
        info_columns[i].metric(
            label,
            display_value,
        )
