import numpy as np
import plotly.graph_objects as go
import streamlit as st

from scipy.ndimage import gaussian_filter1d

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


def plot_lead_importance(
    lead_importance,
):
    lead_importance = np.asarray(
        lead_importance,
        dtype=float,
    )

    fig = go.Figure(
        go.Bar(
            x=LEAD_NAMES,
            y=lead_importance,
            hovertemplate=(
                "<b>%{x}</b><br>" "Model attribution: %{y:.3f}" "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="ECG lead influence",
        xaxis_title="ECG lead",
        yaxis_title="Relative model attribution",
        yaxis=dict(
            rangemode="tozero",
        ),
        height=450,
        margin=dict(
            l=50,
            r=20,
            t=60,
            b=50,
        ),
    )
    return fig


def plot_temporal_importance(
    temporal_importance,
):
    temporal_importance = np.asarray(
        temporal_importance,
        dtype=float,
    ).squeeze()
    samples = np.arange(len(temporal_importance))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=samples,
            y=temporal_importance,
            mode="lines",
            fill="tozeroy",
            hovertemplate=(
                "Sample: %{x}<br>" "Model attribution: %{y:.4f}" "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="ECG regions influencing the prediction",
        xaxis_title="ECG sample",
        yaxis_title="Relative model attribution",
        height=450,
        margin=dict(
            l=50,
            r=20,
            t=60,
            b=50,
        ),
    )
    return fig


def plot_grad_cam(
    waveform,
    grad_cam,
    lead_name="I",
):
    waveform = np.asarray(
        waveform,
        dtype=float,
    )
    grad_cam = np.asarray(
        grad_cam,
        dtype=float,
    ).squeeze()

    if grad_cam.ndim > 1:
        grad_cam = np.mean(
            np.abs(grad_cam),
            axis=0,
        )

    if len(grad_cam) != waveform.shape[0]:
        old_x = np.linspace(
            0,
            1,
            len(grad_cam),
        )
        new_x = np.linspace(
            0,
            1,
            waveform.shape[0],
        )
        grad_cam = np.interp(
            new_x,
            old_x,
            grad_cam,
        )

    grad_cam = np.maximum(
        grad_cam,
        0,
    )
    grad_cam = gaussian_filter1d(
        grad_cam,
        sigma=8,
    )

    max_value = np.max(grad_cam)
    if max_value > 0:
        grad_cam = grad_cam / max_value

    lead_index = LEAD_NAMES.index(lead_name)
    signal = waveform[
        :,
        lead_index,
    ]
    x = np.arange(waveform.shape[0])

    fig = go.Figure()
    colour_scale = [
        [0.0, "#fff5cc"],
        [0.5, "#ff9500"],
        [1.0, "#d00000"],
    ]
    segment_size = 5

    for start in range(
        0,
        len(x) - 1,
        segment_size,
    ):
        end = min(
            start + segment_size + 1,
            len(x),
        )
        value = float(np.mean(grad_cam[start:end]))
        if value < 0.5:
            t = value / 0.5
            colour = (
                f"rgb(" f"255," f"{int(245 - 100 * t)}," f"{int(204 - 204 * t)}" f")"
            )
        else:
            t = (value - 0.5) / 0.5
            colour = f"rgb(" f"255," f"{int(149 - 149 * t)}," f"0" f")"

        fig.add_trace(
            go.Scatter(
                x=x[start:end],
                y=signal[start:end],
                mode="lines",
                line=dict(
                    width=2,
                    color=colour,
                ),
                showlegend=False,
                hovertemplate=(
                    f"<b>{lead_name}</b><br>"
                    "Sample: %{x}<br>"
                    "Amplitude: %{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[None, None],
            y=[None, None],
            mode="markers",
            marker=dict(
                size=0,
                color=[0, 1],
                colorscale=colour_scale,
                cmin=0,
                cmax=1,
                showscale=True,
                colorbar=dict(
                    title="Model attribution",
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=(f"Model focus along Lead {lead_name}"),
        xaxis_title="ECG sample",
        yaxis_title="Amplitude",
        height=500,
        margin=dict(
            l=50,
            r=80,
            t=60,
            b=50,
        ),
    )
    return fig


def plot_tabular_attribution(
    attribution,
    feature_names,
):
    attribution = np.asarray(
        attribution,
        dtype=float,
    ).squeeze()
    attribution = attribution.flatten()

    importance = np.abs(attribution)
    order = np.argsort(importance)[::-1]
    ordered_features = [feature_names[i] for i in order]
    ordered_values = [attribution[i] for i in order]

    fig = go.Figure(
        go.Bar(
            x=ordered_values,
            y=ordered_features,
            orientation="h",
            customdata=np.abs(np.asarray(ordered_values)),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Model attribution: %{x:.4f}<br>"
                "Absolute attribution: %{customdata:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Clinical features influencing the prediction",
        xaxis_title="Model attribution",
        yaxis_title="Clinical feature",
        height=450,
        margin=dict(
            l=160,
            r=20,
            t=60,
            b=50,
        ),
    )

    return fig


def plot_attention(
    attention,
):
    attention = np.asarray(
        attention,
        dtype=float,
    ).squeeze()

    if attention.ndim > 1:
        attention = np.mean(
            attention,
            axis=0,
        )

    samples = np.arange(len(attention))

    fig = go.Figure(
        go.Scatter(
            x=samples,
            y=attention,
            mode="lines",
            fill="tozeroy",
            hovertemplate=(
                "Sequence position: %{x}<br>"
                "Attention weight: %{y:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Model attention across the ECG sequence",
        xaxis_title="ECG sequence position",
        yaxis_title="Attention weight",
        height=400,
        margin=dict(
            l=50,
            r=20,
            t=60,
            b=50,
        ),
    )

    return fig


def display_explainability(
    explanation,
    waveform,
    feature_names,
):
    with st.expander(
        "Which ECG leads influenced the prediction?",
        expanded=True,
    ):
        st.caption(
            "This plot shows which ECG leads contributed "
            "most strongly to the model's prediction. "
            "A higher value means that the model relied "
            "more heavily on information from that lead "
            "for this particular ECG."
        )

        fig = plot_lead_importance(explanation["lead_importance"])

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.info(
            "The values describe how strongly the model "
            "used information from each lead."
        )

    with st.expander(
        "Which parts of the ECG influenced the prediction?",
        expanded=True,
    ):
        st.caption(
            "This plot highlights the parts of the ECG "
            "signal that contributed most strongly to the "
            "model's prediction. Peaks indicate regions "
            "where the model relied more heavily on the "
            "ECG signal."
        )

        fig = plot_temporal_importance(explanation["temporal_importance"])
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.info(
            "The highlighted regions show where the model "
            "found information useful for its prediction."
        )

    grad_cam_result = explanation.get("grad_cam")
    if grad_cam_result is not None:
        with st.expander(
            "Where in the ECG did the model focus?",
            expanded=True,
        ):
            st.caption(
                "The ECG waveform is coloured according "
                "to the strength of the model's focus. "
                "Redder regions indicate stronger model "
                "attribution, while lighter regions indicate "
                "weaker attribution."
            )
            selected_grad_cam_lead = st.selectbox(
                "ECG lead",
                options=LEAD_NAMES,
                index=0,
                key="grad_cam_lead",
            )

            fig = plot_grad_cam(
                waveform=waveform,
                grad_cam=grad_cam_result["grad_cam"],
                lead_name=selected_grad_cam_lead,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.info(
                "This visualisation shows which time regions "
                "of the ECG contributed most strongly to the "
                "CNN representation used for the prediction."
            )

            st.caption(
                "CNN layer used for this explanation: "
                f"{grad_cam_result['target_layer']}"
            )

    tabular_attribution = explanation.get("tabular_attribution")

    if tabular_attribution is not None:
        with st.expander(
            "Which clinical features influenced the prediction?",
            expanded=True,
        ):
            st.caption(
                "This plot shows how the clinical information "
                "provided to the model contributed to its "
                "prediction for this ECG."
            )
            fig = plot_tabular_attribution(
                attribution=tabular_attribution,
                feature_names=feature_names,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
            )
            st.info(
                "Features with larger absolute values had a "
                "stronger influence on this particular prediction. "
                "The direction of the attribution indicates whether "
                "the feature contributed towards or away from the "
                "model's output."
            )

    attention = explanation.get("attention")

    if attention is not None:
        with st.expander(
            "What parts of the ECG sequence received more attention?",
            expanded=False,
        ):
            st.caption(
                "This plot shows how the model distributed "
                "attention across the ECG sequence while "
                "forming its prediction."
            )
            fig = plot_attention(attention)
            st.plotly_chart(
                fig,
                use_container_width=True,
            )
            st.info(
                "Higher attention means that the model gave "
                "greater weight to that part of the ECG sequence "
                "during processing."
            )
