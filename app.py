from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from src.explain import FraudExplainer


st.set_page_config(
    page_title="Explainable Credit Card Fraud Detection",
    layout="wide",
)


@st.cache_resource(show_spinner=True)
def load_explainer() -> FraudExplainer:
    return FraudExplainer()


def _safe_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


def _build_payload(explainer: FraudExplainer) -> dict[str, object]:
    defaults = explainer.bundle.defaults
    options = explainer.bundle.dropdown_options

    left_col, right_col = st.columns(2)

    with left_col:
        amt = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=float(defaults["amt"]),
            step=1.0,
            help="Enter the amount charged in this transaction.",
        )
        category = st.selectbox(
            "Category",
            options["category"],
            index=_safe_index(options["category"], defaults["category"]),
            help="Select the transaction category.",
        )
        merchant = st.selectbox(
            "Merchant",
            options["merchant"],
            index=_safe_index(options["merchant"], defaults["merchant"]),
            help="Choose the merchant involved in the transaction.",
        )

    with right_col:
        transaction_time = st.time_input(
            "Time (Optional)",
            value=defaults["trans_date_trans_time"].time(),
            step=3600,
            help="Defaults to the current time if left unchanged.",
        )

    timestamp = datetime.combine(datetime.now().date(), transaction_time)

    payload = {
        "trans_date_trans_time": pd.Timestamp(timestamp),
        "dob": pd.Timestamp(datetime.now()) - pd.Timedelta(days=30 * 365),
        "amt": amt,
        "category": category,
        "merchant": merchant,
        "gender": defaults["gender"],
        "job": defaults["job"],
        "city": defaults["city"],
        "state": defaults["state"],
        "city_pop": defaults["city_pop"],
        "zip": defaults["zip"],
        "lat": defaults["lat"],
        "long": defaults["long"],
        "merch_lat": defaults["merch_lat"],
        "merch_long": defaults["merch_long"],
    }

    return payload


def _risk_style(level: str) -> tuple[str, str]:
    if level == "High":
        return "#fde7e9", "#b42318"
    if level == "Medium":
        return "#fff4d6", "#b54708"
    return "#e8f7ed", "#067647"


def _render_prediction(result: dict[str, object]) -> None:
    risk_level = str(result["risk_level"])
    risk_title = str(result["risk_title"])
    confidence = float(result["confidence"])
    fraud_probability = float(result["fraud_probability"])
    background, accent = _risk_style(risk_level)

    st.markdown(
        f"""
        <div style="background:{background}; border-left:8px solid {accent}; padding:18px 20px; border-radius:12px; margin: 8px 0 18px 0;">
            <div style="color:{accent}; font-size:28px; font-weight:700;">{risk_title}</div>
            <div style="color:#344054; margin-top:6px;">Confidence: {confidence:.0%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Risk Score", f"{fraud_probability:.0%}")
    metric_col2.metric("Confidence", f"{confidence:.0%}")

    st.subheader("Why this risk level was assigned")
    for point in result["explanation_points"][:4]:
        st.markdown(f"- {point}")

    st.caption("This system provides a risk assessment based on historical patterns and should not be considered a definitive fraud detection system.")


def main() -> None:
    st.title("Transaction Risk Analysis System")
    st.caption("Use a few basic transaction details to estimate transaction risk from historical patterns.")

    explainer = load_explainer()

    with st.form("fraud_form"):
        st.markdown("### Check Transaction Risk")
        payload = _build_payload(explainer)
        submitted = st.form_submit_button("Check Transaction Risk")

    if submitted:
        result = explainer.predict_transaction(payload)
        _render_prediction(result)


if __name__ == "__main__":
    main()
