from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
REQUIRED_FILES = [
    BASE_DIR / "data" / "raw" / "fraudTrain.csv",
    BASE_DIR / "data" / "raw" / "fraudTest.csv",
    BASE_DIR / "data" / "processed" / "data.pkl",
    BASE_DIR / "data" / "processed" / "scaler.pkl",
    BASE_DIR / "models" / "random_forest.pkl",
    BASE_DIR / "models" / "stacking_model.pkl",
]


@dataclass
class DemoBundle:
    defaults: dict[str, Any]
    dropdown_options: dict[str, list[str]]


class DemoExplainer:
    def __init__(self) -> None:
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        self.bundle = DemoBundle(
            defaults={
                "trans_date_trans_time": now,
                "dob": pd.Timestamp("1994-01-01"),
                "amt": 120.0,
                "merchant": "fraud_Kilback LLC",
                "category": "shopping_net",
                "gender": "F",
                "job": "Sales",
                "city": "New York",
                "state": "NY",
                "zip": 10001,
                "lat": 40.7128,
                "long": -74.0060,
                "city_pop": 8804190,
                "merch_lat": 40.7306,
                "merch_long": -73.9352,
            },
            dropdown_options={
                "category": [
                    "gas_transport",
                    "grocery_pos",
                    "grocery_net",
                    "shopping_pos",
                    "shopping_net",
                    "misc_net",
                    "food_dining",
                    "entertainment",
                    "travel",
                    "health_fitness",
                ],
                "merchant": [
                    "fraud_Kilback LLC",
                    "Amazon",
                    "Walmart",
                    "Target",
                    "Shell",
                    "Uber",
                    "Best Buy",
                    "Netflix",
                    "Delta",
                    "Apple Store",
                ],
            },
        )

    def predict_transaction(self, payload: dict[str, Any]) -> dict[str, Any]:
        amount = float(payload["amt"])
        category = str(payload["category"]).lower()
        merchant = str(payload["merchant"]).lower()
        hour = pd.to_datetime(payload["trans_date_trans_time"]).hour

        score = 0.08
        reasons: list[str] = []

        if amount > 5000:
            score += 0.45
            reasons.append("Transaction amount is much higher than typical card activity")
        elif amount > 1000:
            score += 0.25
            reasons.append("Transaction amount is higher than usual")
        elif amount > 300:
            score += 0.12
            reasons.append("Transaction amount is moderately elevated")

        if category in {"gas_transport", "shopping_net", "misc_net"}:
            score += 0.18
            reasons.append("Category has historically shown elevated risk")

        if "fraud" in merchant:
            score += 0.35
            reasons.append("Merchant name appears suspicious")

        if 0 <= hour <= 4:
            score += 0.12
            reasons.append("Transaction timing is unusual")

        probability = min(max(score, 0.01), 0.99)

        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        if not reasons:
            reasons.append("Transaction pattern appears closer to normal historical behavior")

        prediction = 1 if probability >= 0.5 else 0
        confidence = probability if prediction == 1 else 1 - probability

        return {
            "prediction": prediction,
            "prediction_label": "fraudulent" if prediction == 1 else "legitimate",
            "risk_level": risk_level,
            "risk_title": f"{risk_level} Risk Transaction",
            "confidence": confidence,
            "fraud_probability": probability,
            "actual_label": None,
            "actual_label_text": None,
            "error_type": None,
            "top_features": [],
            "top_feature_details": [],
            "explanation": f"This transaction is considered {risk_level.lower()} risk based on simplified fallback rules.",
            "explanation_points": reasons[:4],
            "raw_features": payload,
        }


st.set_page_config(
    page_title="Explainable Credit Card Fraud Detection",
    layout="wide",
)


@st.cache_resource(show_spinner=True)
def load_explainer() -> Any:
    from src.explain import FraudExplainer

    return FraudExplainer()


def _missing_required_files() -> list[Path]:
    return [path for path in REQUIRED_FILES if not path.exists()]


def _resolve_explainer() -> tuple[Any, list[Path], str | None]:
    missing_files = _missing_required_files()
    if missing_files:
        return DemoExplainer(), missing_files, None

    try:
        return load_explainer(), [], None
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        return DemoExplainer(), missing_files, str(exc)


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

    explainer, missing_files, error_text = _resolve_explainer()

    if missing_files or error_text:
        st.info("Running in fallback demo mode because local model/data artifacts are not available in this deployment.")

    with st.form("fraud_form"):
        st.markdown("### Check Transaction Risk")
        payload = _build_payload(explainer)
        submitted = st.form_submit_button("Check Transaction Risk")

    if submitted:
        result = explainer.predict_transaction(payload)
        _render_prediction(result)


if __name__ == "__main__":
    main()
