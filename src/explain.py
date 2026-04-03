from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

from src.inference_utils import (
    PreprocessingBundle,
    load_preprocessing_bundle,
    load_random_forest_model,
    load_stacking_model,
)


CLASS_LABELS = {
    0: "legitimate",
    1: "fraudulent",
}

FEATURE_DESCRIPTIONS = {
    "amt": "the transaction amount",
    "distance": "the customer-to-merchant distance",
    "hour": "the transaction time",
    "day": "the day of the month",
    "month": "the month of the year",
    "age": "the customer's age pattern",
    "category": "the transaction category",
    "merchant": "the merchant pattern",
    "city": "the customer's city",
    "state": "the customer's state",
    "job": "the job profile",
    "city_pop": "the city population",
    "lat": "the customer's latitude",
    "long": "the customer's longitude",
    "merch_lat": "the merchant latitude",
    "merch_long": "the merchant longitude",
    "zip": "the ZIP code",
    "gender": "the customer gender profile",
}


@dataclass
class FeatureContribution:
    feature: str
    raw_value: Any
    display_value: str
    shap_value: float
    direction: str
    summary: str


class FraudExplainer:
    def __init__(self, background_size: int = 200, top_k: int = 4) -> None:
        self.bundle: PreprocessingBundle = load_preprocessing_bundle()
        self.rf_model = load_random_forest_model()
        self.stacking_model = load_stacking_model()
        self.top_k = top_k

        background = self.bundle.X_train_scaled[: min(background_size, len(self.bundle.X_train_scaled))]
        self.explainer = shap.TreeExplainer(
            self.rf_model,
            data=background,
            feature_perturbation="interventional",
        )

    def predict_transaction(
        self,
        payload: dict[str, Any],
        actual_label: int | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        raw_frame, scaled_frame = self.bundle.transform_input(payload)
        prediction = int(self.stacking_model.predict(scaled_frame)[0])
        probability = float(self.stacking_model.predict_proba(scaled_frame)[0, 1])
        shap_values = self._fraud_shap_values(scaled_frame)[0]
        contributions = self._build_feature_contributions(raw_frame.iloc[0], shap_values, top_k or self.top_k)

        return self._format_result(
            raw_features=raw_frame.iloc[0],
            prediction=prediction,
            probability=probability,
            actual_label=actual_label,
            contributions=contributions,
        )

    def explain_test_samples(
        self,
        sample_mode: str = "balanced",
        sample_size: int = 10,
        top_k: int | None = None,
        random_state: int = 42,
    ) -> list[dict[str, Any]]:
        indices = self._sample_test_indices(sample_mode, sample_size, random_state)
        sampled_scaled = self.bundle.X_test_scaled[indices]
        sampled_raw = self.bundle.raw_test_features.iloc[indices].reset_index(drop=True)
        sampled_actual = self.bundle.y_test.iloc[indices].reset_index(drop=True)

        predictions = self.stacking_model.predict(sampled_scaled)
        probabilities = self.stacking_model.predict_proba(sampled_scaled)[:, 1]
        shap_matrix = self._fraud_shap_values(sampled_scaled)

        results: list[dict[str, Any]] = []

        for row_idx in range(len(indices)):
            contributions = self._build_feature_contributions(
                sampled_raw.iloc[row_idx],
                shap_matrix[row_idx],
                top_k or self.top_k,
            )
            results.append(
                self._format_result(
                    raw_features=sampled_raw.iloc[row_idx],
                    prediction=int(predictions[row_idx]),
                    probability=float(probabilities[row_idx]),
                    actual_label=int(sampled_actual.iloc[row_idx]),
                    contributions=contributions,
                )
            )

        return results

    def _sample_test_indices(self, sample_mode: str, sample_size: int, random_state: int) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        y_test = self.bundle.y_test.to_numpy()

        fraud_indices = np.flatnonzero(y_test == 1)
        normal_indices = np.flatnonzero(y_test == 0)

        if sample_mode == "fraud_only":
            size = min(sample_size, len(fraud_indices))
            return rng.choice(fraud_indices, size=size, replace=False)

        if sample_mode == "balanced":
            fraud_count = min(max(sample_size // 2, 1), len(fraud_indices))
            normal_count = min(sample_size - fraud_count, len(normal_indices))

            if normal_count == 0:
                return rng.choice(fraud_indices, size=fraud_count, replace=False)

            fraud_sample = rng.choice(fraud_indices, size=fraud_count, replace=False)
            normal_sample = rng.choice(normal_indices, size=normal_count, replace=False)
            combined = np.concatenate([fraud_sample, normal_sample])
            rng.shuffle(combined)
            return combined

        raise ValueError("sample_mode must be 'balanced' or 'fraud_only'.")

    def _fraud_shap_values(self, scaled_frame: np.ndarray) -> np.ndarray:
        shap_values = self.explainer.shap_values(scaled_frame, check_additivity=False)

        if isinstance(shap_values, list):
            return np.asarray(shap_values[1])

        shap_array = np.asarray(shap_values)

        if shap_array.ndim == 3:
            return shap_array[:, :, 1]

        return shap_array

    def _build_feature_contributions(
        self,
        raw_row: pd.Series,
        shap_values: np.ndarray,
        top_k: int,
    ) -> list[FeatureContribution]:
        ordered_indices = np.argsort(np.abs(shap_values))[::-1]
        contributions: list[FeatureContribution] = []
        seen: set[str] = set()

        for index in ordered_indices:
            feature = self.bundle.feature_columns[int(index)]
            if feature in seen:
                continue

            seen.add(feature)
            shap_value = float(shap_values[int(index)])
            raw_value = raw_row[feature]
            display_value = self._display_feature_value(feature, raw_value)
            direction = "raised risk" if shap_value >= 0 else "lowered risk"
            summary = self._humanize_feature(feature, raw_value, shap_value)

            contributions.append(
                FeatureContribution(
                    feature=feature,
                    raw_value=raw_value,
                    display_value=display_value,
                    shap_value=shap_value,
                    direction=direction,
                    summary=summary,
                )
            )

            if len(contributions) == top_k:
                break

        return contributions

    def _display_feature_value(self, feature: str, raw_value: Any) -> str:
        if feature in self.bundle.categorical_columns:
            return self.bundle.decode_value(feature, raw_value)

        if feature in {"amt", "distance"}:
            suffix = " km" if feature == "distance" else ""
            return f"{float(raw_value):.2f}{suffix}"

        if feature in {"lat", "long", "merch_lat", "merch_long"}:
            return f"{float(raw_value):.4f}"

        return str(raw_value)

    def _humanize_feature(self, feature: str, raw_value: Any, shap_value: float) -> str:
        if feature == "amt":
            return "Higher-than-usual transaction amount" if shap_value >= 0 else "Transaction amount appears closer to typical behavior"
        if feature == "distance":
            return "Transaction location pattern appears less typical" if shap_value >= 0 else "Location pattern appears closer to normal activity"
        if feature == "hour":
            return "Transaction timing appears unusual" if shap_value >= 0 else "Transaction timing appears consistent with normal activity"
        if feature == "merchant":
            return "Merchant history is associated with elevated risk" if shap_value >= 0 else "Merchant history appears lower risk"
        if feature == "category":
            return "Category historically associated with elevated risk" if shap_value >= 0 else "Category appears closer to lower-risk activity"
        if feature == "city":
            return "Location context is associated with elevated risk" if shap_value >= 0 else "Location context appears more typical"
        if feature == "job":
            return "Customer profile pattern appears less typical" if shap_value >= 0 else "Customer profile pattern appears more typical"
        if feature == "age":
            return "Age-related spending pattern appears less typical" if shap_value >= 0 else "Age-related spending pattern appears more typical"
        if feature in self.bundle.categorical_columns:
            description = FEATURE_DESCRIPTIONS.get(feature, feature).replace("the ", "").capitalize()
            return f"{description} is associated with elevated risk" if shap_value >= 0 else f"{description} appears closer to normal activity"

        description = FEATURE_DESCRIPTIONS.get(feature, feature).replace("the ", "").capitalize()
        return f"{description} pattern appears higher risk" if shap_value >= 0 else f"{description} pattern appears lower risk"

    def _format_result(
        self,
        raw_features: pd.Series,
        prediction: int,
        probability: float,
        actual_label: int | None,
        contributions: list[FeatureContribution],
    ) -> dict[str, Any]:
        top_feature_names = [item.feature for item in contributions]
        top_feature_details = [
            {
                "feature": item.feature,
                "value": item.display_value,
                "shap_value": round(item.shap_value, 6),
                "impact": item.direction,
                "summary": item.summary,
            }
            for item in contributions
        ]

        fraud_probability = self._clamp_probability(probability)
        confidence = fraud_probability if prediction == 1 else 1 - fraud_probability
        risk_level = self._risk_level(fraud_probability)
        explanation_points = [item.summary for item in contributions if item.shap_value >= 0]
        if not explanation_points:
            explanation_points = [item.summary for item in contributions]

        explanation = self._compose_explanation(risk_level, explanation_points)

        result = {
            "prediction": prediction,
            "prediction_label": CLASS_LABELS[prediction],
            "risk_level": risk_level,
            "risk_title": f"{risk_level} Risk Transaction",
            "confidence": confidence,
            "fraud_probability": fraud_probability,
            "actual_label": actual_label,
            "actual_label_text": None if actual_label is None else CLASS_LABELS[actual_label],
            "error_type": self._error_type(prediction, actual_label),
            "top_features": top_feature_names,
            "top_feature_details": top_feature_details,
            "explanation": explanation,
            "explanation_points": explanation_points,
            "raw_features": raw_features.to_dict(),
        }

        return result

    def _compose_explanation(
        self,
        risk_level: str,
        explanation_points: list[str],
    ) -> str:
        if not explanation_points:
            return "This transaction has been assessed using historical risk patterns."

        return f"This transaction is considered {risk_level.lower()} risk because of patterns seen in similar historical transactions."

    @staticmethod
    def _error_type(prediction: int, actual_label: int | None) -> str | None:
        if actual_label is None or prediction == actual_label:
            return None
        if prediction == 1 and actual_label == 0:
            return "False Positive"
        if prediction == 0 and actual_label == 1:
            return "False Negative"
        return None

    @staticmethod
    def _clamp_probability(probability: float) -> float:
        return min(max(float(probability), 0.01), 0.99)

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability < 0.3:
            return "Low"
        if probability < 0.7:
            return "Medium"
        return "High"


def _print_cli_examples(results: list[dict[str, Any]]) -> None:
    for index, result in enumerate(results, start=1):
        header = f"Transaction Example {index}: Prediction: {result['prediction']} | "
        header += f"Actual: {result['actual_label'] if result['actual_label'] is not None else 'N/A'}"
        print("\n" + "=" * 72)
        print(header)
        print(f"Label: {result['prediction_label'].title()} | Confidence: {result['confidence']:.2%}")

        if result["error_type"]:
            print(f"Error Type: {result['error_type']}")

        print("\nTop features:")
        for item in result["top_feature_details"]:
            print(f"- {item['feature']} ({item['value']}): {item['impact']}")

        print("\nExplanation:")
        print(result["explanation"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for fraud predictions.")
    parser.add_argument("--sample-mode", choices=["balanced", "fraud_only"], default="balanced")
    parser.add_argument("--sample-size", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    explainer = FraudExplainer(top_k=args.top_k)
    results = explainer.explain_test_samples(
        sample_mode=args.sample_mode,
        sample_size=args.sample_size,
        top_k=args.top_k,
    )
    _print_cli_examples(results)


if __name__ == "__main__":
    main()
