from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pickle


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "fraudTrain.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "data.pkl"
SCALER_PATH = BASE_DIR / "data" / "processed" / "scaler.pkl"
STACKING_MODEL_PATH = BASE_DIR / "models" / "stacking_model.pkl"
RANDOM_FOREST_PATH = BASE_DIR / "models" / "random_forest.pkl"

DROP_COLUMNS = [
    "Unnamed: 0",
    "cc_num",
    "first",
    "last",
    "street",
    "trans_num",
    "unix_time",
]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * radius_km * atan2(sqrt(a), sqrt(1 - a))


@dataclass
class PreprocessingBundle:
    feature_columns: list[str]
    categorical_columns: list[str]
    categorical_classes: dict[str, np.ndarray]
    raw_train_features: pd.DataFrame
    raw_test_features: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    scaler: Any
    defaults: dict[str, Any]
    dropdown_options: dict[str, list[Any]]

    def encode_value(self, column: str, value: Any) -> int:
        classes = self.categorical_classes[column]
        value_str = str(value)

        if value_str not in classes:
            raise ValueError(
                f"Value '{value}' was not seen for '{column}' during training. "
                "Please choose a value from the dataset options."
            )

        return int(np.searchsorted(classes, value_str))

    def decode_value(self, column: str, value: Any) -> str:
        if column not in self.categorical_classes:
            return str(value)

        classes = self.categorical_classes[column]
        index = int(round(float(value)))
        index = max(0, min(index, len(classes) - 1))
        return str(classes[index])

    def transform_input(self, payload: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
        transaction_dt = pd.to_datetime(payload["trans_date_trans_time"])
        dob_dt = pd.to_datetime(payload["dob"])

        age = max(0, int((transaction_dt - dob_dt).days // 365))
        distance = haversine_distance(
            float(payload["lat"]),
            float(payload["long"]),
            float(payload["merch_lat"]),
            float(payload["merch_long"]),
        )

        row = {
            "merchant": self.encode_value("merchant", payload["merchant"]),
            "category": self.encode_value("category", payload["category"]),
            "amt": float(payload["amt"]),
            "gender": self.encode_value("gender", payload["gender"]),
            "city": self.encode_value("city", payload["city"]),
            "state": self.encode_value("state", payload["state"]),
            "zip": int(payload["zip"]),
            "lat": float(payload["lat"]),
            "long": float(payload["long"]),
            "city_pop": int(payload["city_pop"]),
            "job": self.encode_value("job", payload["job"]),
            "merch_lat": float(payload["merch_lat"]),
            "merch_long": float(payload["merch_long"]),
            "hour": int(transaction_dt.hour),
            "day": int(transaction_dt.day),
            "month": int(transaction_dt.month),
            "age": age,
            "distance": distance,
        }

        frame = pd.DataFrame([row], columns=self.feature_columns)
        scaled = self.scaler.transform(frame)
        return frame, scaled


def _fit_label_classes(df: pd.DataFrame) -> dict[str, np.ndarray]:
    classes: dict[str, np.ndarray] = {}

    for column in df.select_dtypes(include="object").columns:
        classes[column] = np.sort(df[column].astype(str).unique())

    return classes


def _encode_dataframe(df: pd.DataFrame, categorical_classes: dict[str, np.ndarray]) -> pd.DataFrame:
    encoded = df.copy()

    for column, classes in categorical_classes.items():
        encoded[column] = np.searchsorted(classes, encoded[column].astype(str))

    return encoded


def _build_defaults(df: pd.DataFrame, categorical_classes: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    defaults = {
        "trans_date_trans_time": datetime.now().replace(minute=0, second=0, microsecond=0),
        "dob": pd.Timestamp("1985-01-01"),
        "amt": float(df["amt"].median()),
        "merchant": str(df["merchant"].mode().iloc[0]),
        "category": str(df["category"].mode().iloc[0]),
        "gender": str(df["gender"].mode().iloc[0]),
        "city": str(df["city"].mode().iloc[0]),
        "state": str(df["state"].mode().iloc[0]),
        "zip": int(df["zip"].median()),
        "lat": float(df["lat"].median()),
        "long": float(df["long"].median()),
        "city_pop": int(df["city_pop"].median()),
        "job": str(df["job"].mode().iloc[0]),
        "merch_lat": float(df["merch_lat"].median()),
        "merch_long": float(df["merch_long"].median()),
    }

    dropdown_options = {
        column: classes.tolist()
        for column, classes in categorical_classes.items()
    }

    return defaults, dropdown_options


def load_preprocessing_bundle() -> PreprocessingBundle:
    raw_df = pd.read_csv(RAW_DATA_PATH)
    working_df = raw_df.drop(columns=DROP_COLUMNS, errors="ignore").copy()

    working_df["trans_date_trans_time"] = pd.to_datetime(working_df["trans_date_trans_time"])
    working_df["dob"] = pd.to_datetime(working_df["dob"])
    working_df["hour"] = working_df["trans_date_trans_time"].dt.hour
    working_df["day"] = working_df["trans_date_trans_time"].dt.day
    working_df["month"] = working_df["trans_date_trans_time"].dt.month
    working_df["age"] = (
        (working_df["trans_date_trans_time"] - working_df["dob"]).dt.days // 365
    ).clip(lower=0)

    working_df["distance"] = working_df.apply(
        lambda row: haversine_distance(
            row["lat"], row["long"], row["merch_lat"], row["merch_long"]
        ),
        axis=1,
    )

    defaults, dropdown_options = _build_defaults(working_df, _fit_label_classes(working_df))

    model_df = working_df.drop(columns=["trans_date_trans_time", "dob"])
    categorical_classes = _fit_label_classes(model_df)
    encoded_df = _encode_dataframe(model_df, categorical_classes)

    X = encoded_df.drop(columns=["is_fraud"])
    y = encoded_df["is_fraud"]

    from sklearn.model_selection import train_test_split

    raw_train_features, raw_test_features, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    with open(PROCESSED_DATA_PATH, "rb") as file:
        X_train_scaled, X_test_scaled, _, _ = pickle.load(file)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    return PreprocessingBundle(
        feature_columns=X.columns.tolist(),
        categorical_columns=list(categorical_classes.keys()),
        categorical_classes=categorical_classes,
        raw_train_features=raw_train_features.reset_index(drop=True),
        raw_test_features=raw_test_features.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        scaler=scaler,
        defaults=defaults,
        dropdown_options=dropdown_options,
    )


def load_stacking_model():
    return joblib.load(STACKING_MODEL_PATH)


def load_random_forest_model():
    return joblib.load(RANDOM_FOREST_PATH)
