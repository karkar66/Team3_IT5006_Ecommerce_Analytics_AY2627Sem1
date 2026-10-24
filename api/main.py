"""Serve the frozen delivery-time regression pipeline.

The bundle's contract is the 17 *engineered* order-level features, not raw CSV
rows: the caller is responsible for reproducing the notebook's checkout feature
builder. See the "Winning Model for FastAPI" section of
notebooks/phase2 regression edits.ipynb.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Annotated, Literal

BUNDLE_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "delivery_regression.joblib"

# Populated once at startup by the lifespan handler below.
STATE: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pickle once per worker process, never per request."""
    if not BUNDLE_PATH.is_file():
        raise RuntimeError(
            f"Model bundle not found at {BUNDLE_PATH}. Run the final cell of "
            "notebooks/phase2 regression edits.ipynb to export it."
        )
    bundle = joblib.load(BUNDLE_PATH)
    pipeline = bundle["pipeline"]

    # Derive the accepted category values from the fitted encoder itself, so
    # request validation can never drift away from what the model knows.
    column_transformer = pipeline.named_steps["preprocess"]
    categorical_columns = next(
        columns for name, _, columns in column_transformer.transformers_ if name == "cat"
    )
    encoder = column_transformer.named_transformers_["cat"].named_steps["onehot"]

    STATE.update(
        bundle=bundle,
        pipeline=pipeline,
        columns=list(bundle["feature_columns"]),
        dtypes=bundle["feature_dtypes"],
        vocabulary={
            column: set(categories.tolist())
            for column, categories in zip(categorical_columns, encoder.categories_)
        },
    )
    yield
    STATE.clear()


app = FastAPI(
    title="Olist delivery-time estimate",
    description=(
        "Predicts purchase-to-delivery days at order placement. Inputs are "
        "engineered order-level checkout features, not raw Olist CSV rows."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class CheckoutFeatures(BaseModel):
    """One order's checkout snapshot, as 17 engineered features.

    The categorical dtypes are deliberately unintuitive and must match how the
    pipeline was fitted: `is_weekend` and `is_interstate` are the *strings*
    "True"/"False", while `purchase_month` and `purchase_hour` are integers.
    """

    # --- numeric (median-imputed inside the pipeline when null) ---
    n_items: Annotated[int, Field(ge=1, examples=[1])]
    n_unique_products: Annotated[int, Field(ge=1, examples=[1])]
    n_product_categories: Annotated[int, Field(ge=1, examples=[1])]
    n_sellers: Annotated[int, Field(ge=1, examples=[1])]
    n_seller_states: Annotated[int, Field(ge=1, examples=[1])]
    total_freight: Annotated[float, Field(ge=0, examples=[13.29])]
    total_product_weight_g: Optional[Annotated[float, Field(ge=0)]] = Field(
        default=None, examples=[650.0]
    )
    total_physical_score: Optional[Annotated[float, Field(ge=0)]] = Field(
        default=None, examples=[2293200.0]
    )
    max_seller_buyer_distance_km: Optional[Annotated[float, Field(ge=0)]] = Field(
        default=None, examples=[301.23]
    )
    freight_ratio: Annotated[float, Field(ge=0, examples=[0.2219])]

    # --- categorical (one-hot encoded) ---
    purchase_month: Annotated[int, Field(ge=1, le=12, examples=[9])]
    purchase_day_of_week: Literal[
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]
    purchase_hour: Annotated[int, Field(ge=0, le=23, examples=[8])]
    main_product_category: Annotated[str, Field(examples=["cool_stuff"])]
    farthest_seller_state: Annotated[str, Field(examples=["SP"])]
    is_weekend: Literal["True", "False"]
    is_interstate: Literal["True", "False"]

    @model_validator(mode="after")
    def reject_unseen_categories(self):
        """Fail loudly on categories the encoder never saw.

        The pipeline uses OneHotEncoder(handle_unknown="ignore"), so an unknown
        value is silently encoded as all-zeros and still returns a plausible
        number. Rejecting it here turns a silent accuracy loss into a 422.
        """
        vocabulary = STATE.get("vocabulary")
        if not vocabulary:  # model not loaded (e.g. schema-only use)
            return self
        unseen = {
            column: getattr(self, column)
            for column, allowed in vocabulary.items()
            if getattr(self, column) not in allowed
        }
        if unseen:
            raise ValueError(
                "values never seen in training, which the encoder would "
                f"silently zero-encode: {unseen}"
            )
        return self


class Prediction(BaseModel):
    predicted_delivery_days: float
    model_name: str
    training_purchase_cutoff: str


def to_frame(rows: List[CheckoutFeatures]) -> pd.DataFrame:
    """Build the exact frame the fitted ColumnTransformer expects.

    Column *order* matters: ColumnTransformer selects by position-resolved name,
    so reindex to the bundle's recorded feature_columns before predicting.
    """
    frame = pd.DataFrame([row.model_dump() for row in rows])[STATE["columns"]]
    return frame.astype(
        {column: dtype for column, dtype in STATE["dtypes"].items() if column in frame.columns}
    )


def _as_prediction(days: float) -> Prediction:
    bundle = STATE["bundle"]
    return Prediction(
        predicted_delivery_days=round(float(days), 2),
        model_name=bundle["model_name"],
        training_purchase_cutoff=bundle["training_purchase_cutoff"],
    )


@app.post("/predict", response_model=Prediction)
def predict(payload: CheckoutFeatures) -> Prediction:
    """Estimate delivery days for a single order."""
    predicted = STATE["pipeline"].predict(to_frame([payload]))[0]
    return _as_prediction(predicted)


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness plus the provenance of the loaded artifact."""
    if "bundle" not in STATE:
        raise HTTPException(status_code=503, detail="model not loaded")
    bundle = STATE["bundle"]
    return {
        "status": "ok",
        "model": bundle["model_name"],
        "target": bundle["target"],
        "prediction_point": bundle["prediction_point"],
        "features": len(STATE["columns"]),
        "training_purchase_cutoff": bundle["training_purchase_cutoff"],
        "trained_with": bundle["versions"],
    }
