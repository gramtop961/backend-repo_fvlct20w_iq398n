import os
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conlist, validator

# Database helpers (MongoDB pre-configured in this environment)
from database import db, create_document, get_documents

app = FastAPI(
    title="Yield Prediction API",
    version="1.0.0",
    description="API for crop yield prediction, simulation, ingestion, and field management."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Pydantic Models (Requests/Responses)
# -----------------------------
class Geometry(BaseModel):
    type: str = Field(..., pattern=r"^(Polygon|MultiPolygon)$")
    coordinates: List[Any]

    @validator("coordinates")
    def validate_coords(cls, v, values):  # basic sanity check
        if "type" in values and values["type"] == "Polygon":
            if not (isinstance(v, list) and len(v) > 0 and isinstance(v[0], list)):
                raise ValueError("Invalid Polygon coordinates")
        return v


class SatelliteRecord(BaseModel):
    timestamp: datetime
    ndvi: Optional[float] = None
    evi: Optional[float] = None
    lai: Optional[float] = None
    red: Optional[float] = None
    green: Optional[float] = None
    blue: Optional[float] = None
    nir: Optional[float] = None
    cloud_mask: Optional[int] = Field(None, description="1 if cloud, 0 if clear")
    satellite_name: Optional[str] = None


class WeatherRecord(BaseModel):
    timestamp: date
    rainfall_mm: Optional[float] = 0.0
    tmean: Optional[float] = None
    tmin: Optional[float] = None
    tmax: Optional[float] = None
    humidity: Optional[float] = None
    wind: Optional[float] = None
    radiation: Optional[float] = None


class SoilData(BaseModel):
    texture: Optional[str] = None
    ph: Optional[float] = None
    organic_matter_pct: Optional[float] = Field(None, ge=0, le=100)
    nitrogen_availability: Optional[str] = None
    drainage_class: Optional[str] = None


class FertilizerEvent(BaseModel):
    date: date
    type: Optional[str] = None
    amount_kg_ha: Optional[float] = None


class IrrigationEvent(BaseModel):
    date: date
    mm: Optional[float] = 0.0


class PesticideEvent(BaseModel):
    date: date
    product: Optional[str] = None


class ManagementData(BaseModel):
    planting_date: Optional[date] = None
    seed_variety: Optional[str] = None
    fertilizer_events: Optional[List[FertilizerEvent]] = []
    irrigation_events: Optional[List[IrrigationEvent]] = []
    pesticide_events: Optional[List[PesticideEvent]] = []


class GroundTruthYield(BaseModel):
    harvest_date: Optional[date] = None
    actual_yield_t_ha: Optional[float] = None


class PredictYieldRequest(BaseModel):
    field_id: Optional[str] = None
    name: Optional[str] = None
    geometry: Geometry
    satellite: List[SatelliteRecord] = []
    weather: List[WeatherRecord] = []
    soil: Optional[SoilData] = None
    management: Optional[ManagementData] = None
    ground_truth: Optional[GroundTruthYield] = None


class PredictionDrivers(BaseModel):
    feature_importances: Dict[str, float]
    shap_summary: Optional[Dict[str, Any]] = None
    narrative: str


class Recommendation(BaseModel):
    type: str
    message: str
    rationale: str


class PredictYieldResponse(BaseModel):
    field_id: Optional[str] = None
    yield_t_ha: float
    p10: float
    p50: float
    p90: float
    drivers: PredictionDrivers
    recommendations: List[Recommendation]


class SimulateRequest(BaseModel):
    geometry: Geometry
    baseline: PredictYieldRequest
    adjustments: Dict[str, Any] = Field(
        ..., description="What-if adjustments e.g. {'fertilizer_pct': +10, 'irrigation_mm': +20}"
    )


class SimulateResponse(BaseModel):
    baseline: PredictYieldResponse
    scenario: PredictYieldResponse
    deltas: Dict[str, float]


class FieldItem(BaseModel):
    id: Optional[str] = None
    name: str
    geometry: Geometry
    area_ha: Optional[float] = None


class UploadSatelliteDataRequest(BaseModel):
    field_id: str
    satellite: List[SatelliteRecord]


# -----------------------------
# Utility functions (lightweight mock ML and rules)
# -----------------------------

def _simple_feature_agg(req: PredictYieldRequest) -> Dict[str, float]:
    """Derive a small set of aggregate features to drive mock predictions."""
    ndvi_vals = [r.ndvi for r in req.satellite if r.ndvi is not None and (r.cloud_mask or 0) == 0]
    mean_ndvi = sum(ndvi_vals) / len(ndvi_vals) if ndvi_vals else 0.5

    rain = sum((w.rainfall_mm or 0.0) for w in req.weather)
    tmean = sum((w.tmean or 0.0) for w in req.weather)

    fert_total = sum((e.amount_kg_ha or 0.0) for e in (req.management.fertilizer_events if req.management else []))
    irr_total = sum((e.mm or 0.0) for e in (req.management.irrigation_events if req.management else []))

    soil_ph = (req.soil.ph if req.soil and req.soil.ph is not None else 6.5)
    om = (req.soil.organic_matter_pct if req.soil and req.soil.organic_matter_pct is not None else 2.0)

    return {
        "mean_ndvi": mean_ndvi,
        "cum_rain": rain,
        "sum_tmean": tmean,
        "fert_total": fert_total,
        "irr_total": irr_total,
        "soil_ph": soil_ph,
        "soil_om": om,
    }


def _mock_predict(features: Dict[str, float]) -> Dict[str, float]:
    """Deterministic pseudo-model combining features into yield and quantiles."""
    base = 2.0 + 6.0 * features["mean_ndvi"]
    water_factor = min(1.5, 0.5 + (features["cum_rain"] + features["irr_total"]) / 400.0)
    fert_factor = min(1.4, 0.6 + features["fert_total"] / 250.0)
    soil_factor = 1.0 - abs((features["soil_ph"] - 6.5)) * 0.05 + min(0.2, features["soil_om"] / 20.0)
    pred = base * water_factor * fert_factor * soil_factor
    pred = max(0.5, min(15.0, pred))

    spread = max(0.3, 0.15 * pred)  # uncertainty grows with yield
    p10 = max(0.3, pred - 1.28 * spread)
    p50 = pred
    p90 = min(20.0, pred + 1.28 * spread)

    return {"pred": pred, "p10": p10, "p50": p50, "p90": p90}


def _mock_drivers(features: Dict[str, float], pred: float) -> PredictionDrivers:
    importances = {
        "mean_ndvi": round(min(0.5, features["mean_ndvi"]) * 0.6, 3),
        "cum_rain": round(min(0.3, features["cum_rain"]/800), 3),
        "fert_total": round(min(0.25, features["fert_total"]/500), 3),
        "irr_total": round(min(0.2, features["irr_total"]/500), 3),
        "soil_ph": 0.1,
        "soil_om": 0.12,
    }
    narrative = (
        f"Yield is driven by canopy vigor (NDVI {features['mean_ndvi']:.2f}), water inputs (rain+irrigation {features['cum_rain']+features['irr_total']:.0f} mm), "
        f"and nutrients (fertilizer {features['fert_total']:.0f} kg/ha)."
    )
    return PredictionDrivers(feature_importances=importances, shap_summary=None, narrative=narrative)


def _rules_recommendations(features: Dict[str, float], pred: float) -> List[Recommendation]:
    recs: List[Recommendation] = []
    if features["mean_ndvi"] < 0.5:
        recs.append(Recommendation(
            type="canopy",
            message="Canopy vigor is below optimal. Investigate nutrient or pest stress.",
            rationale="NDVI below 0.50 suggests low biomass accumulation."
        ))
    water_inputs = features["cum_rain"] + features["irr_total"]
    if water_inputs < 250:
        recs.append(Recommendation(
            type="irrigation",
            message="Consider supplemental irrigation to reach 300–400 mm over the season.",
            rationale="Water-limited environments reduce yield potential."
        ))
    if features["fert_total"] < 120:
        recs.append(Recommendation(
            type="nutrition",
            message="Nitrogen applications appear low for high-yield targets.",
            rationale="Total fertilizer <120 kg/ha commonly limits cereals."
        ))
    if abs(features["soil_ph"] - 6.5) > 0.7:
        recs.append(Recommendation(
            type="soil",
            message="Soil pH outside optimal range; consider liming or sulfur amendments.",
            rationale="pH far from 6.5 reduces nutrient availability."
        ))
    return recs


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "Yield Prediction Backend Running"}


@app.get("/test")
def test_database():
    response = {"backend": "running", "database": "not configured"}
    try:
        if db is not None:
            response["database"] = "connected"
            response["collections"] = db.list_collection_names()
    except Exception as e:
        response["database"] = f"error: {str(e)}"
    return response


@app.post("/v1/predict_yield", response_model=PredictYieldResponse)
async def predict_yield(payload: PredictYieldRequest):
    if payload is None or payload.geometry is None:
        raise HTTPException(status_code=400, detail="Missing geometry")

    feats = _simple_feature_agg(payload)
    pred = _mock_predict(feats)
    drivers = _mock_drivers(feats, pred["pred"])
    recs = _rules_recommendations(feats, pred["pred"])

    # Persist request + response
    record = {
        "type": "prediction",
        "request": payload.model_dump(),
        "features": feats,
        "result": pred,
        "drivers": drivers.model_dump(),
        "recommendations": [r.model_dump() for r in recs],
        "created_at": datetime.utcnow(),
    }
    try:
        if db is not None:
            db["prediction"].insert_one(record)
    except Exception:
        pass

    return PredictYieldResponse(
        field_id=payload.field_id,
        yield_t_ha=pred["pred"],
        p10=pred["p10"],
        p50=pred["p50"],
        p90=pred["p90"],
        drivers=drivers,
        recommendations=recs,
    )


@app.post("/v1/simulate", response_model=SimulateResponse)
async def simulate(payload: SimulateRequest):
    # Baseline
    base_feats = _simple_feature_agg(payload.baseline)
    base_pred = _mock_predict(base_feats)
    base_resp = PredictYieldResponse(
        field_id=payload.baseline.field_id,
        yield_t_ha=base_pred["pred"], p10=base_pred["p10"], p50=base_pred["p50"], p90=base_pred["p90"],
        drivers=_mock_drivers(base_feats, base_pred["pred"]),
        recommendations=_rules_recommendations(base_feats, base_pred["pred"]) 
    )

    # Apply adjustments (very simple effects)
    adj = payload.adjustments or {}
    sim_req = payload.baseline
    # Adjust fertilizer
    fert_pct = float(adj.get("fertilizer_pct", 0))
    if sim_req.management and sim_req.management.fertilizer_events:
        for e in sim_req.management.fertilizer_events:
            if e.amount_kg_ha is not None:
                e.amount_kg_ha *= (1.0 + fert_pct/100.0)
    # Adjust irrigation total add mm
    add_irr = float(adj.get("irrigation_mm", 0))
    if sim_req.management:
        sim_req.management.irrigation_events = sim_req.management.irrigation_events or []
        if add_irr != 0:
            sim_req.management.irrigation_events.append(IrrigationEvent(date=date.today(), mm=max(0.0, add_irr)))

    sim_feats = _simple_feature_agg(sim_req)
    sim_pred = _mock_predict(sim_feats)
    sim_resp = PredictYieldResponse(
        field_id=payload.baseline.field_id,
        yield_t_ha=sim_pred["pred"], p10=sim_pred["p10"], p50=sim_pred["p50"], p90=sim_pred["p90"],
        drivers=_mock_drivers(sim_feats, sim_pred["pred"]),
        recommendations=_rules_recommendations(sim_feats, sim_pred["pred"]) 
    )

    deltas = {"yield_delta": sim_resp.yield_t_ha - base_resp.yield_t_ha}

    return SimulateResponse(baseline=base_resp, scenario=sim_resp, deltas=deltas)


@app.get("/v1/fields", response_model=List[FieldItem])
async def list_fields(limit: int = 50):
    items: List[FieldItem] = []
    try:
        if db is not None:
            for doc in db["field"].find().limit(limit):
                items.append(FieldItem(
                    id=str(doc.get("_id")),
                    name=doc.get("name", "Unnamed Field"),
                    geometry=doc["geometry"],
                    area_ha=doc.get("area_ha")
                ))
    except Exception:
        pass
    return items


@app.post("/v1/upload_satellite_data")
async def upload_satellite_data(payload: UploadSatelliteDataRequest):
    if not payload.field_id:
        raise HTTPException(status_code=400, detail="field_id is required")
    try:
        count = 0
        if db is not None:
            for rec in payload.satellite:
                db["satellite"].insert_one({
                    "field_id": payload.field_id,
                    **rec.model_dump(),
                    "created_at": datetime.utcnow()
                })
                count += 1
        return {"status": "ok", "ingested": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers examples
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return app.responses.JSONResponse(status_code=exc.status_code, content={
        "error": {
            "code": exc.status_code,
            "message": exc.detail
        }
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
