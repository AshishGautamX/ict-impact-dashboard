from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import (
    PredictionRequest,
    SatisfactionPrediction,
    EfficiencyPrediction,
    ScenarioSimulation,
)
from app.services.ml_service import ml_service

router = APIRouter()


@router.post("/satisfaction", response_model=SatisfactionPrediction)
async def predict_satisfaction(request: PredictionRequest):
    """Predict satisfaction level."""
    try:
        features = {
            'infrastructure_score': request.infrastructureScore,
            'barrier_score': request.barrierScore,
            'college_id': request.collegeId,
            'automation_system': request.automationSystem,
            'awareness_level': request.awarenessLevel,
        }
        
        prediction = ml_service.predict_satisfaction(features)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/efficiency", response_model=EfficiencyPrediction)
async def predict_efficiency(request: PredictionRequest):
    """Predict service efficiency."""
    try:
        features = {
            'infrastructure_score': request.infrastructureScore,
            'barrier_score': request.barrierScore,
            'college_id': request.collegeId,
            'automation_system': request.automationSystem,
            'awareness_level': request.awarenessLevel,
        }
        
        prediction = ml_service.predict_efficiency(features)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel

class ScenarioRequest(BaseModel):
    current: PredictionRequest
    proposed: PredictionRequest

@router.post("/scenario", response_model=ScenarioSimulation)
async def simulate_scenario(request: ScenarioRequest):
    """Simulate impact of proposed changes."""
    try:
        current_features = {
            'infrastructure_score': request.current.infrastructureScore,
            'barrier_score': request.current.barrierScore,
            'college_id': request.current.collegeId,
            'automation_system': request.current.automationSystem,
            'awareness_level': request.current.awarenessLevel,
        }
        
        proposed_features = {
            'infrastructure_score': request.proposed.infrastructureScore,
            'barrier_score': request.proposed.barrierScore,
            'college_id': request.proposed.collegeId,
            'automation_system': request.proposed.automationSystem,
            'awareness_level': request.proposed.awarenessLevel,
        }
        
        simulation = ml_service.simulate_scenario(current_features, proposed_features)
        return simulation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clusters")
async def get_college_clusters():
    """Get college clusters using the clustering model."""
    try:
        clusters = ml_service.get_college_clusters()
        return clusters
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations")
async def get_recommendations(college_id: str = Query(...)):
    """Get AI-powered recommendations for a college."""
    try:
        # In production, fetch actual college data
        college_data = {
            'college_id': college_id,
            'infrastructure_score': 3,
            'barrier_score': 3,
            'ict_training_received': False,
        }
        
        recommendations = ml_service.get_recommendations(college_data)
        return {'recommendations': recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
