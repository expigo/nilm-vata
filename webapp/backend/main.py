"""
NILM VATA Web API
FastAPI backend for NILM model serving and real-time disaggregation
"""
from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import os
import json
import asyncio
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from datasets import REDDLoader, DatasetRegistry
from algorithms import (
    CombinatorialOptimization,
    FHMM,
    MeanAlgorithm,
    KNNDisaggregator,
    DEEP_LEARNING_AVAILABLE
)
from utils.model_persistence import ModelManager
from utils.metrics import calculate_metrics
from utils.energy_cost import EnergyCostCalculator

# Create FastAPI app
app = FastAPI(
    title="NILM VATA API",
    description="Non-Intrusive Load Monitoring API for energy disaggregation",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend')
app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, 'static')), name="static")

# Global state
model_manager = ModelManager()
trained_models = {}  # Cache of loaded models
training_status = {}  # Track training progress


# ============================================================================
# Pydantic Models
# ============================================================================

class TrainRequest(BaseModel):
    algorithm: str
    dataset: str = "synthetic"
    building_id: int = 1
    train_ratio: float = 0.7
    params: Optional[Dict] = {}
    model_name: Optional[str] = None


class PredictRequest(BaseModel):
    model_id: str
    power_values: List[float]


class ModelInfo(BaseModel):
    id: str
    algorithm: str
    created_at: str
    metadata: Optional[Dict] = {}


class DatasetInfo(BaseModel):
    name: str
    available: bool
    info: Dict


# ============================================================================
# Utility Functions
# ============================================================================

def get_algorithm_instance(algorithm: str, params: Dict = None):
    """Get algorithm instance by name"""
    params = params or {}

    algorithms = {
        'co': lambda: CombinatorialOptimization(**params),
        'combinatorial_optimization': lambda: CombinatorialOptimization(**params),
        'fhmm': lambda: FHMM(**params),
        'mean': lambda: MeanAlgorithm(),
        'knn': lambda: KNNDisaggregator(**params),
    }

    algo_name = algorithm.lower()
    if algo_name not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return algorithms[algo_name]()


async def train_model_async(task_id: str, request: TrainRequest):
    """Train model asynchronously"""
    try:
        training_status[task_id] = {
            'status': 'loading_data',
            'progress': 0,
            'message': 'Loading dataset...'
        }

        # Load dataset
        if request.dataset == "synthetic":
            loader = REDDLoader()
            train_data, test_data = loader.get_train_test_split(train_ratio=request.train_ratio)
        else:
            # Try to load real dataset
            dataset = DatasetRegistry.get_dataset(request.dataset)
            train_data, test_data = dataset.get_train_test_split(
                building_id=request.building_id,
                train_ratio=request.train_ratio
            )

        training_status[task_id] = {
            'status': 'training',
            'progress': 30,
            'message': f'Training {request.algorithm}...'
        }

        # Create and train model
        model = get_algorithm_instance(request.algorithm, request.params)
        model.train(train_data['mains'], train_data['appliances'])

        training_status[task_id] = {
            'status': 'evaluating',
            'progress': 70,
            'message': 'Evaluating model...'
        }

        # Evaluate
        predictions = model.disaggregate(test_data['mains'], num_samples=1000)
        metrics = calculate_metrics(test_data['appliances'], predictions)

        # Calculate average F1
        avg_f1 = sum(m['F1'] for m in metrics.values()) / len(metrics)

        # Save model
        model_name = request.model_name or f"{request.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        metadata = {
            'algorithm': request.algorithm,
            'dataset': request.dataset,
            'building_id': request.building_id,
            'avg_f1': avg_f1,
            'metrics': {k: v['F1'] for k, v in metrics.items()},
            'params': request.params
        }

        model_path = model_manager.save_model(
            model,
            model_name,
            request.algorithm.upper(),
            metadata
        )

        # Cache the model
        model_id = os.path.basename(model_path)
        trained_models[model_id] = model

        training_status[task_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Training completed!',
            'model_id': model_id,
            'avg_f1': avg_f1,
            'metrics': metrics
        }

    except Exception as e:
        training_status[task_id] = {
            'status': 'failed',
            'progress': 0,
            'message': f'Training failed: {str(e)}'
        }


# ============================================================================
# API Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web dashboard"""
    index_path = os.path.join(FRONTEND_DIR, 'templates', 'index.html')
    with open(index_path, 'r') as f:
        return f.read()


@app.get("/api")
async def api_root():
    """API root endpoint"""
    return {
        "message": "NILM VATA API",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/algorithms")
async def list_algorithms():
    """List available algorithms"""
    algorithms = [
        {
            "id": "co",
            "name": "Combinatorial Optimization",
            "type": "traditional",
            "speed": "fast",
            "accuracy": "medium"
        },
        {
            "id": "fhmm",
            "name": "Factorial HMM",
            "type": "traditional",
            "speed": "medium",
            "accuracy": "medium-high"
        },
        {
            "id": "knn",
            "name": "k-Nearest Neighbors",
            "type": "ml",
            "speed": "medium",
            "accuracy": "high"
        },
        {
            "id": "mean",
            "name": "Mean Baseline",
            "type": "baseline",
            "speed": "very fast",
            "accuracy": "low"
        }
    ]

    if DEEP_LEARNING_AVAILABLE:
        algorithms.extend([
            {
                "id": "seq2seq",
                "name": "Seq2Seq LSTM",
                "type": "deep_learning",
                "speed": "slow",
                "accuracy": "very high"
            },
            {
                "id": "dae",
                "name": "Denoising Autoencoder",
                "type": "deep_learning",
                "speed": "slow",
                "accuracy": "very high"
            }
        ])

    return {"algorithms": algorithms}


@app.get("/api/datasets")
async def list_datasets():
    """List available datasets"""
    datasets = DatasetRegistry.list_datasets()
    availability = DatasetRegistry.check_availability()
    info = DatasetRegistry.get_dataset_info()

    result = []
    for name in datasets:
        dataset_info = {
            "id": name,
            "name": info.get(name, {}).get('name', name.upper()),
            "available": availability.get(name, False),
            "info": info.get(name, {})
        }
        result.append(dataset_info)

    # Add synthetic dataset
    result.insert(0, {
        "id": "synthetic",
        "name": "Synthetic REDD",
        "available": True,
        "info": {
            "homes": "unlimited",
            "sampling": "1s",
            "size": "generated"
        }
    })

    return {"datasets": result}


@app.post("/api/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Train a new model"""
    import uuid
    task_id = str(uuid.uuid4())

    # Start training in background
    background_tasks.add_task(train_model_async, task_id, request)

    return {
        "task_id": task_id,
        "status": "started",
        "message": "Training started in background"
    }


@app.get("/api/train/{task_id}")
async def get_training_status(task_id: str):
    """Get training status"""
    if task_id not in training_status:
        raise HTTPException(status_code=404, detail="Task not found")

    return training_status[task_id]


@app.get("/api/models")
async def list_models():
    """List all saved models"""
    models = model_manager.list_models()

    result = []
    for model_info in models:
        result.append({
            "id": model_info['name'],
            "path": model_info['path'],
            "metadata": model_info.get('metadata', {})
        })

    return {"models": result}


@app.post("/api/predict")
async def predict(request: PredictRequest):
    """Make predictions with a trained model"""
    # Load model if not cached
    if request.model_id not in trained_models:
        # Find model path
        models = model_manager.list_models()
        model_path = None

        for m in models:
            if m['name'] == request.model_id:
                model_path = m['path']
                break

        if not model_path:
            raise HTTPException(status_code=404, detail="Model not found")

        # Load model
        model = model_manager.load_model(model_path)
        trained_models[request.model_id] = model

    model = trained_models[request.model_id]

    # Create DataFrame from power values
    import pandas as pd
    mains_data = pd.DataFrame({'power': request.power_values})

    # Predict
    predictions = model.disaggregate(mains_data, num_samples=len(request.power_values))

    # Convert to JSON-serializable format
    result = {}
    for app_name, app_data in predictions.items():
        result[app_name] = app_data['power'].tolist()

    return {
        "predictions": result,
        "num_samples": len(request.power_values)
    }


@app.websocket("/ws/disaggregate")
async def websocket_disaggregate(websocket: WebSocket):
    """WebSocket endpoint for real-time disaggregation"""
    await websocket.accept()

    try:
        # Receive initial configuration
        config = await websocket.receive_json()
        model_id = config.get('model_id')

        if not model_id:
            await websocket.send_json({"error": "model_id required"})
            await websocket.close()
            return

        # Load model
        if model_id not in trained_models:
            models = model_manager.list_models()
            model_path = None

            for m in models:
                if m['name'] == model_id:
                    model_path = m['path']
                    break

            if not model_path:
                await websocket.send_json({"error": "Model not found"})
                await websocket.close()
                return

            model = model_manager.load_model(model_path)
            trained_models[model_id] = model

        model = trained_models[model_id]

        await websocket.send_json({"status": "ready", "message": "Model loaded"})

        # Process streaming data
        while True:
            data = await websocket.receive_json()

            if 'power' in data:
                power_value = data['power']

                # For real-time, we'd use streaming disaggregator
                # For now, simple prediction
                import pandas as pd
                mains_data = pd.DataFrame({'power': [power_value]})

                try:
                    predictions = model.disaggregate(mains_data, num_samples=1)

                    result = {
                        'timestamp': datetime.now().isoformat(),
                        'mains': power_value,
                        'appliances': {}
                    }

                    for app_name, app_data in predictions.items():
                        result['appliances'][app_name] = float(app_data['power'].iloc[0])

                    await websocket.send_json(result)

                except Exception as e:
                    await websocket.send_json({"error": str(e)})

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


@app.get("/api/cost")
async def calculate_cost(
    appliance_data: str,
    rate_per_kwh: float = 0.12,
    time_period: str = "daily"
):
    """Calculate energy cost"""
    # This is a simplified version
    # In real implementation, would accept actual data

    cost_calc = EnergyCostCalculator(rate_per_kwh=rate_per_kwh)

    return {
        "message": "Cost calculation endpoint",
        "rate": rate_per_kwh,
        "period": time_period
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("="*70)
    print("NILM VATA Web API")
    print("="*70)
    print("\nStarting server...")
    print("API Docs: http://localhost:8000/docs")
    print("WebSocket: ws://localhost:8000/ws/disaggregate")
    print("\nPress Ctrl+C to stop")
    print("="*70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
