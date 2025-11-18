"""
Model persistence utilities for saving and loading trained NILM models
"""
import os
import joblib
import json
from typing import Dict, Any
from datetime import datetime


class ModelManager:
    """Manage saving and loading of NILM models"""

    def __init__(self, models_dir: str = "./models"):
        """
        Initialize model manager

        Args:
            models_dir: Directory to store models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def save_model(
        self,
        model,
        model_name: str,
        algorithm_type: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Save a trained NILM model

        Args:
            model: The trained model object
            model_name: Name for the model (e.g., 'co_model_v1')
            algorithm_type: Type of algorithm ('CO' or 'FHMM')
            metadata: Optional metadata dictionary

        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{algorithm_type}_{model_name}_{timestamp}"

        # Save model
        model_path = os.path.join(self.models_dir, f"{base_name}.pkl")
        joblib.dump(model, model_path)

        # Save metadata
        if metadata is None:
            metadata = {}

        metadata.update({
            'algorithm_type': algorithm_type,
            'model_name': model_name,
            'timestamp': timestamp,
            'saved_at': datetime.now().isoformat(),
        })

        metadata_path = os.path.join(self.models_dir, f"{base_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to: {model_path}")
        print(f"Metadata saved to: {metadata_path}")

        return model_path

    def load_model(self, model_path: str):
        """
        Load a trained NILM model

        Args:
            model_path: Path to the saved model file

        Returns:
            Loaded model object
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")

        # Try to load metadata
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Model metadata: {metadata}")

        return model

    def list_models(self) -> list:
        """
        List all saved models

        Returns:
            List of model file paths
        """
        model_files = [
            f for f in os.listdir(self.models_dir)
            if f.endswith('.pkl')
        ]

        models_info = []
        for model_file in sorted(model_files):
            model_path = os.path.join(self.models_dir, model_file)
            metadata_path = model_path.replace('.pkl', '_metadata.json')

            info = {
                'path': model_path,
                'name': model_file,
            }

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    info['metadata'] = json.load(f)

            models_info.append(info)

        return models_info

    def delete_model(self, model_path: str):
        """
        Delete a saved model and its metadata

        Args:
            model_path: Path to the model file
        """
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Deleted model: {model_path}")

        metadata_path = model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            print(f"Deleted metadata: {metadata_path}")
