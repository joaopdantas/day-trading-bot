"""
Ensemble model module for combining multiple prediction models.
"""

import os
import logging
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Class for creating ensemble models from multiple base models.
    This improves prediction stability and accuracy.
    """
    
    def __init__(self, models: List, ensemble_type: str = 'average'):
        """
        Initialize the ensemble model.
        
        Args:
            models: List of trained PredictionModel instances
            ensemble_type: Type of ensemble ('average', 'weighted')
        """
        self.models = models
        self.ensemble_type = ensemble_type
        self.weights = None
        self.target_scaler = None
        
        # Validate that all models are trained
        for i, model in enumerate(models):
            if model.model is None:
                raise ValueError(f"Model {i} is not trained")
                
        # Use the first model's scaler for inverse transformation if needed
        if models and models[0].scaler is not None:
            self.target_scaler = models[0].scaler
            
        # Default to equal weights
        self.weights = np.ones(len(models)) / len(models)
    
    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set weights for weighted ensemble.
        
        Args:
            weights: Array of weights for each model (must sum to 1)
        """
        if len(weights) != len(self.models):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(self.models)})")
            
        if abs(np.sum(weights) - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {np.sum(weights)}, normalizing to 1.0")
            weights = weights / np.sum(weights)
            
        self.weights = weights
        logger.info(f"Ensemble weights set: {self.weights}")
    
    def predict(self, X: np.ndarray, inverse_transform: bool = True) -> np.ndarray:
        """
        Make predictions using the ensemble of models.
        
        Args:
            X: Input features
            inverse_transform: Whether to inverse-transform the predictions
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        
        # Check for NaN values in input
        if np.isnan(X).any():
            logger.warning("Input data contains NaN values. Replacing with zeros.")
            X = np.nan_to_num(X, nan=0.0)
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            try:
                pred = model.predict(X, inverse_transform=False)  # Keep scaled predictions for now
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error getting prediction from model {i}: {e}")
                # Add placeholder predictions of correct shape
                placeholder = np.zeros((X.shape[0], 1))
                predictions.append(placeholder)
            
        # Stack predictions horizontally for aggregation
        stacked_preds = np.hstack(predictions)
        
        # Apply ensemble method
        if self.ensemble_type == 'weighted':
            # Apply weights for weighted average
            ensemble_preds = np.sum(stacked_preds * self.weights, axis=1, keepdims=True)
        else:
            # Default to simple average
            ensemble_preds = np.mean(stacked_preds, axis=1, keepdims=True)
            
        # Inverse transform if requested
        if inverse_transform and self.target_scaler is not None:
            try:
                ensemble_preds = self.target_scaler.inverse_transform(ensemble_preds)
            except Exception as e:
                logger.warning(f"Could not inverse transform ensemble predictions: {e}")
                
        return ensemble_preds
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, output_dir: str = None) -> Dict[str, float]:
        """
        Evaluate the ensemble model.
        
        Args:
            X_test: Test features
            y_test: Test targets
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Handle NaN values in test data
        if np.isnan(X_test).any():
            logger.warning("Test data contains NaN values. Replacing with zeros.")
            X_test = np.nan_to_num(X_test, nan=0.0)
            
        # Get ensemble predictions
        y_pred = self.predict(X_test)
        
        # Check if y_test needs inverse transform
        if self.target_scaler is not None and not np.array_equal(y_test, self.target_scaler.inverse_transform(y_test)):
            try:
                y_test_orig = self.target_scaler.inverse_transform(y_test)
            except Exception as e:
                logger.warning(f"Error inverse transforming y_test: {e}")
                y_test_orig = y_test
        else:
            y_test_orig = y_test
        
        # Calculate metrics
        try:
            mse = mean_squared_error(y_test_orig, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_orig, y_pred)
            r2 = r2_score(y_test_orig, y_pred)
            
            # Calculate MAPE with safeguards against division by zero
            try:
                # Adjust y_test to avoid division by zero
                y_test_adj = np.where(np.abs(y_test_orig) < 1e-6, 1e-6, y_test_orig)
                mape = np.mean(np.abs((y_test_orig - y_pred) / y_test_adj)) * 100
                # Cap at a reasonable value
                mape = min(float(mape), 1000)
            except Exception as e:
                logger.warning(f"Error calculating MAPE: {e}")
                mape = 1000.0  # Default to high value
            
            # Direction accuracy with safeguards
            if len(y_test_orig) > 1:
                try:
                    actual_diff = np.diff(y_test_orig.flatten())
                    pred_diff = np.diff(y_pred.flatten())
                    
                    # Filter out very small changes to avoid noise
                    valid_indices = np.where(np.abs(actual_diff) > 1e-6)[0]
                    
                    if len(valid_indices) > 0:
                        actual_direction = actual_diff[valid_indices] > 0
                        pred_direction = pred_diff[valid_indices] > 0
                        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
                    else:
                        direction_accuracy = 50.0  # Default to random chance
                except Exception as e:
                    logger.warning(f"Error calculating direction accuracy: {e}")
                    direction_accuracy = 50.0
            else:
                direction_accuracy = 50.0
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'mape': float(mape),
                'direction_accuracy': float(direction_accuracy)
            }
            
            # Log results
            logger.info(f"Ensemble Model Evaluation Results:")
            for metric, value in metrics.items():
                logger.info(f"{metric.upper()}: {value:.4f}")
                
            # Create visualization if output directory provided
            if output_dir:
                self._create_evaluation_plots(y_test_orig, y_pred, metrics, output_dir, X_test)
                
            return metrics
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {}
            
    def _create_evaluation_plots(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        metrics: Dict[str, float],
        output_dir: str,
        X_test: np.ndarray = None
    ) -> None:
        """
        Create evaluation plots for the ensemble model.
        
        Args:
            y_test: Test targets (original scale)
            y_pred: Predictions (original scale)
            metrics: Dictionary of evaluation metrics
            output_dir: Directory to save plots
            X_test: Test features for individual model predictions
        """
        # Create prediction vs actual plot
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual', color='blue', alpha=0.7)
        plt.plot(y_pred, label='Ensemble Prediction', color='red', alpha=0.7)
        plt.title('Ensemble Prediction vs Actual')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add metrics annotation
        metrics_text = "\n".join([f"{m.upper()}: {v:.4f}" for m, v in metrics.items()])
        plt.figtext(0.02, 0.02, metrics_text, fontsize=9, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'ensemble_prediction_vs_actual.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create individual model predictions for comparison
        if X_test is not None:
            plt.figure(figsize=(12, 6))
            plt.plot(y_test, label='Actual', color='blue', alpha=0.7)
            
            for i, model in enumerate(self.models):
                try:
                    y_pred_single = model.predict(X_test)
                    plt.plot(y_pred_single, label=f'Model {i+1}', alpha=0.4)
                except Exception as e:
                    logger.warning(f"Error getting prediction from model {i}: {e}")
                    
            plt.plot(y_pred, label='Ensemble', color='red', linewidth=2)
            plt.title('Individual Models vs Ensemble Prediction')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'ensemble_vs_individual_models.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()