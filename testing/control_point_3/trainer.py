import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                            accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                             GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load

# Configure logging
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Class for training and evaluating machine learning models for financial forecasting.
    
    This class handles the entire model training pipeline including data preparation,
    model selection, hyperparameter tuning, and evaluation.
    """
    
    def __init__(self, 
                 target_type: str = 'regression',
                 cv_folds: int = 5,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize the model trainer.
        
        Args:
            target_type: Type of prediction task ('regression' or 'classification')
            cv_folds: Number of folds for cross-validation
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 to use all processors)
        """
        self.target_type = target_type
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        
        logger.info(f"Initialized ModelTrainer for {target_type} task")
    
    def prepare_data(self, 
                    data: pd.DataFrame, 
                    target_col: str,
                    feature_cols: List[str] = None, 
                    scale_features: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training and evaluation.
        
        Args:
            data: DataFrame with features and target
            target_col: Column name of the target variable
            feature_cols: List of feature column names (if None, use all except target)
            scale_features: Whether to scale features using StandardScaler
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            # Remove rows with NaN in target column
            data = data.dropna(subset=[target_col])
            
            # Select features
            if feature_cols is None:
                # Exclude target and future-related columns
                exclude_patterns = ['future_', 'target_', 'label', target_col]
                feature_cols = [col for col in data.columns 
                              if not any(pattern in col for pattern in exclude_patterns)]
            
            # Get features and target
            X = data[feature_cols].copy()
            y = data[target_col].copy()
            
            # Handle any remaining NaNs in features
            X = X.fillna(method='ffill').fillna(0)
            
            # Split data chronologically (time series aware)
            train_size = int((1 - self.test_size) * len(data))
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
            
            # Scale features if requested
            if scale_features:
                scaler = StandardScaler()
                X_train = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                X_test = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )
                logger.info("Features scaled using StandardScaler")
            
            # Store column names for later use
            self.feature_cols = list(X_train.columns)
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def get_default_models(self) -> Dict[str, Any]:
        """
        Get default models based on the target type.
        
        Returns:
            Dictionary of model names and initialized models
        """
        if self.target_type == 'regression':
            models = {
                'ridge': Ridge(random_state=self.random_state),
                'lasso': Lasso(random_state=self.random_state),
                'rf': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
                'gbm': GradientBoostingRegressor(random_state=self.random_state),
                'xgb': xgb.XGBRegressor(random_state=self.random_state, n_jobs=self.n_jobs),
                'lgbm': lgb.LGBMRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
            }
        else:  # classification
            models = {
                'logistic': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                'gbm': GradientBoostingClassifier(random_state=self.random_state),
                'xgb': xgb.XGBClassifier(random_state=self.random_state, n_jobs=self.n_jobs),
                'lgbm': lgb.LGBMClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
            }
            
        logger.info(f"Created {len(models)} default models for {self.target_type}")
        return models
    
    def get_default_param_grids(self) -> Dict[str, Dict[str, List[Any]]]:
        """
        Get default parameter grids for hyperparameter tuning.
        
        Returns:
            Dictionary of model names and their parameter grids
        """
        if self.target_type == 'regression':
            param_grids = {
                'ridge': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                },
                'lasso': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                },
                'rf': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                },
                'gbm': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                },
                'xgb': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                },
                'lgbm': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                }
            }
        else:  # classification
            param_grids = {
                'logistic': {
                    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l2'],
                },
                'rf': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                },
                'gbm': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                },
                'xgb': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                },
                'lgbm': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                }
            }
            
        logger.info(f"Created parameter grids for {len(param_grids)} models")
        return param_grids
    
    def get_scoring_metric(self) -> str:
        """
        Get appropriate scoring metric based on target type.
        
        Returns:
            String name of scoring metric
        """
        if self.target_type == 'regression':
            return 'neg_mean_squared_error'
        else:  # classification
            return 'roc_auc'
    
    def train_models(self, 
                    X_train: pd.DataFrame, 
                    y_train: pd.Series,
                    models: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train multiple models on the training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            models: Dictionary of model name and model object (if None, use defaults)
            
        Returns:
            Dictionary of trained models
        """
        if models is None:
            models = self.get_default_models()
            
        trained_models = {}
        
        for name, model in models.items():
            try:
                logger.info(f"Training {name} model...")
                model.fit(X_train, y_train)
                trained_models[name] = model
                logger.info(f"{name} model trained successfully")
            except Exception as e:
                logger.error(f"Error training {name} model: {e}")
                
        self.models = trained_models
        return trained_models
    
    def tune_hyperparameters(self,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           models: Dict[str, Any] = None,
                           param_grids: Dict[str, Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters for models using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            models: Dictionary of model name and model object (if None, use defaults)
            param_grids: Dictionary of parameter grids (if None, use defaults)
            
        Returns:
            Dictionary of tuned models
        """
        if models is None:
            models = self.get_default_models()
            
        if param_grids is None:
            param_grids = self.get_default_param_grids()
            
        scoring = self.get_scoring_metric()
        tuned_models = {}
        
        # Create time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        for name, model in models.items():
            if name in param_grids:
                try:
                    logger.info(f"Tuning hyperparameters for {name} model...")
                    
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grids[name],
                        cv=tscv,
                        scoring=scoring,
                        n_jobs=self.n_jobs,
                        verbose=1
                    )
                    
                    grid_search.fit(X_train, y_train)
                    
                    best_model = grid_search.best_estimator_
                    tuned_models[name] = best_model
                    
                    logger.info(f"{name} model tuned successfully")
                    logger.info(f"Best parameters: {grid_search.best_params_}")
                    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error tuning {name} model: {e}")
                    # Fall back to untuned model if tuning fails
                    if name in self.models:
                        tuned_models[name] = self.models[name]
                        logger.warning(f"Using untuned {name} model instead")
            else:
                # If no param grid provided, use the already trained model
                if name in self.models:
                    tuned_models[name] = self.models[name]
                
        self.models = tuned_models
        return tuned_models
    
    def evaluate_models(self,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       models: Dict[str, Any] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate models on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            models: Dictionary of trained models (if None, use self.models)
            
        Returns:
            Dictionary of model names and their evaluation metrics
        """
        if models is None:
            models = self.models
            
        if not models:
            logger.warning("No models to evaluate")
            return {}
            
        evaluation_results = {}
        best_score = -np.inf
        best_model_name = None
        
        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                
                metrics = {}
                
                if self.target_type == 'regression':
                    # Regression metrics
                    metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
                    metrics['mae'] = mean_absolute_error(y_test, y_pred)
                    metrics['r2'] = r2_score(y_test, y_pred)
                    
                    logger.info(f"{name} model evaluation:")
                    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
                    logger.info(f"  MAE: {metrics['mae']:.4f}")
                    logger.info(f"  R^2: {metrics['r2']:.4f}")
                    
                    # Track best model (by R^2)
                    if metrics['r2'] > best_score:
                        best_score = metrics['r2']
                        best_model_name = name
                        
                else:  # classification
                    # Convert probabilities to binary predictions
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_test)[:, 1]
                        y_pred_binary = (y_prob >= 0.5).astype(int)
                    else:
                        y_pred_binary = y_pred
                        y_prob = y_pred
                    
                    # Classification metrics
                    metrics['accuracy'] = accuracy_score(y_test, y_pred_binary)
                    metrics['precision'] = precision_score(y_test, y_pred_binary, zero_division=0)
                    metrics['recall'] = recall_score(y_test, y_pred_binary, zero_division=0)
                    metrics['f1'] = f1_score(y_test, y_pred_binary, zero_division=0)
                    
                    try:
                        metrics['auc'] = roc_auc_score(y_test, y_prob)
                    except:
                        metrics['auc'] = 0.5  # Default for failed AUC
                    
                    logger.info(f"{name} model evaluation:")
                    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                    logger.info(f"  Precision: {metrics['precision']:.4f}")
                    logger.info(f"  Recall: {metrics['recall']:.4f}")
                    logger.info(f"  F1: {metrics['f1']:.4f}")
                    logger.info(f"  AUC: {metrics['auc']:.4f}")
                    
                    # Track best model (by AUC)
                    if metrics['auc'] > best_score:
                        best_score = metrics['auc']
                        best_model_name = name
                
                evaluation_results[name] = metrics
                
            except Exception as e:
                logger.error(f"Error evaluating {name} model: {e}")
        
        # Store the best model
        if best_model_name:
            self.best_model = models[best_model_name]
            self.best_model_name = best_model_name
            logger.info(f"Best model: {best_model_name}")
        
        return evaluation_results
    
    def get_feature_importance(self, model=None, top_n: int = 20) -> pd.DataFrame:
        """
        Extract feature importance from the model.
        
        Args:
            model: Model to extract feature importance from (if None, use best_model)
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if model is None:
            if self.best_model is None:
                logger.warning("No best model available for feature importance")
                return pd.DataFrame()
            model = self.best_model
            
        try:
            # Different models store feature importance differently
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
                if len(importance.shape) > 1 and importance.shape[0] == 1:
                    importance = importance[0]
            else:
                logger.warning("Model doesn't have feature importance attribute")
                return pd.DataFrame()
                
            # Create DataFrame with feature names and importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': importance
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            # Get top N features
            if top_n is not None and top_n < len(feature_importance):
                feature_importance = feature_importance.head(top_n)
                
            self.feature_importance = feature_importance
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error extracting feature importance: {e}")
            return pd.DataFrame()
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to display
            figsize: Figure size
        """
        feature_importance = self.get_feature_importance(top_n=top_n)
        
        if feature_importance.empty:
            logger.warning("No feature importance available to plot")
            return
            
        plt.figure(figsize=figsize)
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title(f"Top {len(feature_importance)} Feature Importance")
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, 
                        X_test: pd.DataFrame, 
                        y_test: pd.Series,
                        model=None,
                        figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot actual vs predicted values.
        
        Args:
            X_test: Test features
            y_test: Test target
            model: Model to use for predictions (if None, use best_model)
            figsize: Figure size
        """
        if model is None:
            if self.best_model is None:
                logger.warning("No best model available for predictions")
                return
            model = self.best_model
            
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Create plot
            plt.figure(figsize=figsize)
            
            # Plot actual values
            plt.plot(y_test.index, y_test.values, label='Actual', color='blue')
            
            # Plot predicted values
            plt.plot(y_test.index, y_pred, label='Predicted', color='red', alpha=0.7)
            
            plt.title('Actual vs Predicted Values')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            if self.target_type == 'regression':
                # Scatter plot for regression
                plt.figure(figsize=(8, 8))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.title('Actual vs Predicted Scatter Plot')
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting predictions: {e}")
    
    def plot_confusion_matrix(self, 
                             X_test: pd.DataFrame, 
                             y_test: pd.Series,
                             model=None,
                             figsize: Tuple[int, int] = (8, 6)) -> None:
        """
        Plot confusion matrix for classification models.
        
        Args:
            X_test: Test features
            y_test: Test target
            model: Model to use for predictions (if None, use best_model)
            figsize: Figure size
        """
        if self.target_type != 'classification':
            logger.warning("Confusion matrix is only applicable to classification problems")
            return
            
        if model is None:
            if self.best_model is None:
                logger.warning("No best model available for confusion matrix")
                return
            model = self.best_model
            
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot
            plt.figure(figsize=figsize)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.show()
            
            # Print classification report
            print(classification_report(y_test, y_pred))
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
    
    def save_model(self, model=None, filepath: str = 'best_model.joblib') -> None:
        """
        Save model to file.
        
        Args:
            model: Model to save (if None, use best_model)
            filepath: Path where model will be saved
        """
        if model is None:
            if self.best_model is None:
                logger.warning("No best model available to save")
                return
            model = self.best_model
            
        try:
            dump(model, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str) -> Any:
        """
        Load model from file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        try:
            model = load(filepath)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def full_training_pipeline(self,
                              data: pd.DataFrame,
                              target_col: str,
                              feature_cols: List[str] = None,
                              tune_hyperparams: bool = True,
                              scale_features: bool = True,
                              save_model_path: str = None) -> Dict[str, Dict[str, float]]:
        """
        Run the full model training pipeline.
        
        Args:
            data: DataFrame with features and target
            target_col: Column name of the target variable
            feature_cols: List of feature column names (if None, use all except target)
            tune_hyperparams: Whether to tune hyperparameters
            scale_features: Whether to scale features
            save_model_path: Path to save the best model (if None, don't save)
            
        Returns:
            Dictionary of evaluation results
        """
        try:
            logger.info("Starting full model training pipeline")
            
            # 1. Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(
                data=data,
                target_col=target_col,
                feature_cols=feature_cols,
                scale_features=scale_features
            )
            
            # 2. Train initial models
            self.train_models(X_train, y_train)
            
            # 3. Tune hyperparameters if requested
            if tune_hyperparams:
                self.tune_hyperparameters(X_train, y_train)
            
            # 4. Evaluate models
            evaluation_results = self.evaluate_models(X_test, y_test)
            
            # 5. Get feature importance
            if self.best_model is not None:
                feature_importance = self.get_feature_importance()
                if not feature_importance.empty:
                    logger.info("Top 10 important features:")
                    for _, row in feature_importance.head(10).iterrows():
                        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            # 6. Save best model if requested
            if save_model_path is not None and self.best_model is not None:
                self.save_model(filepath=save_model_path)
            
            logger.info("Model training pipeline completed successfully")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            return {}


class ModelEnsembler:
    """
    Class for creating ensemble models from multiple base models.
    
    This class implements various ensembling techniques including
    simple averaging, weighted averaging, and stacking.
    """
    
    def __init__(self, 
                 target_type: str = 'regression',
                 ensemble_method: str = 'weighted',
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize the model ensembler.
        
        Args:
            target_type: Type of prediction task ('regression' or 'classification')
            ensemble_method: Ensembling method ('simple', 'weighted', or 'stacked')
            cv_folds: Number of folds for cross-validation in stacking
            random_state: Random seed for reproducibility
        """
        self.target_type = target_type
        self.ensemble_method = ensemble_method
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.base_models = {}
        self.weights = None
        self.meta_model = None
        
        logger.info(f"Initialized ModelEnsembler with {ensemble_method} method")
    
    def add_model(self, name: str, model: Any) -> None:
        """
        Add a trained model to the ensemble.
        
        Args:
            name: Name of the model
            model: Trained model instance
        """
        self.base_models[name] = model
        logger.info(f"Added {name} model to ensemble")
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set weights for weighted averaging ensemble.
        
        Args:
            weights: Dictionary with model names as keys and weights as values
        """
        # Normalize weights to sum to 1
        total = sum(weights.values())
        self.weights = {name: weight / total for name, weight in weights.items()}
        logger.info(f"Set ensemble weights: {self.weights}")
    
    def fit_ensemble(self, 
                    X_train: pd.DataFrame, 
                    y_train: pd.Series,
                    meta_model=None) -> None:
        """
        Fit the ensemble model.
        
        For simple and weighted averaging, this just sets up the weights.
        For stacking, this trains the meta-model.
        
        Args:
            X_train: Training features
            y_train: Training target
            meta_model: Model to use as meta-learner (if None, use default)
        """
        if not self.base_models:
            logger.error("No base models provided for ensemble")
            return
            
        if self.ensemble_method == 'simple':
            # Equal weights for all models
            weights = {name: 1.0 / len(self.base_models) for name in self.base_models}
            self.set_weights(weights)
            logger.info("Simple averaging ensemble ready")
            
        elif self.ensemble_method == 'weighted':
            if self.weights is None:
                # Calculate weights based on cross-validation performance
                weights = {}
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                
                for name, model in self.base_models.items():
                    try:
                        if self.target_type == 'regression':
                            scores = cross_val_score(
                                model, X_train, y_train, 
                                cv=tscv, 
                                scoring='neg_mean_squared_error'
                            )
                            # Higher is better for neg_mse, but we want lower error
                            weights[name] = -np.mean(scores)
                        else:  # classification
                            scores = cross_val_score(
                                model, X_train, y_train, 
                                cv=tscv, 
                                scoring='roc_auc'
                            )
                            weights[name] = np.mean(scores)
                            
                    except Exception as e:
                        logger.error(f"Error calculating weight for {name} model: {e}")
                        weights[name] = 0.0
                        
                # Invert weights for regression (lower MSE is better)
                if self.target_type == 'regression' and any(weights.values()):
                    weights = {name: 1.0 / (w + 1e-10) for name, w in weights.items()}
                
                # Normalize weights
                if sum(weights.values()) > 0:
                    self.set_weights(weights)
                else:
                    # Fall back to simple averaging if all weights are 0
                    weights = {name: 1.0 / len(self.base_models) for name in self.base_models}
                    self.set_weights(weights)
                    
                logger.info("Weighted ensemble weights calculated")
                
        elif self.ensemble_method == 'stacked':
            # Train meta-model on base model predictions
            if meta_model is None:
                if self.target_type == 'regression':
                    meta_model = Ridge(random_state=self.random_state)
                else:  # classification
                    meta_model = LogisticRegression(random_state=self.random_state)
            
            # Generate predictions from base models using cross-validation
            base_predictions = np.zeros((X_train.shape[0], len(self.base_models)))
            
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            
            for i, (name, model) in enumerate(self.base_models.items()):
                try:
                    logger.info(f"Generating cross-validated predictions for {name}")
                    
                    # Use cross-validation to get out-of-fold predictions
                    preds = np.zeros(X_train.shape[0])
                    
                    for train_idx, val_idx in tscv.split(X_train):
                        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_fold_train = y_train.iloc[train_idx]
                        
                        # Fit on fold training data
                        model.fit(X_fold_train, y_fold_train)
                        
                        # Predict on fold validation data
                        if self.target_type == 'classification' and hasattr(model, 'predict_proba'):
                            fold_preds = model.predict_proba(X_fold_val)[:, 1]
                        else:
                            fold_preds = model.predict(X_fold_val)
                            
                        preds[val_idx] = fold_preds
                    
                    base_predictions[:, i] = preds
                    
                except Exception as e:
                    logger.error(f"Error generating stacking features for {name}: {e}")
                    # Fill with zeros if failed
                    base_predictions[:, i] = 0
            
            # Train meta-model on base model predictions
            logger.info("Training meta-model for stacked ensemble")
            self.meta_model = meta_model
            self.meta_model.fit(base_predictions, y_train)
            
            # Refit base models on the full training data
            for name, model in self.base_models.items():
                try:
                    model.fit(X_train, y_train)
                except Exception as e:
                    logger.error(f"Error refitting {name} model: {e}")
            
            logger.info("Stacked ensemble trained successfully")
            
        else:
            logger.error(f"Unknown ensemble method: {self.ensemble_method}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions from the ensemble model.
        
        Args:
            X: Feature data
            
        Returns:
            Array of predictions
        """
        if not self.base_models:
            logger.error("No base models available for prediction")
            return np.zeros(X.shape[0])
            
        try:
            if self.ensemble_method in ['simple', 'weighted']:
                # Get predictions from all base models
                predictions = {}
                for name, model in self.base_models.items():
                    if self.target_type == 'classification' and hasattr(model, 'predict_proba'):
                        predictions[name] = model.predict_proba(X)[:, 1]
                    else:
                        predictions[name] = model.predict(X)
                
                # Combine predictions using weights
                weights = self.weights if self.weights else {name: 1.0 / len(self.base_models) for name in self.base_models}
                
                # Weighted average
                ensemble_pred = np.zeros(X.shape[0])
                for name, pred in predictions.items():
                    if name in weights:
                        ensemble_pred += weights[name] * pred
                
                return ensemble_pred
                
            elif self.ensemble_method == 'stacked':
                if self.meta_model is None:
                    logger.error("Meta-model not trained for stacked ensemble")
                    return np.zeros(X.shape[0])
                
                # Get predictions from all base models
                base_predictions = np.zeros((X.shape[0], len(self.base_models)))
                
                for i, (name, model) in enumerate(self.base_models.items()):
                    try:
                        if self.target_type == 'classification' and hasattr(model, 'predict_proba'):
                            base_predictions[:, i] = model.predict_proba(X)[:, 1]
                        else:
                            base_predictions[:, i] = model.predict(X)
                    except Exception as e:
                        logger.error(f"Error getting predictions from {name}: {e}")
                        # Fill with zeros if failed
                        base_predictions[:, i] = 0
                
                # Use meta-model to make final predictions
                return self.meta_model.predict(base_predictions)
                
            else:
                logger.error(f"Unknown ensemble method: {self.ensemble_method}")
                return np.zeros(X.shape[0])
                
        except Exception as e:
            logger.error(f"Error generating ensemble predictions: {e}")
            return np.zeros(X.shape[0])