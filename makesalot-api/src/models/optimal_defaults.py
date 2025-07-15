"""
Optimal model parameters discovered through hyperparameter optimization.
"""

OPTIMAL_PARAMETERS = {'lstm': {'lstm_units': [128, 64], 'dropout_rate': 0.3, 'learning_rate': 0.001, 'batch_size': 16}, 'cnn': {'filters': [64, 32, 16], 'kernel_sizes': [5, 3, 3], 'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 16}}

def get_optimal_params(model_type):
    """Get optimal parameters for a model type."""
    return OPTIMAL_PARAMETERS.get(model_type, {})
