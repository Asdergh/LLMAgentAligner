import numpy as np
import scipy.stats as st
import plotly.express as px
import plotly.graph_objects as go
import torch.optim as optim
from plotly.subplots import make_subplots
from inspect import signature

def extract_arguments(instance: object, kwargs: dict):
    sig = signature(instance.__init__)
    valid_arguments = {}
    for (key, _) in sig.parameters.items():
        if key in kwargs:
            valid_arguments[key] = kwargs[key]
    return valid_arguments

def create_optimizer(optimizer_name, model_parameters, lr, **kwargs):
    optimizers = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
        'adamax': optim.Adamax,
    }
    
    optimizer_name = optimizer_name.lower()
    if optimizer_name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    optimizer_class = optimizers[optimizer_name]
    valid_arguments = extract_arguments(optimizer_class, kwargs)
    return optimizer_class(model_parameters, lr=lr, **valid_arguments)


def plot_q_q(y_target: np.ndarray, 
            y_model: np.ndarray, 
            plot: bool=False):

    n = y_target.shape[0]
    e = y_target - y_model
    
    probs = (np.arange(1, n + 1) - 0.5) / n
    quantile_theoretical = st.norm.ppf(probs)
    line = np.std(e) * quantile_theoretical + np.mean(e)

    quantile_e = np.sort(e)
    diffs = (quantile_e - quantile_theoretical)
    diffs /= np.abs(diffs).max()
    if plot:
        fig = go.Figure(data=[
            go.Scatter(quantile_theoretical, line, 
                        mode="line",
                        marker=dict(color="orange")),
            go.Scatter(quantile_theoretical, quantile_e,
                        marker=dict(color=diffs,
                                    cmas=diffs.max(), cmin=diffs.min(),
                                    colorscale="PyiG"))
        ])
        fig.show()
    else:
        return {"line": line, 
                "quantile_e": quantile_e, 
                "quantile_theo": quantile_theoretical}


def regression_val_plots(y_target: np.ndarray, y_model: np.ndarray):
    pass