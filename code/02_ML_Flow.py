#!/usr/bin/env python
# coding: utf-8

# # SARIMA From Scratch with MLflow
# 
# Monthly sales forecasting using custom SARIMA implementation (no statsmodels).
# 
# **Key Features:**
# - Log transform with shift constant (no Z-score normalization)
# - Explicit double differencing: Seasonal(12) → Regular(1)
# - Correct inverse differencing for forecast reconstruction
# - Joint AR+MA training with shared residuals
# - Baseline comparisons (seasonal naive)
# - Complete MLflow state persistence for reproducible inference

# ## 1. Setup and Configuration

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import os

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

warnings.filterwarnings('ignore')


# In[24]:


DATA_PATH = "../dataset/dataset_monthly_cust_l3_sarima.csv"
EXPERIMENT_NAME = "SARIMA_Refactored_v3"
ARTIFACTS_DIR = "./artifacts"

SARIMA_P = 1
SARIMA_D = 1
SARIMA_Q = 1
SARIMA_CAP_P = 0
SARIMA_CAP_D = 1
SARIMA_CAP_Q = 0
SARIMA_S = 12

train_end_date = pd.Timestamp('2025-01-01')
test_end_date = pd.Timestamp('2025-10-01')

ITERATIONS = 10_000
LEARNING_RATE = 0.01
L2_REGULARIZATION = 0.001

INITIAL_TRAIN_SIZE = 15
HORIZON = 12

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

mlflow.end_run()
try:
    mlflow.autolog(disable=True)
except:
    pass
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow experiment: {EXPERIMENT_NAME}")


# ## 2. Transformation Functions
# 
# Log transform with shift constant and double differencing for stationarity.

# In[ ]:


def compute_log_shift(y):
    min_val = np.min(y)
    if min_val > 0:
        return 0.0
    return 1.0 - min_val


def log_transform(y, shift_constant):
    return np.log(y + shift_constant)


def inverse_log_transform(z, shift_constant):
    return np.exp(z) - shift_constant


# In[ ]:


def double_difference(z, seasonal_period=12):
    seasonal_diff = z[seasonal_period:] - z[:-seasonal_period]
    w = seasonal_diff[1:] - seasonal_diff[:-1]
    return w, seasonal_diff


def inverse_double_difference(w_forecast, log_history, seasonal_period=12):
    z_extended = list(log_history)
    
    for w in w_forecast:
        z_new = w + z_extended[-1] + z_extended[-seasonal_period] - z_extended[-seasonal_period - 1]
        z_extended.append(z_new)
    
    return np.array(z_extended[len(log_history):])


# ## 3. Full SARIMA Functions
# 
# Multiplicative SARIMA with non-seasonal and seasonal AR/MA terms.
# SARIMA(p,d,q)(P,D,Q)[s] on doubly differenced series w_t:
# (1 - φ₁B - ... - φₚBᵖ)(1 - Φ₁Bˢ - ... - ΦₚBᴾˢ)w_t = 
# (1 + θ₁B + ... + θqBᵍ)(1 + Θ₁Bˢ + ... + ΘQBᴽˢ)ε_t

# In[ ]:


def build_sarima_lag_structure(p, q, cap_p, cap_q, s):
    ar_lags = list(range(1, p + 1))
    seasonal_ar_lags = [s * i for i in range(1, cap_p + 1)]
    interaction_ar_lags = []
    for non_seas_lag in ar_lags:
        for seas_lag in seasonal_ar_lags:
            interaction_ar_lags.append(non_seas_lag + seas_lag)
    
    ma_lags = list(range(1, q + 1))
    seasonal_ma_lags = [s * i for i in range(1, cap_q + 1)]
    interaction_ma_lags = []
    for non_seas_lag in ma_lags:
        for seas_lag in seasonal_ma_lags:
            interaction_ma_lags.append(non_seas_lag + seas_lag)
    
    all_ar_lags = sorted(set(ar_lags + seasonal_ar_lags + interaction_ar_lags))
    all_ma_lags = sorted(set(ma_lags + seasonal_ma_lags + interaction_ma_lags))
    
    return {
        'ar_lags': ar_lags,
        'seasonal_ar_lags': seasonal_ar_lags,
        'interaction_ar_lags': interaction_ar_lags,
        'ma_lags': ma_lags,
        'seasonal_ma_lags': seasonal_ma_lags,
        'interaction_ma_lags': interaction_ma_lags,
        'all_ar_lags': all_ar_lags,
        'all_ma_lags': all_ma_lags,
        'max_ar_lag': max(all_ar_lags) if all_ar_lags else 0,
        'max_ma_lag': max(all_ma_lags) if all_ma_lags else 0
    }


def sarima_predict_step(data_history, errors_history, lag_structure, 
                        ar_coeffs, seasonal_ar_coeffs, interaction_ar_coeffs,
                        ma_coeffs, seasonal_ma_coeffs, interaction_ma_coeffs):
    prediction = 0.0
    
    for i, lag in enumerate(lag_structure['ar_lags']):
        if lag <= len(data_history) and i < len(ar_coeffs):
            prediction += ar_coeffs[i] * data_history[-lag]
    
    for i, lag in enumerate(lag_structure['seasonal_ar_lags']):
        if lag <= len(data_history) and i < len(seasonal_ar_coeffs):
            prediction += seasonal_ar_coeffs[i] * data_history[-lag]
    
    for i, lag in enumerate(lag_structure['interaction_ar_lags']):
        if lag <= len(data_history) and i < len(interaction_ar_coeffs):
            prediction += interaction_ar_coeffs[i] * data_history[-lag]
    
    for i, lag in enumerate(lag_structure['ma_lags']):
        if lag <= len(errors_history) and i < len(ma_coeffs):
            prediction += ma_coeffs[i] * errors_history[-lag]
    
    for i, lag in enumerate(lag_structure['seasonal_ma_lags']):
        if lag <= len(errors_history) and i < len(seasonal_ma_coeffs):
            prediction += seasonal_ma_coeffs[i] * errors_history[-lag]
    
    for i, lag in enumerate(lag_structure['interaction_ma_lags']):
        if lag <= len(errors_history) and i < len(interaction_ma_coeffs):
            prediction += interaction_ma_coeffs[i] * errors_history[-lag]
    
    return prediction


# In[ ]:


def sarima_fit_full(data, p, q, cap_p, cap_q, s, iterations, learning_rate, l2_reg=0.01):
    n = len(data)
    lag_structure = build_sarima_lag_structure(p, q, cap_p, cap_q, s)
    
    ar_coeffs = np.zeros(p)
    seasonal_ar_coeffs = np.zeros(cap_p)
    interaction_ar_coeffs = np.zeros(len(lag_structure['interaction_ar_lags']))
    
    ma_coeffs = np.zeros(q)
    seasonal_ma_coeffs = np.zeros(cap_q)
    interaction_ma_coeffs = np.zeros(len(lag_structure['interaction_ma_lags']))
    
    max_lag = max(lag_structure['max_ar_lag'], lag_structure['max_ma_lag'], 1)
    start_t = max_lag
    
    if start_t >= n:
        return {
            'ar_coeffs': ar_coeffs.tolist(),
            'seasonal_ar_coeffs': seasonal_ar_coeffs.tolist(),
            'interaction_ar_coeffs': interaction_ar_coeffs.tolist(),
            'ma_coeffs': ma_coeffs.tolist(),
            'seasonal_ma_coeffs': seasonal_ma_coeffs.tolist(),
            'interaction_ma_coeffs': interaction_ma_coeffs.tolist(),
            'lag_structure': lag_structure,
            'residual_tail': [0.0] * max(lag_structure['max_ma_lag'], 1)
        }
    
    errors_padded = [0.0] * max_lag
    
    for iteration in range(iterations):
        errors_list = list(errors_padded)
        
        for t in range(start_t, n):
            data_slice = list(data[:t])
            
            pred = sarima_predict_step(
                data_slice, errors_list, lag_structure,
                ar_coeffs, seasonal_ar_coeffs, interaction_ar_coeffs,
                ma_coeffs, seasonal_ma_coeffs, interaction_ma_coeffs
            )
            
            error = data[t] - pred
            
            for i, lag in enumerate(lag_structure['ar_lags']):
                if lag <= len(data_slice) and i < len(ar_coeffs):
                    grad = learning_rate * (error * data_slice[-lag] - l2_reg * ar_coeffs[i])
                    ar_coeffs[i] += grad
            
            for i, lag in enumerate(lag_structure['seasonal_ar_lags']):
                if lag <= len(data_slice) and i < len(seasonal_ar_coeffs):
                    grad = learning_rate * (error * data_slice[-lag] - l2_reg * seasonal_ar_coeffs[i])
                    seasonal_ar_coeffs[i] += grad
            
            for i, lag in enumerate(lag_structure['interaction_ar_lags']):
                if lag <= len(data_slice) and i < len(interaction_ar_coeffs):
                    grad = learning_rate * (error * data_slice[-lag] - l2_reg * interaction_ar_coeffs[i])
                    interaction_ar_coeffs[i] += grad
            
            for i, lag in enumerate(lag_structure['ma_lags']):
                if lag <= len(errors_list) and i < len(ma_coeffs):
                    grad = learning_rate * (error * errors_list[-lag] - l2_reg * ma_coeffs[i])
                    ma_coeffs[i] += grad
            
            for i, lag in enumerate(lag_structure['seasonal_ma_lags']):
                if lag <= len(errors_list) and i < len(seasonal_ma_coeffs):
                    grad = learning_rate * (error * errors_list[-lag] - l2_reg * seasonal_ma_coeffs[i])
                    seasonal_ma_coeffs[i] += grad
            
            for i, lag in enumerate(lag_structure['interaction_ma_lags']):
                if lag <= len(errors_list) and i < len(interaction_ma_coeffs):
                    grad = learning_rate * (error * errors_list[-lag] - l2_reg * interaction_ma_coeffs[i])
                    interaction_ma_coeffs[i] += grad
            
            errors_list.append(error)
        
        ar_coeffs = np.clip(ar_coeffs, -0.95, 0.95)
        seasonal_ar_coeffs = np.clip(seasonal_ar_coeffs, -0.95, 0.95)
        interaction_ar_coeffs = np.clip(interaction_ar_coeffs, -0.95, 0.95)
        ma_coeffs = np.clip(ma_coeffs, -0.95, 0.95)
        seasonal_ma_coeffs = np.clip(seasonal_ma_coeffs, -0.95, 0.95)
        interaction_ma_coeffs = np.clip(interaction_ma_coeffs, -0.95, 0.95)
        
        errors_padded = errors_list[-max_lag:] if max_lag > 0 else [0.0]
    
    residual_tail_len = max(lag_structure['max_ma_lag'], 1)
    residual_tail = errors_list[-residual_tail_len:] if len(errors_list) >= residual_tail_len else errors_list
    
    return {
        'ar_coeffs': ar_coeffs.tolist(),
        'seasonal_ar_coeffs': seasonal_ar_coeffs.tolist(),
        'interaction_ar_coeffs': interaction_ar_coeffs.tolist(),
        'ma_coeffs': ma_coeffs.tolist(),
        'seasonal_ma_coeffs': seasonal_ma_coeffs.tolist(),
        'interaction_ma_coeffs': interaction_ma_coeffs.tolist(),
        'lag_structure': lag_structure,
        'residual_tail': residual_tail
    }


# ## 4. SARIMA Fit and Forecast Functions

# In[ ]:


def sarima_fit(data, p, d, q, P, D, Q, s, iterations, learning_rate):
    shift_constant = compute_log_shift(data)
    log_data = log_transform(data, shift_constant)
    
    w, seasonal_diff = double_difference(log_data, s)
    
    lag_structure = build_sarima_lag_structure(p, q, P, Q, s)
    max_lag = max(lag_structure['max_ar_lag'], lag_structure['max_ma_lag'], 1)
    min_required = max_lag + 1
    
    if len(w) < min_required:
        raise ValueError(f"Not enough data after differencing. Need at least {min_required}, got {len(w)}")
    
    fit_result = sarima_fit_full(w, p, q, P, Q, s, iterations, learning_rate, L2_REGULARIZATION)
    
    start_t = max_lag
    in_sample_predictions = []
    errors_list = [0.0] * max_lag
    
    for t in range(start_t, len(w)):
        data_slice = list(w[:t])
        pred = sarima_predict_step(
            data_slice, errors_list, fit_result['lag_structure'],
            fit_result['ar_coeffs'], fit_result['seasonal_ar_coeffs'], fit_result['interaction_ar_coeffs'],
            fit_result['ma_coeffs'], fit_result['seasonal_ma_coeffs'], fit_result['interaction_ma_coeffs']
        )
        in_sample_predictions.append(pred)
        error = w[t] - pred
        errors_list.append(error)
    
    diff_tail_len = max(max_lag, s + 1)
    
    return {
        'shift_constant': shift_constant,
        'ar_coeffs': fit_result['ar_coeffs'],
        'seasonal_ar_coeffs': fit_result['seasonal_ar_coeffs'],
        'interaction_ar_coeffs': fit_result['interaction_ar_coeffs'],
        'ma_coeffs': fit_result['ma_coeffs'],
        'seasonal_ma_coeffs': fit_result['seasonal_ma_coeffs'],
        'interaction_ma_coeffs': fit_result['interaction_ma_coeffs'],
        'lag_structure': fit_result['lag_structure'],
        'log_history': list(log_data[-(s + 1):]),
        'diff_tail': list(w[-diff_tail_len:]),
        'residual_tail': fit_result['residual_tail'],
        'in_sample_predictions': np.array(in_sample_predictions),
        'w': w,
        'log_data': log_data,
        'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q, 's': s
    }


# In[ ]:


def sarima_forecast(model_state, num_steps):
    s = model_state['s']
    log_history = model_state['log_history']
    diff_tail = list(model_state['diff_tail'])
    residual_tail = list(model_state['residual_tail'])
    shift_constant = model_state['shift_constant']
    lag_structure = model_state['lag_structure']
    
    ar_coeffs = model_state['ar_coeffs']
    seasonal_ar_coeffs = model_state['seasonal_ar_coeffs']
    interaction_ar_coeffs = model_state['interaction_ar_coeffs']
    ma_coeffs = model_state['ma_coeffs']
    seasonal_ma_coeffs = model_state['seasonal_ma_coeffs']
    interaction_ma_coeffs = model_state['interaction_ma_coeffs']
    
    max_ma_lag = lag_structure['max_ma_lag'] if lag_structure['max_ma_lag'] > 0 else 1
    
    w_forecast = []
    
    for step in range(num_steps):
        w_pred = sarima_predict_step(
            diff_tail, residual_tail, lag_structure,
            ar_coeffs, seasonal_ar_coeffs, interaction_ar_coeffs,
            ma_coeffs, seasonal_ma_coeffs, interaction_ma_coeffs
        )
        
        w_forecast.append(w_pred)
        diff_tail.append(w_pred)
        
        residual_tail.append(0.0)
        if len(residual_tail) > max_ma_lag:
            residual_tail.pop(0)
    
    z_forecast = inverse_double_difference(w_forecast, log_history, s)
    y_forecast = inverse_log_transform(z_forecast, shift_constant)
    y_forecast = np.maximum(y_forecast, 0)
    
    return y_forecast


# ## 5. Baseline Forecast Functions

# In[ ]:


def seasonal_naive_forecast(train_data, num_steps, seasonal_period=12):
    forecasts = []
    for i in range(num_steps):
        idx = len(train_data) - seasonal_period + (i % seasonal_period)
        if idx < 0:
            idx = len(train_data) - 1
        forecasts.append(train_data[idx])
    return np.array(forecasts)


def last_value_forecast(train_data, num_steps):
    return np.full(num_steps, train_data[-1])


# ## 6. SARIMA Model Class (MLflow PythonModel)

# In[ ]:


class SARIMAModel(mlflow.pyfunc.PythonModel):
    def __init__(self, ar_order=1, diff_order=1, ma_order=1, 
                 seasonal_ar_order=0, seasonal_diff_order=1, seasonal_ma_order=0, 
                 seasonal_period=12, iterations=500, learning_rate=0.01):
        self.ar_order = ar_order
        self.diff_order = diff_order
        self.ma_order = ma_order
        self.seasonal_ar_order = seasonal_ar_order
        self.seasonal_diff_order = seasonal_diff_order
        self.seasonal_ma_order = seasonal_ma_order
        self.seasonal_period = seasonal_period
        self.iterations = iterations
        self.learning_rate = learning_rate
        
        self.shift_constant = None
        self.ar_coeffs = None
        self.seasonal_ar_coeffs = None
        self.interaction_ar_coeffs = None
        self.ma_coeffs = None
        self.seasonal_ma_coeffs = None
        self.interaction_ma_coeffs = None
        self.lag_structure = None
        self.log_history = None
        self.diff_tail = None
        self.residual_tail = None
        self.in_sample_predictions = None
        self.w = None
        self.log_data = None
    
    def fit(self, data):
        result = sarima_fit(
            data, self.ar_order, self.diff_order, self.ma_order, 
            self.seasonal_ar_order, self.seasonal_diff_order, self.seasonal_ma_order, 
            self.seasonal_period,
            self.iterations, self.learning_rate
        )
        
        self.shift_constant = result['shift_constant']
        self.ar_coeffs = result['ar_coeffs']
        self.seasonal_ar_coeffs = result['seasonal_ar_coeffs']
        self.interaction_ar_coeffs = result['interaction_ar_coeffs']
        self.ma_coeffs = result['ma_coeffs']
        self.seasonal_ma_coeffs = result['seasonal_ma_coeffs']
        self.interaction_ma_coeffs = result['interaction_ma_coeffs']
        self.lag_structure = result['lag_structure']
        self.log_history = result['log_history']
        self.diff_tail = result['diff_tail']
        self.residual_tail = result['residual_tail']
        self.in_sample_predictions = result['in_sample_predictions']
        self.w = result['w']
        self.log_data = result['log_data']
        
        return self
    
    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            num_steps = int(model_input.iloc[0, 0])
        else:
            num_steps = int(model_input)
        
        model_state = {
            'p': self.ar_order,
            'q': self.ma_order,
            's': self.seasonal_period,
            'ar_coeffs': self.ar_coeffs,
            'seasonal_ar_coeffs': self.seasonal_ar_coeffs,
            'interaction_ar_coeffs': self.interaction_ar_coeffs,
            'ma_coeffs': self.ma_coeffs,
            'seasonal_ma_coeffs': self.seasonal_ma_coeffs,
            'interaction_ma_coeffs': self.interaction_ma_coeffs,
            'lag_structure': self.lag_structure,
            'log_history': self.log_history,
            'diff_tail': self.diff_tail,
            'residual_tail': self.residual_tail,
            'shift_constant': self.shift_constant
        }
        
        return sarima_forecast(model_state, num_steps)
    
    def get_residuals(self):
        if self.w is None or self.in_sample_predictions is None:
            return None
        
        if self.lag_structure is not None:
            max_lag = max(self.lag_structure['max_ar_lag'], self.lag_structure['max_ma_lag'], 1)
        else:
            max_lag = max(self.ar_order, self.ma_order)
            if max_lag == 0:
                max_lag = 1
        
        actual = self.w[max_lag:]
        predicted = self.in_sample_predictions
        
        min_len = min(len(actual), len(predicted))
        return actual[:min_len] - predicted[:min_len]
    
    def get_model_state(self):
        return {
            'p': self.ar_order, 'd': self.diff_order, 'q': self.ma_order,
            'P': self.seasonal_ar_order, 'D': self.seasonal_diff_order, 'Q': self.seasonal_ma_order,
            's': self.seasonal_period,
            'iterations': self.iterations, 'learning_rate': self.learning_rate,
            'shift_constant': self.shift_constant,
            'ar_coeffs': self.ar_coeffs, 
            'seasonal_ar_coeffs': self.seasonal_ar_coeffs,
            'interaction_ar_coeffs': self.interaction_ar_coeffs,
            'ma_coeffs': self.ma_coeffs,
            'seasonal_ma_coeffs': self.seasonal_ma_coeffs,
            'interaction_ma_coeffs': self.interaction_ma_coeffs,
            'lag_structure': self.lag_structure,
            'log_history': self.log_history, 'diff_tail': self.diff_tail, 'residual_tail': self.residual_tail
        }


# ## 7. Metrics and Plotting Functions

# In[ ]:


def calculate_metrics(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    n = len(actual)
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    
    mask = actual != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = np.nan
    
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2, 'N': n}


def compute_acf(residuals, max_lag=12):
    n = len(residuals)
    mean = np.mean(residuals)
    var = np.var(residuals)
    
    if var == 0:
        return np.zeros(max_lag + 1)
    
    acf = []
    for k in range(max_lag + 1):
        if k == 0:
            acf.append(1.0)
        else:
            cov = np.sum((residuals[:-k] - mean) * (residuals[k:] - mean)) / n
            acf.append(cov / var)
    
    return np.array(acf)


# In[ ]:


def plot_residuals(residuals, segment, target, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(residuals, 'b-', linewidth=0.8)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[0].set_title(f'Residuals in Double-Differenced Log Space - {segment} {target}')
    axes[0].set_xlabel('Time Index')
    axes[0].set_ylabel('Residual')
    axes[0].grid(True, alpha=0.3)
    
    acf = compute_acf(residuals, max_lag=12)
    lags = np.arange(len(acf))
    
    axes[1].bar(lags, acf, color='steelblue', edgecolor='black')
    n = len(residuals)
    conf_interval = 1.96 / np.sqrt(n)
    axes[1].axhline(y=conf_interval, color='r', linestyle='--', alpha=0.7, label='95% CI')
    axes[1].axhline(y=-conf_interval, color='r', linestyle='--', alpha=0.7)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_title('Residual ACF')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Autocorrelation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_test_vs_forecast(test_dates, test_actual, sarima_forecast_vals, baseline_forecast, 
                          segment, target, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))

    test_actual_units = np.clip(np.rint(np.asarray(test_actual, dtype=float)), 0, None).astype(int)
    sarima_forecast_units = np.clip(np.rint(np.asarray(sarima_forecast_vals, dtype=float)), 0, None).astype(int)
    baseline_units = np.clip(np.rint(np.asarray(baseline_forecast, dtype=float)), 0, None).astype(int)
    
    ax.plot(test_dates, test_actual_units, 'b-o', linewidth=2, markersize=6, label='Actual')
    ax.plot(test_dates, sarima_forecast_units, 'r--s', linewidth=2, markersize=6, label='SARIMA Forecast')
    ax.plot(test_dates, baseline_units, 'g:^', linewidth=2, markersize=6, label='Seasonal Naive')
    
    ax.set_title(f'Test vs Forecast - {segment} {target}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Units')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ## 8. Residual Diagnostics and Model Selection

# In[ ]:


def check_residual_diagnostics(residuals, max_lag=None, lag_structure=None, significance_level=0.05):
    if max_lag is None:
        if lag_structure is not None:
            max_lag = max(24, lag_structure.get('max_ar_lag', 12) + 1, lag_structure.get('max_ma_lag', 12) + 1)
        else:
            max_lag = 24
    
    if residuals is None or len(residuals) < max_lag + 1:
        return False, "Insufficient residuals"
    
    n = len(residuals)
    acf = compute_acf(residuals, max_lag)
    ci = 1.96 / np.sqrt(n)
    
    violations = 0
    violation_lags = []
    for lag in range(1, max_lag + 1):
        if abs(acf[lag]) > ci:
            violations += 1
            violation_lags.append(lag)
    
    max_allowed_violations = max(2, int(max_lag * significance_level * 2))
    passes = violations <= max_allowed_violations
    
    return passes, f"ACF violations at lags {violation_lags}: {violations}/{max_lag} (threshold: {max_allowed_violations})"


def rolling_origin_evaluation(data, p, d, q, P, D, Q, s, 
                               initial_train_size=None, horizon=None,
                               iterations=1000, learning_rate=0.01):
    if initial_train_size is None:
        initial_train_size = INITIAL_TRAIN_SIZE
    if horizon is None:
        horizon = HORIZON
    
    n = len(data)
    min_required_size = initial_train_size + horizon + s + 2
    
    if n < min_required_size:
        return {
            'avg_rmse': np.inf,
            'avg_mape': np.inf,
            'fold_count': 0,
            'successful_folds': 0,
            'stable': False
        }
    
    fold_rmses = []
    fold_mapes = []
    total_folds = 0
    failed_folds = 0
    
    current_train_end = initial_train_size
    
    while current_train_end + horizon <= n:
        train_data = data[:current_train_end]
        test_data = data[current_train_end:current_train_end + horizon]
        total_folds += 1
        
        try:
            model = SARIMAModel(
                ar_order=p, diff_order=d, ma_order=q,
                seasonal_ar_order=P, seasonal_diff_order=D, seasonal_ma_order=Q,
                seasonal_period=s,
                iterations=iterations, learning_rate=learning_rate
            )
            model.fit(train_data)
            predictions = model.predict(None, len(test_data))
            
            metrics = calculate_metrics(test_data, predictions)
            
            if not np.isnan(metrics['RMSE']) and not np.isinf(metrics['RMSE']):
                fold_rmses.append(metrics['RMSE'])
            else:
                fold_rmses.append(np.inf)
                failed_folds += 1
            if not np.isnan(metrics['MAPE']) and not np.isinf(metrics['MAPE']):
                fold_mapes.append(metrics['MAPE'])
                
        except Exception:
            fold_rmses.append(np.inf)
            failed_folds += 1
        
        current_train_end += 1
    
    successful_folds = len([r for r in fold_rmses if r < np.inf])
    min_successful_required = max(1, total_folds // 2)
    
    if successful_folds < min_successful_required:
        return {
            'avg_rmse': np.inf,
            'avg_mape': np.inf,
            'fold_count': total_folds,
            'successful_folds': successful_folds,
            'stable': False
        }
    
    valid_rmses = [r for r in fold_rmses if r < np.inf]
    avg_rmse = np.mean(valid_rmses)
    avg_mape = np.mean(fold_mapes) if fold_mapes else np.inf
    std_rmse = np.std(valid_rmses)
    
    stability_threshold = 2.0
    stable = std_rmse < stability_threshold * avg_rmse
    
    return {
        'avg_rmse': avg_rmse,
        'avg_mape': avg_mape,
        'std_rmse': std_rmse,
        'fold_count': total_folds,
        'successful_folds': successful_folds,
        'stable': stable,
        'fold_rmses': fold_rmses
    }


# In[ ]:


def grid_search_sarima(data, s=12, d=1, D=1,
                       p_range=(0, 1), q_range=(0, 1, 2),
                       P_range=(0, 1), Q_range=(0, 1),
                       initial_train_size=None, horizon=None,
                       iterations=1000, learning_rate=0.01):
    if initial_train_size is None:
        initial_train_size = INITIAL_TRAIN_SIZE
    if horizon is None:
        horizon = HORIZON
    candidates = []
    
    for p in p_range:
        for q in q_range:
            for P in P_range:
                for Q in Q_range:
                    candidates.append((p, d, q, P, D, Q, s))
    
    results = []
    
    for p, d, q, P, D, Q, s in candidates:
        order_str = f"({p},{d},{q})({P},{D},{Q})[{s}]"
        
        try:
            eval_result = rolling_origin_evaluation(
                data, p, d, q, P, D, Q, s,
                initial_train_size=initial_train_size,
                horizon=horizon,
                iterations=iterations,
                learning_rate=learning_rate
            )
            
            if eval_result['fold_count'] == 0:
                continue
            
            try:
                full_model = SARIMAModel(
                    ar_order=p, diff_order=d, ma_order=q,
                    seasonal_ar_order=P, seasonal_diff_order=D, seasonal_ma_order=Q,
                    seasonal_period=s,
                    iterations=iterations, learning_rate=learning_rate
                )
                full_model.fit(data)
                residuals = full_model.get_residuals()
                passes_diag, diag_msg = check_residual_diagnostics(residuals)
            except Exception:
                passes_diag = False
                diag_msg = "Fitting failed"
            
            complexity = p + q + P + Q
            
            results.append({
                'order': order_str,
                'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q, 's': s,
                'avg_rmse': eval_result['avg_rmse'],
                'avg_mape': eval_result['avg_mape'],
                'fold_count': eval_result['fold_count'],
                'stable': eval_result['stable'],
                'passes_diagnostics': passes_diag,
                'diag_message': diag_msg,
                'complexity': complexity
            })
            
        except Exception as e:
            continue
    
    if not results:
        return None, []
    
    passing_results = [r for r in results if r['passes_diagnostics']]
    
    if passing_results:
        search_pool = passing_results
    else:
        search_pool = results
    
    search_pool.sort(key=lambda x: (x['avg_rmse'], x['complexity']))
    
    best = search_pool[0]
    
    if len(search_pool) > 1:
        best_rmse = best['avg_rmse']
        tolerance = 0.05
        similar = [r for r in search_pool if r['avg_rmse'] <= best_rmse * (1 + tolerance)]
        similar.sort(key=lambda x: x['complexity'])
        best = similar[0]
    
    return best, results


# ## 9. Load and Prepare Data

# In[ ]:


monthly_df = pd.read_csv(DATA_PATH)
monthly_df['date'] = pd.to_datetime(monthly_df['ACCOUNTING_PERIOD'], format='%Y%m')
monthly_df = monthly_df.sort_values(['CUST_STEERING_L3_NAME', 'date'])

segments = monthly_df['CUST_STEERING_L3_NAME'].unique().tolist()
print(f"Segments: {segments}")
print(f"Date range: {monthly_df['date'].min()} to {monthly_df['date'].max()}")
monthly_df.head()


# ## 10. Train/Test Split Configuration

# In[ ]:




print(f"Training period: Start - {train_end_date}")
print(f"Test period: {train_end_date} - {test_end_date}")


def split_data(df, segment, target_col, train_end, test_end):
    segment_df = df[df['CUST_STEERING_L3_NAME'] == segment].copy()
    segment_df = segment_df.sort_values('date')
    
    train_df = segment_df[segment_df['date'] <= train_end]
    test_df = segment_df[(segment_df['date'] > train_end) & (segment_df['date'] <= test_end)]
    
    train_data = train_df[target_col].values
    test_data = test_df[target_col].values
    train_dates = train_df['date'].values
    test_dates = test_df['date'].values
    
    return train_data, test_data, train_dates, test_dates


# ## 11. Training Function with MLflow Logging

# In[ ]:


def train_and_log_model(segment, target_col, target_label, 
                        p, d, q, P, D, Q, s,
                        iterations, learning_rate):
    train_data, test_data, train_dates, test_dates = split_data(
        monthly_df, segment, target_col, train_end_date, test_end_date
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{segment}_{target_label}_p{p}d{d}q{q}_P{P}D{D}Q{Q}s{s}_{timestamp}"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("segment", segment)
        mlflow.log_param("target", target_label)
        mlflow.log_param("p", p)
        mlflow.log_param("d", d)
        mlflow.log_param("q", q)
        mlflow.log_param("seasonal_P", P)
        mlflow.log_param("seasonal_D", D)
        mlflow.log_param("seasonal_Q", Q)
        mlflow.log_param("s", s)
        mlflow.log_param("iterations", iterations)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("train_size", len(train_data))
        mlflow.log_param("test_size", len(test_data))
        
        model = SARIMAModel(
            ar_order=p, diff_order=d, ma_order=q,
            seasonal_ar_order=P, seasonal_diff_order=D, seasonal_ma_order=Q,
            seasonal_period=s,
            iterations=iterations,
            learning_rate=learning_rate
        )
        model.fit(train_data)
        
        mlflow.log_param("shift_constant", model.shift_constant)
        mlflow.log_param("ar_coeffs", str(model.ar_coeffs))
        mlflow.log_param("ma_coeffs", str(model.ma_coeffs))
        
        if len(test_data) > 0:
            test_predictions = model.predict(None, len(test_data))
            sarima_metrics = calculate_metrics(test_data, test_predictions)
            
            baseline_preds = seasonal_naive_forecast(train_data, len(test_data), s)
            baseline_metrics = calculate_metrics(test_data, baseline_preds)
        else:
            test_predictions = np.array([])
            sarima_metrics = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'R2': np.nan}
            baseline_preds = np.array([])
            baseline_metrics = {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'R2': np.nan}
        
        for metric_name, metric_value in sarima_metrics.items():
            if metric_name != 'N' and not np.isnan(metric_value):
                mlflow.log_metric(f"sarima_{metric_name}", metric_value)
        
        for metric_name, metric_value in baseline_metrics.items():
            if metric_name != 'N' and not np.isnan(metric_value):
                mlflow.log_metric(f"baseline_seasonal_{metric_name}", metric_value)
        
        if sarima_metrics['RMSE'] < baseline_metrics['RMSE']:
            mlflow.log_metric("beats_baseline", 1)
        else:
            mlflow.log_metric("beats_baseline", 0)
        
        residuals = model.get_residuals()
        if residuals is not None and len(residuals) > 0:
            residuals_path = os.path.join(ARTIFACTS_DIR, f"residuals_{segment}_{target_label}.png")
            plot_residuals(residuals, segment, target_label, residuals_path)
            mlflow.log_artifact(residuals_path)
        
        if len(test_data) > 0:
            test_dates_pd = pd.to_datetime(test_dates)
            forecast_path = os.path.join(ARTIFACTS_DIR, f"forecast_{segment}_{target_label}.png")
            plot_test_vs_forecast(
                test_dates_pd, test_data, test_predictions, baseline_preds,
                segment, target_label, forecast_path
            )
            mlflow.log_artifact(forecast_path)
        
        example = pd.DataFrame({'forecast_periods': [12]})
        signature = infer_signature(example, np.array([0.0] * 12))
        
        mlflow.pyfunc.log_model(
            artifact_path="sarima_model",
            python_model=model,
            input_example=example,
            signature=signature
        )
        
        run_id = mlflow.active_run().info.run_id
    
    return {
        'model': model,
        'sarima_metrics': sarima_metrics,
        'baseline_metrics': baseline_metrics,
        'test_predictions': test_predictions,
        'baseline_predictions': baseline_preds,
        'test_data': test_data,
        'test_dates': test_dates,
        'train_data': train_data,
        'train_dates': train_dates,
        'run_id': run_id
    }


# ## 12. Model Configuration and Training with Grid Search

# In[ ]:




targets = {
    'net_sales_units': 'net_sales',
    'returns_units': 'returns'
}

print(f"Grid Search Configuration:")
print(f"  Fixed: d={SARIMA_D}, D={SARIMA_CAP_D}, s={SARIMA_S}")
print(f"  Search: p in {{0,1}}, q in {{0,1,2}}, P in {{0,1}}, Q in {{0,1}}")
print(f"  Using rolling-origin evaluation with initial window={INITIAL_TRAIN_SIZE}, horizon={HORIZON}")
print(f"Iterations: {ITERATIONS}, Learning Rate: {LEARNING_RATE}")


# In[ ]:


all_results = {}
all_metrics = []
best_orders = {}

for segment in segments:
    print(f"\n{'='*60}")
    print(f"Training: {segment}")
    print('='*60)
    
    all_results[segment] = {}
    best_orders[segment] = {}
    
    for target_col, target_label in targets.items():
        print(f"\n  Target: {target_label}")
        print(f"    Running grid search on training data only...")
        
        train_data, test_data, train_dates, test_dates = split_data(
            monthly_df, segment, target_col, train_end_date, test_end_date
        )
        
        best_order, search_results = grid_search_sarima(
            train_data,
            s=SARIMA_S, d=SARIMA_D, D=SARIMA_CAP_D,
            p_range=(0, 1), q_range=(0, 1, 2),
            P_range=(0, 1), Q_range=(0, 1),
            iterations=ITERATIONS, learning_rate=LEARNING_RATE
        )
        
        if best_order is None:
            n_after_diff = len(train_data) - SARIMA_S - 1
            safe_p = min(SARIMA_P, 1)
            safe_q = min(SARIMA_Q, 2)
            
            if n_after_diff < SARIMA_S + 2:
                safe_P, safe_Q = 0, 0
            else:
                safe_P = SARIMA_CAP_P if n_after_diff >= SARIMA_S + SARIMA_CAP_P + 1 else 0
                safe_Q = SARIMA_CAP_Q if n_after_diff >= SARIMA_S + SARIMA_CAP_Q + 1 else 0
            
            p, d, q = safe_p, SARIMA_D, safe_q
            P, D, Q = safe_P, SARIMA_CAP_D, safe_Q
            print(f"    Grid search failed, using safe fallback: ({p},{d},{q})({P},{D},{Q})[{SARIMA_S}]")
        else:
            p, d, q = best_order['p'], best_order['d'], best_order['q']
            P, D, Q = best_order['P'], best_order['D'], best_order['Q']
            print(f"    Best order: {best_order['order']} (CV RMSE: {best_order['avg_rmse']:.2f})")
            print(f"    Passes diagnostics: {best_order['passes_diagnostics']}")
        
        best_orders[segment][target_label] = {'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q}
        
        result = train_and_log_model(
            segment, target_col, target_label,
            p, d, q, P, D, Q, SARIMA_S,
            ITERATIONS, LEARNING_RATE
        )
        
        all_results[segment][target_label] = result
        
        sarima_m = result['sarima_metrics']
        baseline_m = result['baseline_metrics']
        
        all_metrics.append({
            'Segment': segment,
            'Target': target_label,
            'Order': f"({p},{d},{q})({P},{D},{Q})[{SARIMA_S}]",
            'SARIMA_RMSE': sarima_m['RMSE'],
            'SARIMA_MAPE': sarima_m['MAPE'],
            'SARIMA_R2': sarima_m['R2'],
            'Baseline_RMSE': baseline_m['RMSE'],
            'Baseline_MAPE': baseline_m['MAPE'],
            'Beats_Baseline': sarima_m['RMSE'] < baseline_m['RMSE']
        })
        
        print(f"    SARIMA  -> RMSE: {sarima_m['RMSE']:.2f}, MAPE: {sarima_m['MAPE']:.2f}%")
        print(f"    Baseline -> RMSE: {baseline_m['RMSE']:.2f}, MAPE: {baseline_m['MAPE']:.2f}%")
        print(f"    Beats Baseline: {sarima_m['RMSE'] < baseline_m['RMSE']}")

print("\n" + "="*60)
print("Training complete!")
print("="*60)


# ## 13. Accuracy Metrics Summary

# In[ ]:


metrics_df = pd.DataFrame(all_metrics)
print("\n" + "="*80)
print("SARIMA MODEL ACCURACY METRICS (Test Set)")
print("="*80)
print(metrics_df.to_string())

metrics_df.to_csv(os.path.join(ARTIFACTS_DIR, 'accuracy_metrics.csv'), index=False)


# ## 14. Generate Future Forecasts

# In[ ]:


forecast_horizon = 12
last_date = monthly_df['date'].max()
forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')

print(f"Last closed month: {last_date}")
print(f"Forecast period: {forecast_dates[0]} to {forecast_dates[-1]}")

future_forecasts = {}

for segment in segments:
    segment_df = monthly_df[monthly_df['CUST_STEERING_L3_NAME'] == segment].sort_values('date')
    full_net_sales = segment_df['net_sales_units'].values
    full_returns = segment_df['returns_units'].values
    
    ns_order = best_orders.get(segment, {}).get('net_sales', 
        {'p': SARIMA_P, 'd': SARIMA_D, 'q': SARIMA_Q, 'P': SARIMA_CAP_P, 'D': SARIMA_CAP_D, 'Q': SARIMA_CAP_Q})
    ret_order = best_orders.get(segment, {}).get('returns',
        {'p': SARIMA_P, 'd': SARIMA_D, 'q': SARIMA_Q, 'P': SARIMA_CAP_P, 'D': SARIMA_CAP_D, 'Q': SARIMA_CAP_Q})
    
    model_ns = SARIMAModel(
        ar_order=ns_order['p'], diff_order=ns_order['d'], ma_order=ns_order['q'],
        seasonal_ar_order=ns_order['P'], seasonal_diff_order=ns_order['D'], seasonal_ma_order=ns_order['Q'],
        seasonal_period=SARIMA_S,
        iterations=ITERATIONS, learning_rate=LEARNING_RATE
    )
    model_ns.fit(full_net_sales)
    forecast_ns = np.clip(np.rint(model_ns.predict(None, forecast_horizon)), 0, None).astype(int)
    
    model_ret = SARIMAModel(
        ar_order=ret_order['p'], diff_order=ret_order['d'], ma_order=ret_order['q'],
        seasonal_ar_order=ret_order['P'], seasonal_diff_order=ret_order['D'], seasonal_ma_order=ret_order['Q'],
        seasonal_period=SARIMA_S,
        iterations=ITERATIONS, learning_rate=LEARNING_RATE
    )
    model_ret.fit(full_returns)
    forecast_ret = np.clip(np.rint(model_ret.predict(None, forecast_horizon)), 0, None).astype(int)
    
    forecast_total = np.maximum(forecast_ns - forecast_ret, 0)
    
    future_forecasts[segment] = {
        'dates': forecast_dates,
        'net_sales': forecast_ns,
        'returns': forecast_ret,
        'total_net': forecast_total,
        'historical_dates': segment_df['date'].values,
        'historical_total': segment_df['total_net_units'].values
    }
    
    print(f"\n{segment} Forecast (orders: net_sales={ns_order}, returns={ret_order}):")
    for i in range(forecast_horizon):
        print(f"  {forecast_dates[i].strftime('%Y-%m')}: Net={forecast_ns[i]:.0f}, Returns={forecast_ret[i]:.0f}, Total={forecast_total[i]:.0f}")


# ## 15. Forecast Visualization

# In[ ]:


fig, axes = plt.subplots(len(segments), 1, figsize=(14, 4 * len(segments)))

if len(segments) == 1:
    axes = [axes]

for idx, segment in enumerate(segments):
    ax = axes[idx]
    data = future_forecasts[segment]
    
    hist_dates = pd.to_datetime(data['historical_dates'])
    hist_values = data['historical_total']
    
    ax.plot(hist_dates, hist_values, 'b-', linewidth=2, label='Historical', marker='o', markersize=4)
    
    fore_dates = data['dates']
    fore_values = data['total_net']
    
    ax.plot(fore_dates, fore_values, 'r--', linewidth=2, label='Forecast', marker='s', markersize=4)
    
    ax.axvline(x=hist_dates.max(), color='gray', linestyle=':', alpha=0.7, linewidth=2)
    
    ax.set_title(f'{segment} - Total Net Units', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Units')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, 'sarima_forecast_visualization.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\nForecast visualization saved.")


# ## 16. Export Results

# In[ ]:


all_forecasts = []

for segment in segments:
    data = future_forecasts[segment]
    for i, date in enumerate(data['dates']):
        all_forecasts.append({
            'segment': segment,
            'forecast_date': date,
            'net_sales_forecast': int(data['net_sales'][i]),
            'returns_forecast': int(data['returns'][i]),
            'total_net_forecast': int(data['total_net'][i])
        })

forecasts_df = pd.DataFrame(all_forecasts)
forecasts_df.to_csv('../dataset/sarima_forecasts.csv', index=False)
print("Forecasts exported to: ../dataset/sarima_forecasts.csv")

print(forecasts_df.head(15).to_string())


# ## 17. Summary

# In[ ]:


print("\n" + "="*60)
print("SARIMA IMPROVED PIPELINE COMPLETE")
print("="*60)
print(f"\nModels trained: {len(segments) * 2}")
print(f"\nKey Features:")
print("  - True multiplicative SARIMA with seasonal AR/MA terms")
print("  - Seasonal interaction lags (e.g., 1+12=13)")
print("  - Grid search per segment (24 order combinations)")
print("  - Rolling-origin cross-validation for model selection")
print("  - Residual diagnostics filtering")
print("  - Log transform with shift constant")
print("  - Correct inverse double differencing")
print("  - Baseline comparisons logged")
print(f"\nSelected Orders per Segment:")
for seg in best_orders:
    print(f"  {seg}:")
    for target, order in best_orders[seg].items():
        print(f"    {target}: ({order['p']},{order['d']},{order['q']})({order['P']},{order['D']},{order['Q']})[{SARIMA_S}]")
print(f"\nMLflow Experiment: {EXPERIMENT_NAME}")
print("\nTo view MLflow UI: mlflow ui --port 5000")
