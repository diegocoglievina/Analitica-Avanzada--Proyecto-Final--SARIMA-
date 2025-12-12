Okay, this is the dataframe that we got:

CUST_STEERING_L3_NAME	ACCOUNTING_PERIOD	net_sales_units	returns_units	total_net_units
DEALER	202301	8755	3626	5129

Now, I am ready to make the ml flow. Remember, the objective is to do evertything from scratch, so here are a couple of functions that might come in handy:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.mstats import winsorize

def difference(data, order):
    diff = []
    for i in range(order, len(data)):
        diff.append(data[i] - data[i - order])
    return np.array(diff)

def seasonal_difference(data, period):
    diff = []
    for i in range(period, len(data)):
        diff.append(data[i] - data[i - period])
    return np.array(diff)

def autoregression(data, lags, coefficients):
    total = 0
    for i in range(lags):
        total += coefficients[i] * data[i]
    return total

def moving_average(errors, lags, coefficients):
    total = 0
    for i in range(lags):
        total += coefficients[i] * errors[i]
    return total

def autoreg_fit(data, order, iterations, learning_rate):
    n = len(data)
    q = order
    coefficients = [0] * q 

    predictions = []

    for iteration in range(iterations):
        total_error = 0

        for t in range(q, n):
            predict = 0
            for lag in range(q):
                predict += coefficients[lag] * data[t - lag - 1]
            predictions.append(predict)

            error = data[t] - predict
            total_error += error ** 2

            for lag in range(q):
                grad = learning_rate * error * data[t - lag - 1]
                coefficients[lag] += grad

        if iteration % 100 == 0:
            print(f'Iteration: {iteration}, Error: {total_error}')

    return coefficients

def moving_average_fit(data, order, iterations, learning_rate):
    n = len(data)
    coefficients = [0] * order
    mean = np.mean(data)
    errors = [0] * order
    
    for iteration in range(iterations):
        predictions = []
        total_error = 0

        for t in range(n):
            predict = mean
            for lag in range(order):
                if t - lag - 1 >= 0: 
                    predict += coefficients[lag] * errors[-(lag + 1)]
            predictions.append(predict)

            if t < n:
                error = data[t] - predict
                total_error += error ** 2

                for lag in range(order):
                    if t - lag - 1 >= 0:
                        grad = learning_rate * error * errors[-(lag + 1)]

                        coefficients[lag] += grad
            
                errors.append(error)
                if len(errors) > order:
                    errors.pop(0)
        if iteration % 100 == 0:
            print(f'Iteration: {iteration}, Error: {total_error}')
    
    return coefficients

def sarima_model(data, seasonal_period, order, iterations, learning_rate):
    s = seasonal_period

    seasonal_differenced_data = seasonal_difference(data, s)
    differenced_data = difference(seasonal_differenced_data, 1)
    n = len(differenced_data)

    ar_coeffs = autoreg_fit(differenced_data, order, iterations, learning_rate)
    ma_coeffs = moving_average_fit(differenced_data, order, iterations, learning_rate)

    errors = [0] * order
    for t in range(n - order):
        predict = autoregression(differenced_data[t:t + order], order, ar_coeffs)
        if t < n:
            error = differenced_data[t] - predict
            errors.append(error)
            errors.pop(0)

    predictions = []
    for t in range(n - order):
        ar = autoregression(differenced_data[t:t + order], order, ar_coeffs)
        ma = moving_average(errors, order, ma_coeffs)
        predict = ar + ma
        predictions.append(predict)

        if t < n:
            error = data[t] - predict
            errors.append(error)
            if len(errors) > order:
                errors.pop(0)

    return np.array(predictions), ar_coeffs, ma_coeffs

def reverse_difference(differenced_data, original_data, lag):
    reversed_data = []
    for i in range(len(differenced_data)):
        reversed_value = differenced_data[i] + original_data[i + lag]
        reversed_data.append(reversed_value)
    return np.array(reversed_data)

def reverse_seasonal_difference(differenced_data, original_data, seasonal_period):
    reversed_data = []
    for i in range(len(differenced_data)):
        reversed_value = differenced_data[i] + original_data[i + seasonal_period]
        reversed_data.append(reversed_value)
    return np.array(reversed_data)

def reverse_transform(predictions, transformed_data, seasonal_period, difference_order, normalized_data):
    regular_reversed = reverse_difference(predictions, transformed_data, difference_order)
    seasonal_reversed = reverse_seasonal_difference(regular_reversed, normalized_data, seasonal_period)
    denormalized_predictions = seasonal_reversed * transformed_data.std() + transformed_data.mean()
    reverse_predictions = denormalized_predictions ** 2

    temp = reverse_predictions
    best_rmse = np.inf
    best_div = 0
    for i in range(1, 200):
        final_predictions = temp / i
        rmse = np.sqrt(np.mean((data['precipitation'][order:order + len(final_predictions)] - final_predictions) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_div = i
    # print(f'Best RMSE: {best_rmse}, Divisor: {best_div}')
    final_predictions = temp / best_div # undo sqaure root in a more controlled way, dividing by 31 was the best value found
    
    return final_predictions

def forecast(ar_coeffs, ma_coeffs, num_predictions, order, seasonal_period, data, transformed_data):
    predictions = []
    errors = [0] * order

    for i in range(num_predictions):
        ar = autoregression(data[-order:], order, ar_coeffs)
        ma = moving_average(errors, order, ma_coeffs)
        
        predict = ar + ma
        predictions.append(predict)

        data = np.append(data, predict)
        
        if i > 1:
            error = predict - predictions[-2] # Difference from the last prediction because no real errors, since it's predicting
        else:
              error = 0
        errors.append(error)
        errors.pop(0)

    final_predictions = reverse_transform(predictions, transformed_data, seasonal_period, order, data)
    
    return final_predictions


learning_rate = 0.001
num_iterations = 1001
order = 1
difference_order = 1
seasonal_period = 12
num_predictions = 12

data = pd.read_csv('precipitation.csv')
data = data[data['state'] == 'BA']
data = data.drop('state', axis=1)
data['date'] = pd.to_datetime(data['date'], dayfirst=True)
data = data.set_index('date')

winsorized_data = winsorize(data['precipitation'], limits=[0, 0.03]) # Trim top 3% because of outliers
winsorized_data = pd.Series(winsorized_data)

transformed_data = np.sqrt(data)  # Use square root because data has a .7 skew
transformed_data = transformed_data['precipitation'].values
normalized_data = (transformed_data - transformed_data.mean()) / transformed_data.std()

predictions, ar_coeffs, ma_coeffs = sarima_model(normalized_data, seasonal_period, order, num_iterations, learning_rate)

final_predictions = reverse_transform(predictions, transformed_data, seasonal_period, difference_order, normalized_data)
final_predictions = pd.Series(final_predictions, index=data.index[order:order + len(final_predictions)])

rmse = np.sqrt(np.mean((data['precipitation'][order:order + len(final_predictions)] - final_predictions) ** 2))
print(f'RMSE: {rmse}')
per_rmse = rmse / data['precipitation'][order:order + len(final_predictions)].mean() * 100
print(f'RMSE Percentage: {per_rmse}')

plt.plot(data['precipitation'][order:order + len(final_predictions)])
plt.plot(final_predictions, color='red')
plt.xlabel('Date')
plt.ylabel('Precipitation')
plt.title('SARIMA Model')
plt.legend(['Actual', 'Predicted'])
plt.show()

# forecasted = forecast(ar_coeffs, ma_coeffs, num_predictions, order, seasonal_period, normalized_data, transformed_data)
# forecasted = pd.Series(forecasted, index=data.index[-num_predictions:])
# rmse = np.sqrt(np.mean((data['precipitation'][-len(forecasted):] - forecasted)) ** 2)
# print(f'12 Months RMSE: {rmse}')
# per_rmse = rmse / data['precipitation'][-len(forecasted):].mean() * 100
# print(f'12 Months RMSE Percentage: {per_rmse}')

# plt.plot(data['precipitation'][-len(forecasted):])
# plt.plot(forecasted, color='red')
# plt.xlabel('Date')
# plt.ylabel('Precipitation')
# plt.title('Forecast')
# plt.legend(['Actual', 'Forecast'])
# plt.show()

Now, finally, here is the ML Flow set up, with another example of course:

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

# %% [markdown]
# # Selección de características

# %%
file_path = 'C:/Users/cogli/OneDrive - Red de Universidades Anáhuac/Semestres/Posgrado/Analítica Avanzada/mini_proyecto/source/precipitacion_dataframe.csv'

# %%
precipitacion_df = pd.read_csv(file_path)
precipitacion_df.head()

# %% [markdown]
# # Definición de datos de entrenamiento y prueba

# %%
X = precipitacion_df[['temperatura_maxima', 'temperatura_minima', 'precipitacion']]

Y = precipitacion_df['lluvia_mañana']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# %% [markdown]
# # Modelado

# %% [markdown]
# ## Utils

# %%
class RegresionLogistica(mlflow.pyfunc.PythonModel):
    def __init__(self, tasa_aprendizaje=0.01, n_iteraciones=1000):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.n_iteraciones = n_iteraciones
        self.pesos = None   
        self.bias = 0

    def sigmoide(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_muestras, n_caracteristicas = X.shape
        self.pesos = np.zeros(n_caracteristicas)
        for _ in range(self.n_iteraciones):
            eta =  self.pesos @ X.T + self.bias
            y_pred = self.sigmoide(eta)

            grad_pesos = X.T @ (y_pred - y) / n_muestras
            grad_bias = np.mean(y_pred - y)

            self.pesos -= self.tasa_aprendizaje * grad_pesos
            self.bias -= self.tasa_aprendizaje * grad_bias

        return self
    
    def predict(self, X):
        logits = X @ self.pesos + self.bias      
        probs = self.sigmoide(logits)              
        return (probs >= 0.5).astype(int)


# %%

lr = 0.01
iters = 2000

example = X_train.iloc[:1]
signature = infer_signature(X_train, y_train)

# %%


with mlflow.start_run(run_name="regresion_logistica"):

    model_scratch = RegresionLogistica(tasa_aprendizaje=lr, n_iteraciones=iters)
    model_scratch.fit(X_train, y_train)
    
    y_pred = model_scratch.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("n_iterations", iters)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    mlflow.pyfunc.log_model(
        name="model_scratch",
        python_model=model_scratch,
        input_example=example,
        signature=signature
    )

# %%
mlflow.end_run()

# %% [markdown]
# ## Modelo 2 (Regularizado - Sklearn)
# Entrenamiento de regresión logística con regularización L2 y C=0.1 usando scikit-learn.

# %%
with mlflow.start_run(run_name="regularized_logistic_regression_L2"):

    import sklearn
    import cloudpickle

    lr_reg = LogisticRegression(penalty='l2', C=0.1, random_state=42, solver='lbfgs', max_iter=2000)
    lr_reg.fit(X_train, y_train)

    y_pred_reg    = lr_reg.predict(X_test)
    y_prob_reg    = lr_reg.predict_proba(X_test)[:, 1]

    accuracy_reg  = accuracy_score(y_test,  y_pred_reg)
    precision_reg = precision_score(y_test, y_pred_reg, zero_division=0)
    recall_reg    = recall_score(y_test,    y_pred_reg, zero_division=0)
    auc_reg       = roc_auc_score(y_test,   y_prob_reg)

    # Use dynamic versions to avoid mismatches
    minimal_reqs = [f"scikit-learn=={sklearn.__version__}", f"cloudpickle=={cloudpickle.__version__}"]
    
    mlflow.log_metric("accuracy",  accuracy_reg )
    mlflow.log_metric("precision", precision_reg)
    mlflow.log_metric("recall",    recall_reg   )
    mlflow.log_metric("auc",       auc_reg      )
    
    mlflow.log_param("C", 0.1)
    mlflow.log_param("penalty", "l2")

    mlflow.sklearn.log_model(
        sk_model=lr_reg,
        name="model",
        pip_requirements=minimal_reqs,
        input_example=example,
        signature=signature
    )




