import numpy as np

def conditional_mean_forecaster(context, horizon, mu):
    forecast = []

    for i in range(1, horizon + 1):
        future_value = context[-1] + mu * i
        forecast.append(future_value)

    return forecast

def naive_forecaster(context, horizon):
    forecast = []

    for i in range(1, horizon + 1):
        future_value = context[-1]
        forecast.append(future_value)

    return forecast

def squared_error(forecast, actual, horizon):
    error = actual[horizon - 1] - forecast[horizon - 1]
    squared_error = error ** 2

    return squared_error


def mean_squared_error(squared_errors):
    total = 0

    for i in range(len(squared_errors)):
        total = total + squared_errors[i]

    mse = total / len(squared_errors)

    return mse

def root_mse(squared_errors):
    mse = mean_squared_error(squared_errors)
    rmse = mse ** 0.5

    return rmse



def rmse_floor(sigma, horizon):
    floor_rmse = sigma * (horizon ** 0.5)

    return floor_rmse



def ratio_rmse_floor(squared_errors, sigma, horizon):
    rmse = root_mse(squared_errors)
    floor = rmse_floor(sigma, horizon)

    ratio = rmse/floor

    return ratio



def skill_vs_naive(model_errors, naive_errors):
    model_mse = mean_squared_error(model_errors)
    naive_mse = mean_squared_error(naive_errors)
    ratio = (model_mse / naive_mse) ** 0.5

    return ratio



def skill_vs_conditional(model_errors, conditional_errors):
    model_mse = mean_squared_error(model_errors)
    conditional_mse = mean_squared_error(conditional_errors)
    ratio = (model_mse / conditional_mse) ** 0.5

    return ratio


def drift_mle(context):
    m = len(context) - 1
    mu_hat = (context[-1] - context[0]) / m

    return mu_hat


def plugin_forecaster(context, horizon):
    mu_hat = drift_mle(context)
    forecast = []

    for i in range(1, horizon + 1):
        future_value = context[-1] + mu_hat * i
        forecast.append(future_value)

    return forecast