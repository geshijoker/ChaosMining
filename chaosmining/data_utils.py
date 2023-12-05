from typing import List, Tuple, Dict, Set, Union
import numpy as np
import scipy
from sympy import *

def create_simulation_data(
    function: str,
    num_features: int = 1,
    num_noises: int = 10,
    num_data: int = 10000,
    X_var: float = 0.33,
    y_var: float = 0.01,
    enable_der: bool = False,
    enable_int: bool = False,
    ):
    """ Creating simulation data with noisy (normal distribution) features and labels.
        Args:
            pair: a pair of instance -- (number of variables, function string)
            num_features: number of relevant features
            num_noises: number of noisy/irrelevant features
            num_data: number of vectorized data to create
            X_var: the scale of the variance of features 
            y_var: the scale of noises adding to targets
        Return:
            X: the features
            y_true: the ground truth targets
            y_noise: the noise adding to targets
            derivatives: the value of derivatives
            integrations: the value of integrations
    """
    # create symbolic variables
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = symbols('a b c d e f g h i j k l m n o p q r s t u v w x y z')
    variables = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z]
    
    X = np.clip(np.random.randn(num_data, num_features+num_noises)*X_var, -1, 1)
    X_features = [X[:, ind] for ind in range(num_features)]
    y_noise = np.random.randn(num_data)*y_var
    
    expression = sympify(function, evaluate=False)
    exp_func = lambdify(variables[:num_features], expression, 'numpy')
    y_true = exp_func(*X_features)
    
    if enable_der:
        derivatives = []
        for ind in range(num_features):
            derivative = diff(expression, variables[ind])
            der_func = lambdify(variables[:num_features], derivative, 'numpy')
            der_true = der_func(*X_features)+np.zeros(num_data)
            derivatives.append(der_true)
    
    if enable_int:
        integrations = []
        for ind in range(num_features):
            integration = integrate(expression, variables[ind])
            int_func = lambdify(variables[:num_features], integration, 'numpy')
            int_true = int_func(*X_features)
            integrations.append(int_true)
            
    if enable_der and enable_int:
        return X, y_true, y_noise, derivatives, integrations
    elif enable_der:
        return X, y_true, y_noise, derivatives
    elif enable_int:
        return X, y_true, y_noise, integrations
    else:
        return X, y_true, y_noise
    
if __name__ == '__main__':
    function = 'E**a*sin(b)-cos(d)**3/(c**2+1)'
    num_features = 4
    num_noises = 0
    num_data = 10000
    X_var = 0.33
    y_var = 0.01
    X, y_true, y_noise, derivatives, integrations = create_simulation_data(function, num_features, num_noises, num_data, X_var, y_var, enable_der=True, enable_int=True)
    print('X', X.shape, 'y true', y_true.shape, 'y noise', y_noise.shape, 
          'derivatives', len(derivatives), derivatives[0].shape, 
          'integrations', len(integrations), integrations[0].shape)