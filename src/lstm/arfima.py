import numpy as np

def simulate_arma(n, ar_params=[], ma_params=[]):
    """
    Simulate an ARMA process of length n.
    
    Parameters:
      n         : int, number of time steps
      ar_params : list or array of AR coefficients (phi1, phi2, ...)
      ma_params : list or array of MA coefficients (theta1, theta2, ...)
    
    Returns:
      x         : simulated ARMA process (array of length n)
    """
    p = len(ar_params)
    q = len(ma_params)
    max_lag = max(p, q)

    x = np.zeros(n + max_lag)
    e = np.random.normal(size=n + max_lag)
    
    # Simulate ARMA process
    for t in range(max_lag, n + max_lag):
        ar_term = np.dot(ar_params, x[t-1::-1][:p]) if p > 0 else 0
        ma_term = np.dot(ma_params, e[t-1::-1][:q]) if q > 0 else 0
        x[t] = ar_term + e[t] + ma_term
    
    return x[max_lag:]

def frac_integration(x, d, lag_max=None):
    """
    Apply fractional integration to a time series x.
    
    Parameters:
      x       : array, the input time series
      d       : fractional integration parameter (float)
      lag_max : maximum number of lag weights to compute. If None, uses len(x).
    
    Returns:
      y       : fractionally integrated series
    """
    n = len(x)
    if lag_max is None:
        lag_max = n

    # Compute the fractional weights
    psi = np.empty(lag_max)
    psi[0] = 1.0
    for j in range(1, lag_max):
        psi[j] = psi[j-1] * ((j - 1 + d) / j)

    # Convolve x with the weights
    y = np.zeros(n)
    for t in range(n):
        y[t] = np.dot(psi[:t+1], x[t::-1])

    return y

def simulate_arfima(n, d, ar_params=[], ma_params=[], lag_max=None):
    """
    Simulate an ARFIMA(p, d, q) process.
    
    This function first simulates an ARMA(p, q) process of length n,
    and then applies fractional integration to incorporate the long-memory
    component with fractional differencing parameter d.
    
    Parameters:
      n         : int, length of the series to simulate
      d         : float, fractional integration parameter
      ar_params : list, AR coefficients
      ma_params : list, MA coefficients
      lag_max   : int, optional maximum number of weights for fractional integration
    
    Returns:
      y         : simulated ARFIMA series (array of length n)
    """
    x = simulate_arma(n, ar_params, ma_params)

    y = frac_integration(x, d, lag_max)

    return y


if __name__ == "__main__":
    np.random.seed(42)
    n = 1_000           # length of the series
    d = 0.4             # Differencing parameter
    ar_params = [0.5]   # AR(1)
    ma_params = [-0.1]  # MA(1)

    y = simulate_arfima(n, d, ar_params, ma_params)

    import matplotlib.pyplot as plt
    plt.plot(y)
    plt.title("Simulated ARFIMA Process")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

    from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
    plt.figure(figsize=(8, 4))
    plot_acf(y, lags=50, ax=plt.gca())
    plt.title("Autocorrelation Function (ACF)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plot_pacf(y, lags=50, ax=plt.gca(), method='ywm')
    plt.title("Partial Autocorrelation Function (PACF)")
    plt.tight_layout()
    plt.show()
