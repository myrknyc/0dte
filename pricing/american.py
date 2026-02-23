import numpy as np
from scipy.stats import norm


def compute_laguerre_basis(x, degree):
    n = len(x)
    basis = np.zeros((n, degree + 1))
    
    # L₀(x) = 1
    basis[:, 0] = 1
    
    if degree >= 1:
        # L₁(x) = 1 - x
        basis[:, 1] = 1 - x
    
    # Recurrence relation: (k+1)L_{k+1}(x) = (2k+1-x)L_k(x) - k·L_{k-1}(x)
    for k in range(1, degree):
        basis[:, k + 1] = ((2*k + 1 - x) * basis[:, k] - k * basis[:, k - 1]) / (k + 1)
    
    # Apply exponential weighting (REQUIRED for orthogonality)
    for k in range(degree + 1):
        basis[:, k] *= np.exp(-x / 2)
    
    return basis


def price_american_lsm(S_paths, K, r, dt_array, option_type='call', basis_type='laguerre', degree=3):
    n_paths, n_steps_plus_1 = S_paths.shape
    n_steps = n_steps_plus_1 - 1
    
    # Compute immediate exercise value at all times
    if option_type == 'call':
        exercise_value = np.maximum(S_paths - K, 0)
    elif option_type == 'put':
        exercise_value = np.maximum(K - S_paths, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Initialize cash flows with terminal payoff
    cash_flows = exercise_value[:, -1].copy()
    
    # Backward induction (from T-1 to 1)
    for t in range(n_steps - 1, 0, -1):
        # Discount factor for this step
        df = np.exp(-r * dt_array[t])
        
        # Identify ITM paths only
        itm_mask = exercise_value[:, t] > 0
        n_itm = np.sum(itm_mask)
        
        if n_itm == 0:
            # No ITM paths, just discount
            cash_flows *= df
            continue
        
        # Extract ITM prices
        S_itm = S_paths[itm_mask, t]
        
        # Normalize to moneyness
        x = S_itm / K
        
        # Compute basis functions
        if basis_type == 'laguerre':
            basis = compute_laguerre_basis(x, degree)
        else:  # polynomial
            basis = np.column_stack([S_itm**k for k in range(degree + 1)])
        
        # Continuation values (discounted future cash flows)
        continuation = cash_flows[itm_mask] * df
        
        # Least squares regression
        # continuation ≈ β₀·basis₀ + β₁·basis₁ + ... + β_d·basis_d
        try:
            beta = np.linalg.lstsq(basis, continuation, rcond=None)[0]
        except np.linalg.LinAlgError:
            # If regression fails, just discount
            cash_flows *= df
            continue
        
        # Estimated continuation value
        continuation_estimate = basis @ beta
        
        # Immediate exercise value
        immediate = exercise_value[itm_mask, t]
        
        # Exercise decision: exercise if immediate > continuation
        exercise_now = immediate > continuation_estimate
        
        # Update cash flows
        cash_flows[itm_mask] = np.where(
            exercise_now,
            immediate,  # Exercise now
            cash_flows[itm_mask] * df  # Continue holding
        )
        
        # Discount cash flows for non-ITM paths
        cash_flows[~itm_mask] *= df
    
    # Discount all cash flows to t=0
    final_discount = np.exp(-r * dt_array[0])
    american_price = np.mean(cash_flows) * final_discount
    
    return american_price


def price_american_with_standard_error(S_paths, K, r, dt_array, option_type='call', 
                                       basis_type='laguerre', degree=3):
    n_paths = S_paths.shape[0]
    
    # Price using LSM
    american_price = price_american_lsm(S_paths, K, r, dt_array, option_type, basis_type, degree)
    
    # Compute standard error (approximate, since LSM is biased)
    # Use terminal payoffs as proxy for variance
    if option_type == 'call':
        terminal_payoffs = np.maximum(S_paths[:, -1] - K, 0)
    else:
        terminal_payoffs = np.maximum(K - S_paths[:, -1], 0)
    
    T = np.sum(dt_array)
    df = np.exp(-r * T)
    discounted_payoffs = df * terminal_payoffs
    
    std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)
    
    ci_lower = american_price - 1.96 * std_error
    ci_upper = american_price + 1.96 * std_error
    
    return {
        'price': american_price,
        'std_error': std_error,
        'confidence_interval': (ci_lower, ci_upper),
    }
