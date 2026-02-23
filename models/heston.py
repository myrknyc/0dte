import numpy as np
from config import MIN_VARIANCE


def _compute_conditional_moments(v_current, kappa, theta_v, sigma_v, dt):
    exp_kappa_dt = np.exp(-kappa * dt)
    
    # Conditional mean
    m = theta_v + (v_current - theta_v) * exp_kappa_dt
    
    # Conditional variance (two terms)
    term1 = v_current * (sigma_v**2 / kappa) * (exp_kappa_dt - np.exp(-2 * kappa * dt))
    term2 = theta_v * (sigma_v**2 / (2 * kappa)) * (1 - exp_kappa_dt)**2
    s2 = term1 + term2
    
    return m, s2


def _qe_quadratic_case(m, s2, psi):
    psi_inv = 1.0 / psi
    
    # Compute b² (guard sqrt argument)
    b_squared = 2 * psi_inv - 1 + np.sqrt(max(2 * psi_inv, 0)) * np.sqrt(max(2 * psi_inv - 1, 0))
    b = np.sqrt(max(b_squared, 0.0))
    
    # Compute a
    a = m / (1 + b_squared)
    
    # Quadratic branch: NO point mass at zero (that's only in the exponential branch)
    Z = np.random.standard_normal()
    v_next = a * (b + Z)**2
    
    return v_next


def _qe_exponential_case(m, s2, psi):
    # Probability of zero — CLAMP to [0, 1]
    p = (psi - 1) / (psi + 1)
    p = np.clip(p, 0.0, 1.0)
    
    # Exponential rate parameter (guard denominator)
    beta = (1 - p) / max(m, 1e-12)
    
    # Sample uniform
    U = np.random.uniform(0, 1)
    
    if U <= p:
        v_next = 0.0
    else:
        v_next = (1.0 / beta) * np.log(max((1 - p) / (1 - U), 1e-12))
    
    return v_next


def heston_qe_step(v_current, kappa, theta_v, sigma_v, dt, psi_c=1.5):
    # Input validation
    if v_current < 0:
        raise ValueError(f"v_current must be non-negative, got {v_current}")
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if theta_v <= 0:
        raise ValueError(f"theta_v must be positive, got {theta_v}")
    if sigma_v <= 0:
        raise ValueError(f"sigma_v must be positive, got {sigma_v}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    
    # Handle edge case: v_current ≈ 0
    if v_current < MIN_VARIANCE:
        v_current = MIN_VARIANCE
    
    # Step 1: Compute conditional moments
    m, s2 = _compute_conditional_moments(v_current, kappa, theta_v, sigma_v, dt)
    
    # Step 2: Compute scaled variance
    # Avoid division by zero
    if m < 1e-12:
        psi = 1e12  # Very large → use exponential case
    else:
        psi = s2 / (m * m)
    
    # Step 3: Branch based on psi
    if psi <= psi_c:
        # Quadratic approximation (low variance regime)
        v_next = _qe_quadratic_case(m, s2, psi)
    else:
        # Exponential approximation (high variance regime)
        v_next = _qe_exponential_case(m, s2, psi)
    
    # Step 4: Ensure positivity
    v_next = max(v_next, MIN_VARIANCE)
    
    return v_next


def simulate_variance_path(v0, kappa, theta_v, sigma_v, times, dt_array, psi_c=1.5):
    n_steps = len(dt_array)
    v_path = np.zeros(n_steps + 1)
    v_path[0] = v0
    
    for i in range(n_steps):
        v_path[i + 1] = heston_qe_step(
            v_current=v_path[i],
            kappa=kappa,
            theta_v=theta_v,
            sigma_v=sigma_v,
            dt=dt_array[i],
            psi_c=psi_c
        )
    
    return v_path


def simulate_variance_paths(v0, kappa, theta_v, sigma_v, times, dt_array, 
                            n_paths, psi_c=1.5, seed=None):
    """Original slow version - kept for reference/testing."""
    if seed is not None:
        np.random.seed(seed)
    
    n_steps = len(dt_array)
    v_paths = np.zeros((n_paths, n_steps + 1))
    v_paths[:, 0] = v0
    
    # Simulate each path
    for p in range(n_paths):
        for i in range(n_steps):
            v_paths[p, i + 1] = heston_qe_step(
                v_current=v_paths[p, i],
                kappa=kappa,
                theta_v=theta_v,
                sigma_v=sigma_v,
                dt=dt_array[i],
                psi_c=psi_c
            )
    
    return v_paths


def heston_qe_step_vectorized(v_current, kappa, theta_v, sigma_v, dt, psi_c=1.5):
    n_paths = len(v_current)
    
    # Ensure minimum variance
    v_current = np.maximum(v_current, MIN_VARIANCE)
    
    # Step 1: Compute conditional moments (vectorized)
    exp_kappa_dt = np.exp(-kappa * dt)
    m = theta_v + (v_current - theta_v) * exp_kappa_dt
    
    term1 = v_current * (sigma_v**2 / kappa) * (exp_kappa_dt - np.exp(-2 * kappa * dt))
    term2 = theta_v * (sigma_v**2 / (2 * kappa)) * (1 - exp_kappa_dt)**2
    s2 = term1 + term2
    
    # Step 2: Compute psi (vectorized)
    psi = np.where(m > 1e-12, s2 / (m * m), 1e12)
    
    # Initialize result
    v_next = np.zeros(n_paths)
    
    # Generate all random numbers at once
    U = np.random.uniform(0, 1, n_paths)
    Z = np.random.standard_normal(n_paths)
    
    # ---- QUADRATIC CASE (psi <= psi_c) ----
    # No point mass at zero here — that only exists in the exponential branch
    quad_mask = psi <= psi_c
    if np.any(quad_mask):
        m_q = m[quad_mask]
        psi_q = psi[quad_mask]
        Z_q = Z[quad_mask]
        
        psi_inv = 1.0 / psi_q
        b_squared = (
            2 * psi_inv - 1
            + np.sqrt(np.maximum(2 * psi_inv, 0.0)) * np.sqrt(np.maximum(2 * psi_inv - 1, 0.0))
        )
        b = np.sqrt(np.maximum(b_squared, 0.0))
        a = m_q / (1.0 + b_squared)
        
        v_next[quad_mask] = a * (b + Z_q) ** 2
    
    # ---- EXPONENTIAL CASE (psi > psi_c) ----
    exp_mask = ~quad_mask
    if np.any(exp_mask):
        m_e = m[exp_mask]
        psi_e = psi[exp_mask]
        U_e = U[exp_mask]
        
        p = (psi_e - 1) / (psi_e + 1)
        p = np.clip(p, 0.0, 1.0)  # CLAMP probability
        beta = (1 - p) / np.maximum(m_e, 1e-12)
        
        # Zero with probability p, otherwise exponential
        # F⁻¹(u) = -(1/β)·log(1-u) shifted for probability mass at zero
        v_exp = np.where(
            U_e <= p, 
            0.0, 
            (1.0 / beta) * np.log(np.maximum((1 - p) / (1 - U_e), 1e-12))
        )
        v_next[exp_mask] = v_exp
    
    # Ensure positivity
    v_next = np.maximum(v_next, MIN_VARIANCE)
    
    return v_next


def simulate_variance_paths_fast(v0, kappa, theta_v, sigma_v, times, dt_array, 
                                  n_paths, psi_c=1.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    n_steps = len(dt_array)
    v_paths = np.zeros((n_paths, n_steps + 1))
    v_paths[:, 0] = v0
    
    # Simulate all paths together, one time step at a time
    for i in range(n_steps):
        v_paths[:, i + 1] = heston_qe_step_vectorized(
            v_current=v_paths[:, i],
            kappa=kappa,
            theta_v=theta_v,
            sigma_v=sigma_v,
            dt=dt_array[i],
            psi_c=psi_c
        )
    
    return v_paths


def check_feller_condition(kappa, theta_v, sigma_v):
    feller_lhs = 2 * kappa * theta_v
    feller_rhs = sigma_v**2
    
    ratio = feller_lhs / feller_rhs if feller_rhs > 0 else np.inf
    satisfied = feller_lhs >= feller_rhs
    
    return satisfied, ratio

