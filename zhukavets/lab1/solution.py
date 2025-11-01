import pandas as pd
import numpy as np
import re 
import matplotlib.pyplot as plt

from math import erf, sqrt
from scipy.stats import norm


def _normal_cdf(x: float, mean: float, std: float) -> float:
    """Cumulative distribution function for Normal(mean, std^2) using erf."""
    if std <= 0:
        return 0.0
    z = (x - mean) / (std * sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


def manual_chi_squared_test(data: np.ndarray, bins: int | str = 'auto'):
    """Compute Pearson chi-square statistic for goodness-of-fit to normal with sample mean/std.
    Returns (chi2_stat, dof, observed_counts, expected_counts).
    """
    data = np.asarray(data, dtype=float)
    n = data.size
    if n < 2 or not np.isfinite(data).all():
        return np.nan, 0, None, None

    sample_mean = float(np.mean(data))
    sample_std = float(np.std(data, ddof=0))  # population std to match moments above

    counts, edges = np.histogram(data, bins=bins)
    k = counts.size

    # Avoid too-fine binning: merge bins to ensure expected >= 5
    # First compute expected probabilities via normal CDF
    probs = []
    for i in range(k):
        p = _normal_cdf(edges[i + 1], sample_mean, sample_std) - _normal_cdf(edges[i], sample_mean, sample_std)
        probs.append(max(p, 1e-12))
    probs = np.array(probs)

    expected = n * probs

    # Merge adjacent bins with expected < 5
    merged_obs = []
    merged_exp = []
    acc_o = 0.0
    acc_e = 0.0
    for o, e in zip(counts, expected):
        acc_o += o
        acc_e += e
        if acc_e >= 5.0:
            merged_obs.append(acc_o)
            merged_exp.append(acc_e)
            acc_o = 0.0
            acc_e = 0.0
    if acc_e > 0.0:
        if merged_exp:
            # add remainder to last bin
            merged_obs[-1] += acc_o
            merged_exp[-1] += acc_e
        else:
            merged_obs.append(acc_o)
            merged_exp.append(acc_e)

    merged_obs = np.array(merged_obs, dtype=float)
    merged_exp = np.array(merged_exp, dtype=float)

    # Compute chi-square statistic
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = (merged_obs - merged_exp) ** 2 / merged_exp
        terms[~np.isfinite(terms)] = 0.0
        chi2 = float(np.sum(terms))

    # degrees of freedom: (#bins - 1 - #estimated params)
    # We estimated mean and std ⇒ 2 params
    dof = max(int(merged_obs.size - 1 - 2), 1)

    return chi2, dof, counts, expected


def calculate_moments(data):
    """Calculates Mean, Population dispersia, assymetry, and Excess Kurtosis manually."""
    n = len(data)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan

    mean = data.mean()
    
    # Calculate central moments
    m2 = ((data - mean) ** 2).sum() / n  # Second central moment (Population dispersia)
    m3 = ((data - mean) ** 3).sum() / n  # Third central moment
    m4 = ((data - mean) ** 4).sum() / n  # Fourth central moment
    
    dispersia = m2
    std_dev = np.sqrt(dispersia)
    
    # assymetry (gamma1): Third standardized moment
    assymetry = m3 / (std_dev ** 3) if std_dev > 0 else 0
    
    # Excess Kurtosis (gamma2): Fourth standardized moment minus 3
    kurtosis = (m4 / (m2 ** 2)) - 3 if m2 > 0 else 0
    
    return mean, dispersia, assymetry, kurtosis


def manual_ecdf(data: np.ndarray, ax, title: str):
    data = np.asarray(data, dtype=float)
    x = np.sort(data)
    y = np.arange(1, x.size + 1) / x.size
    ax.step(x, y, where='post')
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('ECDF')


def manual_qqplot(data: np.ndarray, ax, title: str):
    data = np.asarray(data, dtype=float)
    n = data.size
    if n == 0:
        ax.set_title(title)
        return
    x = np.sort(data)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    p = (np.arange(1, n + 1) - 0.5) / n
    # Inverse standard normal via scipy.stats.norm.ppf
    z = norm.ppf(p)
    theoretical = mean + std * z
    ax.scatter(theoretical, x, s=10, alpha=0.7)
    # reference line
    mn = min(theoretical[0], x[0])
    mx = max(theoretical[-1], x[-1])
    ax.plot([mn, mx], [mn, mx], color='red', linewidth=1)
    ax.set_title(title)
    ax.set_xlabel('Theoretical Quantiles (Normal)')
    ax.set_ylabel('Sample Quantiles')


def _sanitize_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", text)


def perform_analysis(data, title_prefix="Initial", label="Data"):
    """Performs full statistical analysis (Tasks I & II) on a given dataset."""
    print(f"\n--- Analysis for {title_prefix} {label} (n={len(data)}) ---")
    
    # --- I. Calculate Characteristics ---
    mean, dispersia, assymetry, kurtosis = calculate_moments(data)
    
    quantiles = np.quantile(data, [0.25, 0.5, 0.75])
    q1 = quantiles[0]
    median = quantiles[1]
    q3 = quantiles[2]
    
    # Mode - Using pandas Series method
    mode = pd.Series(data).mode()
    
    iqr = q3 - q1

    print("\n[I. Characteristics]")
    print(f"1. Mean: {mean:.4f}")
    print(f"2. Dispersia: {dispersia:.4f} (Population)")
    print(f"3. Mode: {list([int(i) for i in list(mode.values)])}")
    print(f"4. Median: {median:.4f}")
    print(f"5. Quantiles (0.25, 0.5, 0.75): Q1={q1:.4f}, Q2(Median)={median:.4f}, Q3={q3:.4f}")
    print(f"6. Kurtosis (Excess): {kurtosis:.4f}")
    print(f"7. Assymerty: {assymetry:.4f}")
    print(f"8. Interquartile Range (IQR): {iqr:.4f}")
    
    # --- II. Check for Normality ---
    print("\n[II. Normality Check]")
    
    # 1. Chi-squared Test (Manual)
    chi_sq_stat, dof, _, _ = manual_chi_squared_test(np.asarray(data, dtype=float))
    if not np.isnan(chi_sq_stat):
        print("1. Chi-squared Test (Manual Implementation):")
        print(f"   Chi-square Statistic: {chi_sq_stat:.4f}, DoF: {dof}")
    
    # 2. assymetry and Kurtosis
    print("\n2. assymetry and Kurtosis Criteria:")
    print(f"   assymetry ({assymetry:.4f}): Closer to 0 is better. Suggests non-normality if large.")
    print(f"   Kurtosis ({kurtosis:.4f}): Closer to 0 is better. Positive means heavy tails.")

    # --- I. Build Plots ---
    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean plot style
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Histogram
    axes[0].hist(data, bins='auto', density=True, alpha=0.7, edgecolor='black')
    axes[0].set_title(f'{title_prefix} {label}: Histogram')
    axes[0].set_xlabel(label)
    axes[0].set_ylabel('Density')
    
    # 2. Empirical Distribution Function (ECDF)
    manual_ecdf(np.asarray(data, dtype=float), axes[1], f'{title_prefix} {label}: ECDF')
    
    # 3. Q-Q plot (Part of Task II)
    manual_qqplot(np.asarray(data, dtype=float), axes[2], f'{title_prefix} {label}: Q-Q Plot')
    
    plt.tight_layout()
    # Save unique filename for each analysis
    safe_prefix = _sanitize_filename(str(title_prefix)).lower()
    safe_label = _sanitize_filename(str(label)).lower()
    out_path = f'zhukavets/{safe_prefix}_{safe_label}.png'
    plt.savefig(out_path)
    # plt.show()


def analyze_transforms(x: np.ndarray, label: str):
    # Standardization
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    if std > 0:
        z = (x - mean) / std
        perform_analysis(z, title_prefix="Transformed Zscore", label=label)
    # Log1p (for non-negative values)
    x_nonneg = x[x >= 0]
    if x_nonneg.size > 0:
        perform_analysis(np.log1p(x_nonneg), title_prefix="Transformed Log1p", label=label)


def analyze_by_group(df: pd.DataFrame, value_col: str, group_col: str = 'School_Grade'):
    if group_col not in df.columns:
        print(f"Column '{group_col}' not found for grouping.")
        return
    g = df[[group_col, value_col]].dropna()
    groups = g.groupby(group_col)[value_col]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    for name, s in groups:
        s = pd.to_numeric(s, errors='coerce').dropna()
        if s.empty:
            continue
        plt.hist(s, bins='auto', alpha=0.4, density=True, label=str(name))
    plt.title(f"Histograms by {group_col} for {value_col}")
    plt.xlabel(value_col)
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    # plt.show()

    print(f"\nGroup stats (mean, variance) by {group_col}:")
    stats = groups.agg(['mean', 'var']).rename(columns={'var': 'variance'})
    print(stats)

# --- Execution of Tasks I & II on Initial Data ---

try:
    df = pd.read_csv('datasets/teen_phone_addiction_dataset.csv')

    target_column_name = 'Sleep_Hours'

    X = np.array([float(x) for x in df[target_column_name]])
    
    perform_analysis(X, title_prefix="Initial", label=target_column_name)

    # III. Transforms and re-check
    analyze_transforms(X, label=target_column_name)

    # IV. Group analysis
    analyze_by_group(df, value_col=target_column_name, group_col='School_Grade')
except FileNotFoundError:
    print('couldn`t load dataset')
except:
    print()