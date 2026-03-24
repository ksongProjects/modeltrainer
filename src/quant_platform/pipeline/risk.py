from __future__ import annotations

import numpy as np
import pandas as pd


def compute_var_cvar(returns: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    cutoff = np.quantile(returns, 1 - alpha)
    tail = returns[returns <= cutoff]
    return float(cutoff), float(tail.mean() if len(tail) else cutoff)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    running_max = cumulative_returns.cummax()
    drawdowns = cumulative_returns / running_max - 1.0
    return float(drawdowns.min())


def annualized_metrics(returns: pd.Series, benchmark: pd.Series | None = None) -> dict[str, float]:
    daily_mean = returns.mean()
    daily_vol = returns.std(ddof=0)
    annual_return = float((1 + daily_mean) ** 252 - 1)
    annual_vol = float(daily_vol * np.sqrt(252))
    downside_vol = float(returns[returns < 0].std(ddof=0) * np.sqrt(252) or 0.0)
    sharpe = float((daily_mean * 252) / annual_vol) if annual_vol else 0.0
    sortino = float((daily_mean * 252) / downside_vol) if downside_vol else 0.0
    hit_rate = float((returns > 0).mean())
    cumulative = (1 + returns).cumprod()
    metrics = {
        "annualized_return": annual_return,
        "annualized_volatility": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "hit_rate": hit_rate,
        "max_drawdown": max_drawdown(cumulative),
    }
    if benchmark is not None and len(benchmark) == len(returns):
        active = returns - benchmark
        tracking_error = float(active.std(ddof=0) * np.sqrt(252))
        information_ratio = float((active.mean() * 252) / tracking_error) if tracking_error else 0.0
        beta = float(np.cov(returns, benchmark)[0, 1] / np.var(benchmark)) if np.var(benchmark) else 0.0
        metrics["information_ratio"] = information_ratio
        metrics["beta"] = beta
        metrics["benchmark_alpha"] = float((returns.mean() - benchmark.mean()) * 252)
    else:
        metrics["information_ratio"] = 0.0
        metrics["beta"] = 0.0
        metrics["benchmark_alpha"] = 0.0
    return metrics


def bootstrap_drawdown_distribution(
    returns: pd.Series,
    iterations: int = 250,
    sample_size: int | None = None,
) -> list[float]:
    rng = np.random.default_rng(42)
    sample_size = sample_size or len(returns)
    results = []
    values = returns.to_numpy()
    for _ in range(iterations):
        sample = pd.Series(rng.choice(values, size=sample_size, replace=True))
        cumulative = (1 + sample).cumprod()
        results.append(max_drawdown(cumulative))
    return results


def cholesky_stress(returns_frame: pd.DataFrame, shock_correlation: float = 0.8) -> dict[str, float]:
    if returns_frame.shape[1] < 2:
        return {"stressed_joint_loss": float(-returns_frame.mean().abs().sum())}
    corr = returns_frame.corr().fillna(0.0).to_numpy()
    corr[:] = shock_correlation
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    rng = np.random.default_rng(11)
    independent = rng.normal(0, 1, size=(returns_frame.shape[1], 200))
    shocked = chol @ independent
    stressed_loss = float(np.percentile(shocked.mean(axis=0), 5))
    return {"stressed_joint_loss": stressed_loss}


def student_t_tail_simulation(
    returns: pd.Series,
    iterations: int = 300,
    degrees_of_freedom: int = 4,
) -> dict[str, float]:
    rng = np.random.default_rng(13)
    scale = float(returns.std(ddof=0) or 0.01)
    drift = float(returns.mean())
    simulated = drift + scale * rng.standard_t(df=degrees_of_freedom, size=iterations)
    var_95, cvar_95 = compute_var_cvar(pd.Series(simulated), alpha=0.95)
    ruin_probability = float((simulated < -0.1).mean())
    return {
        "student_t_var_95": var_95,
        "student_t_cvar_95": cvar_95,
        "probability_of_ruin": ruin_probability,
    }
