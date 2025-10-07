# fund_backend.py
from dataclasses import dataclass
from typing import List
import math

@dataclass
class FundConfig:
    fund_size: float = 600_000_000.0
    investment_period_years: int = 4
    fund_life_years: int = 8
    mgmt_fee_rate_investment: float = 0.02  # 2%
    mgmt_fee_rate_post: float = 0.015       # 1.5%

@dataclass
class Deal:
    invested_amount: float
    invest_year: int
    exit_year: int
    moic: float

    def exit_proceeds(self) -> float:
        return self.invested_amount * self.moic

def validate_deals(deals: List[Deal], cfg: FundConfig) -> List[str]:
    errors = []
    for i, d in enumerate(deals, start=1):
        if d.invest_year < 1 or d.invest_year > cfg.investment_period_years:
            errors.append(f"Deal {i}: invest_year {d.invest_year} must be in 1..{cfg.investment_period_years}.")
        if d.exit_year < 1 or d.exit_year > cfg.fund_life_years:
            errors.append(f"Deal {i}: exit_year {d.exit_year} must be in 1..{cfg.fund_life_years}.")
        if d.exit_year < d.invest_year:
            errors.append(f"Deal {i}: exit_year {d.exit_year} cannot be before invest_year {d.invest_year}.")
        if d.invested_amount < 0:
            errors.append(f"Deal {i}: invested_amount must be non-negative.")
        if d.moic < 0:
            errors.append(f"Deal {i}: MOIC must be non-negative.")
    return errors

def invested_outstanding_by_year(deals: List[Deal], cfg: FundConfig) -> List[float]:
    outstanding = [0.0] * cfg.fund_life_years
    for t in range(1, cfg.fund_life_years + 1):
        total = 0.0
        for d in deals:
            if d.invest_year <= t and d.exit_year >= t:
                total += d.invested_amount
        outstanding[t - 1] = total
    return outstanding

def investment_calls_by_year(deals: List[Deal], cfg: FundConfig) -> List[float]:
    calls = [0.0] * cfg.fund_life_years
    for d in deals:
        calls[d.invest_year - 1] -= d.invested_amount
    return calls

def management_fees_by_year(deals: List[Deal], cfg: FundConfig) -> List[float]:
    fees = [0.0] * cfg.fund_life_years
    outstanding = invested_outstanding_by_year(deals, cfg)
    for t in range(1, cfg.fund_life_years + 1):
        if t <= cfg.investment_period_years:
            fee = -cfg.mgmt_fee_rate_investment * cfg.fund_size
        else:
            fee = -cfg.mgmt_fee_rate_post * outstanding[t - 1]
        fees[t - 1] = fee
    return fees

def exits_by_year(deals: List[Deal], cfg: FundConfig) -> List[float]:
    exits = [0.0] * cfg.fund_life_years
    for d in deals:
        exits[d.exit_year - 1] += d.exit_proceeds()
    return exits

def annual_cash_flows(deals: List[Deal], cfg: FundConfig) -> List[float]:
    calls = investment_calls_by_year(deals, cfg)
    fees  = management_fees_by_year(deals, cfg)
    ex    = exits_by_year(deals, cfg)
    return [calls[i] + fees[i] + ex[i] for i in range(cfg.fund_life_years)]

def irr(cashflows: List[float], guess: float = 0.15, max_iter: int = 100, tol: float = 1e-7) -> float:
    if all(cf >= 0 for cf in cashflows) or all(cf <= 0 for cf in cashflows):
        raise ValueError("IRR undefined: cash flows do not have at least one sign change.")
    r = guess
    for _ in range(max_iter):
        npv = 0.0
        d_npv = 0.0
        for t, cf in enumerate(cashflows, start=1):
            denom = (1 + r) ** t
            npv += cf / denom
            d_npv += -t * cf / denom / (1 + r)
        if abs(d_npv) < 1e-12:
            break
        new_r = r - npv / d_npv
        if new_r <= -0.999999:
            new_r = (r - 0.5) if r > -0.5 else (r / 2)
        if abs(new_r - r) < tol:
            return new_r
        r = new_r
    lo, hi = -0.999, 10.0
    def npv_at(rate):
        return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows, start=1))
    f_lo, f_hi = npv_at(lo), npv_at(hi)
    for _ in range(60):
        if f_lo * f_hi <= 0:
            break
        hi *= 0.5
        f_hi = npv_at(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = npv_at(mid)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    raise ValueError("IRR failed to converge after bisection.")