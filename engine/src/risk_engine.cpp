#include "risk_engine.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <spdlog/spdlog.h>

namespace quant_core {

// Approximation of normal inverse CDF using Beasley-Springer-Moro algorithm
double RiskEngine::_normal_inverse_cdf(double p) {
    if (p <= 0.0 || p >= 1.0) {
        return 0.0; // Edge cases
    }
    
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                               -2.759285104469687e+02, 1.383577518672690e+02,
                               -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                               -1.556989798598866e+02, 6.680131188771972e+01,
                               -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                               -2.400758277161838e+00, -2.549732539343734e+00,
                                4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                               2.445134137142996e+00, 3.754408661907416e+00};
    
    const double q = std::min(p, 1.0 - p);
    double t, u;
    
    if (q > 0.02425) {
        // Rational approximation for central region
        t = q - 0.5;
        u = t * t;
        u = t * (((a[0] * u + a[1]) * u + a[2]) * u + a[3]) * u + a[4]) * u + a[5];
        u = u / ((((b[0] * u + b[1]) * u + b[2]) * u + b[3]) * u + b[4]) * u + 1.0);
    } else {
        // Rational approximation for tail region
        t = std::sqrt(-2.0 * std::log(q));
        u = (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) /
             ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0);
    }
    
    return (p > 0.5) ? -u : u;
}

RiskEngine::RiskEngine(const RiskConfig& config)
    : config_(config)
    , daily_pnl_(0.0)
    , portfolio_value_(config.capital)
    , peak_equity_(config.capital)
    , current_drawdown_(0.0)
    , circuit_breaker_active_(false)
    , circuit_breaker_recovery_days_(0) {
    
    // Initialize sector limits from config
    sector_limits_["BANKNIFTY"] = 0.30;
    sector_limits_["NIFTY"] = 0.30;
    sector_limits_["IT"] = 0.30;
    sector_limits_["PHARMA"] = 0.30;
    sector_limits_["AUTO"] = 0.30;
    sector_limits_["FMCG"] = 0.30;
    sector_limits_["ENERGY"] = 0.30;
    sector_limits_["METALS"] = 0.30;
    
    // Initialize ADV data
    adv_data_["NIFTY"] = 5e10;
    adv_data_["BANKNIFTY"] = 3e10;
    adv_data_["RELIANCE"] = 2e9;
    adv_data_["HDFCBANK"] = 1.5e9;
    adv_data_["INFY"] = 1e9;
    
    spdlog::info("RiskEngine initialized with capital: ₹{:.2f}", config.capital);
}

RiskCheckResult RiskEngine::pre_trade_check(
    const std::string& symbol,
    const std::string& sector,
    int64_t quantity,
    double price,
    int side,
    double regime_multiplier
) {
    RiskCheckResult result;
    result.adjusted_quantity = static_cast<double>(quantity);
    
    // 1. Position size limit
    double position_value = quantity * price;
    double max_position_value = config_.capital * config_.max_position_pct;
    
    if (position_value > max_position_value) {
        result.adjusted_quantity = max_position_value / price;
        result.warnings.push_back("Position size reduced due to limit");
    }
    
    // 2. Sector concentration limit
    double current_sector_exposure = sector_exposure_.count(sector) ? sector_exposure_.at(sector) : 0.0;
    double new_position_value = result.adjusted_quantity * price;
    double new_sector_exposure = current_sector_exposure + new_position_value;
    double max_sector_exposure = config_.capital * config_.max_sector_exposure_pct;
    
    if (new_sector_exposure > max_sector_exposure) {
        double allowed_sector_value = max_sector_exposure - current_sector_exposure;
        result.adjusted_quantity = std::min(result.adjusted_quantity, allowed_sector_value / price);
        result.warnings.push_back("Position size reduced due to sector concentration");
    }
    
    // 3. Daily loss limit
    double daily_loss_pct = (daily_pnl_ < 0) ? std::abs(daily_pnl_) / portfolio_value_ : 0.0;
    if (daily_loss_pct > config_.max_daily_loss_pct) {
        result.approved = false;
        result.reason = "Daily loss limit exceeded";
        return result;
    }
    
    // 4. Drawdown limit
    if (current_drawdown_ > config_.max_daily_loss_pct) {
        result.approved = false;
        result.reason = "Max drawdown exceeded";
        return result;
    }
    
    // 5. Circuit breaker check
    if (circuit_breaker_active_) {
        result.adjusted_quantity *= 0.5; // Reduce by 50% during recovery
        result.warnings.push_back("Position size reduced due to circuit breaker recovery");
    }
    
    // 6. Regime adjustment
    result.adjusted_quantity *= regime_multiplier;
    
    // 7. Portfolio VaR check
    if (!historical_returns_.empty()) {
        double portfolio_var = _calculate_var(historical_returns_);
        if (portfolio_var > config_.capital * config_.var_cap_pct) {
            result.adjusted_quantity *= 0.8;
            result.warnings.push_back("Position size reduced due to portfolio VaR limit");
        }
    }
    
    // Final check
    if (result.adjusted_quantity <= 0) {
        result.approved = false;
        result.reason = "Risk checks failed";
    } else if (result.adjusted_quantity < quantity * 0.5) {
        result.approved = true;
        result.reason = "Position size reduced";
    }
    
    return result;
}

RiskMetrics RiskEngine::calculate_risk_metrics(
    const std::vector<Position>& positions,
    const std::vector<double>& portfolio_returns,
    double daily_pnl,
    double weekly_pnl
) {
    RiskMetrics metrics;
    
    if (portfolio_returns.empty()) {
        return metrics;
    }
    
    // Calculate VaR with Cornish-Fisher
    metrics.var = _calculate_var(portfolio_returns, true);
    
    // Cap VaR at configured percentage of AUM
    metrics.var = std::min(metrics.var, config_.capital * config_.var_cap_pct);
    
    // Calculate CVaR
    metrics.cvar = _calculate_cvar(portfolio_returns);
    
    // Calculate Liquidity-adjusted VaR
    metrics.l_var = _calculate_liquidity_adjusted_var(positions, portfolio_returns, metrics.var);
    
    // Calculate volatility targeting multiplier
    metrics.vol_target_multiplier = _calculate_volatility_target_multiplier(portfolio_returns);
    
    // Calculate portfolio heat (simplified - would need correlation matrix in production)
    metrics.portfolio_heat = 0.0; // Placeholder - would compute from correlation matrix
    
    // Calculate tail risk
    metrics.tail_risk = _calculate_tail_risk(portfolio_returns);
    
    // Check circuit breaker
    auto [cb_triggered, cb_reason] = _check_circuit_breaker(daily_pnl, weekly_pnl);
    metrics.circuit_breaker_active = circuit_breaker_active_;
    metrics.circuit_breaker_recovery_days = circuit_breaker_recovery_days_;
    
    // PnL percentages
    metrics.daily_pnl_pct = daily_pnl / config_.capital;
    metrics.weekly_pnl_pct = weekly_pnl / config_.capital;
    
    return metrics;
}

double RiskEngine::calculate_position_size(
    double signal_strength,
    double win_rate,
    double avg_win,
    double avg_loss,
    double available_capital,
    double asset_volatility
) {
    // Kelly fraction
    double kelly = _calculate_kelly_fraction(0.001, win_rate, avg_win, avg_loss);
    
    // Conservative Kelly (15% of optimal)
    double kelly_conservative = kelly * 0.15;
    
    // Volatility targeting multiplier
    double vol_mult = config_.risk_target / std::max(asset_volatility, 0.05);
    vol_mult = std::min(2.0, vol_mult);
    vol_mult = std::max(0.5, vol_mult);
    
    // Calculate size
    double size = available_capital * kelly_conservative * vol_mult * std::abs(signal_strength);
    
    // Apply position limit
    size = std::min(size, available_capital * config_.max_position_pct);
    
    // Apply risk per trade limit
    double max_risk_size = available_capital * config_.risk_per_trade_pct;
    size = std::min(size, max_risk_size);
    
    // Minimum size
    size = std::max(size, available_capital * 0.001);
    
    return size;
}

void RiskEngine::update_position(const Position& position) {
    double position_value = std::abs(position.quantity) * position.current_price;
    std::string sector = position.sector.empty() ? "Unknown" : position.sector;
    
    // Update sector exposure
    if (positions_.count(position.symbol)) {
        const Position& old_pos = positions_.at(position.symbol);
        double old_value = std::abs(old_pos.quantity) * old_pos.current_price;
        std::string old_sector = old_pos.sector.empty() ? "Unknown" : old_pos.sector;
        sector_exposure_[old_sector] = sector_exposure_.count(old_sector) ? 
            sector_exposure_[old_sector] - old_value : 0.0;
    }
    
    // Store position
    positions_[position.symbol] = position;
    
    // Update sector exposure
    sector_exposure_[sector] = sector_exposure_.count(sector) ? 
        sector_exposure_[sector] + position_value : position_value;
}

void RiskEngine::close_position(const std::string& symbol, double closing_price) {
    if (!positions_.count(symbol)) {
        return;
    }
    
    const Position& pos = positions_.at(symbol);
    int64_t quantity = pos.quantity;
    std::string sector = pos.sector.empty() ? "Unknown" : pos.sector;
    double old_price = pos.entry_price;
    
    // Calculate PnL
    double pnl = 0.0;
    if (quantity > 0) {
        pnl = (closing_price - old_price) * quantity;
    } else {
        pnl = (old_price - closing_price) * std::abs(quantity);
    }
    
    // Update daily PnL
    daily_pnl_ += pnl;
    
    // Update portfolio value
    portfolio_value_ += pnl;
    
    // Update sector exposure
    double position_value = std::abs(quantity) * closing_price;
    sector_exposure_[sector] = sector_exposure_.count(sector) ? 
        sector_exposure_[sector] - position_value : 0.0;
    
    // Remove position
    positions_.erase(symbol);
    
    // Update drawdown
    if (portfolio_value_ > peak_equity_) {
        peak_equity_ = portfolio_value_;
        current_drawdown_ = 0.0;
    } else {
        current_drawdown_ = (peak_equity_ - portfolio_value_) / peak_equity_;
    }
    
    spdlog::info("Closed position {}: PnL=₹{:.2f}, Portfolio=₹{:.2f}, Drawdown={:.2%}",
                 symbol, pnl, portfolio_value_, current_drawdown_);
}

void RiskEngine::update_daily_returns(double portfolio_return) {
    historical_returns_.push_back(portfolio_return);
    
    // Keep only last 252 returns (1 year)
    if (historical_returns_.size() > 252) {
        historical_returns_.erase(historical_returns_.begin());
    }
}

void RiskEngine::reset_daily() {
    daily_pnl_ = 0.0;
    
    // Decrement circuit breaker recovery days
    if (circuit_breaker_recovery_days_ > 0) {
        circuit_breaker_recovery_days_--;
        if (circuit_breaker_recovery_days_ == 0) {
            circuit_breaker_active_ = false;
            spdlog::info("Circuit breaker recovery complete");
        }
    }
}

void RiskEngine::force_liquidate_all() {
    spdlog::warn("FORCE LIQUIDATING ALL POSITIONS");
    positions_.clear();
    sector_exposure_.clear();
}

std::vector<Position> RiskEngine::get_positions() const {
    std::vector<Position> result;
    result.reserve(positions_.size());
    for (const auto& [symbol, pos] : positions_) {
        result.push_back(pos);
    }
    return result;
}

void RiskEngine::update_config(const RiskConfig& config) {
    config_ = config;
    portfolio_value_ = config.capital;
    spdlog::info("RiskEngine config updated");
}

const RiskConfig& RiskEngine::get_config() const {
    return config_;
}

void RiskEngine::set_adv_data(const std::unordered_map<std::string, double>& adv_data) {
    adv_data_ = adv_data;
}

void RiskEngine::set_sector_limits(const std::unordered_map<std::string, double>& sector_limits) {
    sector_limits_ = sector_limits;
}

// Private helper methods

double RiskEngine::_calculate_var(const std::vector<double>& returns, bool use_cornish_fisher) {
    if (returns.size() < 20) {
        return 0.0;
    }
    
    // Calculate moments
    double mean = 0.0, variance = 0.0;
    for (double r : returns) {
        mean += r;
    }
    mean /= returns.size();
    
    for (double r : returns) {
        variance += (r - mean) * (r - mean);
    }
    variance /= (returns.size() - 1);
    double sigma = std::sqrt(variance);
    
    if (use_cornish_fisher && returns.size() > 3) {
        double skew = _calculate_skewness(returns);
        double kurt = _calculate_kurtosis(returns);
        
        // Cornish-Fisher expansion
        double z = _normal_inverse_cdf(config_.var_confidence);
        double z_cf = z + (z * z - 1.0) * skew / 6.0 + (z * z * z - 3.0 * z) * kurt / 24.0;
        
        return config_.capital * (mean - z_cf * sigma);
    } else {
        // Standard parametric VaR
        double z = _normal_inverse_cdf(config_.var_confidence);
        return config_.capital * (mean - z * sigma);
    }
}

double RiskEngine::_calculate_cvar(const std::vector<double>& returns) {
    if (returns.empty()) {
        return 0.0;
    }
    
    std::vector<double> sorted_returns = returns;
    std::sort(sorted_returns.begin(), sorted_returns.end());
    
    size_t idx = static_cast<size_t>(sorted_returns.size() * (1.0 - config_.cvar_confidence));
    idx = std::max(idx, static_cast<size_t>(1));
    
    double tail_mean = 0.0;
    for (size_t i = 0; i < idx; ++i) {
        tail_mean += sorted_returns[i];
    }
    tail_mean /= idx;
    
    return -config_.capital * tail_mean;
}

double RiskEngine::_calculate_liquidity_adjusted_var(
    const std::vector<Position>& positions,
    const std::vector<double>& returns,
    double base_var
) {
    double liquidity_adjustment = 0.0;
    
    for (const auto& pos : positions) {
        double position_value = std::abs(pos.quantity) * pos.current_price;
        double adv = adv_data_.count(pos.symbol) ? adv_data_.at(pos.symbol) : 1e9;
        
        liquidity_adjustment += position_value / adv;
    }
    
    return base_var * (1.0 + liquidity_adjustment);
}

double RiskEngine::_calculate_volatility_target_multiplier(const std::vector<double>& returns) {
    if (returns.size() < 20) {
        return 1.0;
    }
    
    double mean = 0.0;
    for (double r : returns) {
        mean += r;
    }
    mean /= returns.size();
    
    double variance = 0.0;
    for (double r : returns) {
        variance += (r - mean) * (r - mean);
    }
    variance /= (returns.size() - 1);
    
    double current_vol = std::sqrt(variance) * std::sqrt(252.0);
    current_vol = std::max(current_vol, 0.01);
    
    double vol_mult = config_.risk_target / current_vol;
    vol_mult = std::min(2.0, vol_mult);
    vol_mult = std::max(0.5, vol_mult);
    
    return vol_mult;
}

double RiskEngine::_calculate_kelly_fraction(double expected_return, double win_rate, double avg_win, double avg_loss) {
    if (avg_loss == 0.0) {
        return 0.0;
    }
    
    double b = avg_win / avg_loss; // Odds
    double p = win_rate;
    
    double kelly = (p * b - (1.0 - p)) / b;
    
    // Cap at 25%
    kelly = std::min(0.25, std::max(0.0, kelly));
    
    return kelly;
}

double RiskEngine::_calculate_portfolio_heat(
    const std::vector<Position>& positions,
    const std::vector<std::vector<double>>& returns_matrix
) {
    // Simplified - in production, compute actual correlation matrix
    // This is a placeholder that would need full correlation computation
    return 0.0;
}

double RiskEngine::_calculate_tail_risk(const std::vector<double>& returns, double percentile) {
    if (returns.empty()) {
        return 0.0;
    }
    
    std::vector<double> sorted_returns = returns;
    std::sort(sorted_returns.begin(), sorted_returns.end());
    
    size_t idx = static_cast<size_t>(sorted_returns.size() * percentile);
    idx = std::min(idx, sorted_returns.size() - 1);
    
    return -config_.capital * sorted_returns[idx];
}

std::pair<bool, std::string> RiskEngine::_check_circuit_breaker(double daily_pnl, double weekly_pnl) {
    double daily_loss_pct = (daily_pnl < 0) ? std::abs(daily_pnl) / config_.capital : 0.0;
    double weekly_loss_pct = (weekly_pnl < 0) ? std::abs(weekly_pnl) / config_.capital : 0.0;
    
    if (daily_loss_pct > config_.max_daily_loss_pct) {
        circuit_breaker_active_ = true;
        circuit_breaker_recovery_days_ = 5;
        return {true, "Daily circuit breaker triggered"};
    }
    
    if (weekly_loss_pct > config_.max_weekly_loss_pct) {
        circuit_breaker_active_ = true;
        circuit_breaker_recovery_days_ = 10;
        return {true, "Weekly circuit breaker triggered"};
    }
    
    return {false, ""};
}

double RiskEngine::_calculate_moment(const std::vector<double>& returns, int moment) {
    if (returns.size() < 2) {
        return 0.0;
    }
    
    double mean = 0.0;
    for (double r : returns) {
        mean += r;
    }
    mean /= returns.size();
    
    double m = 0.0;
    for (double r : returns) {
        double diff = r - mean;
        double power = diff;
        for (int i = 1; i < moment; ++i) {
            power *= diff;
        }
        m += power;
    }
    
    return m / returns.size();
}

double RiskEngine::_calculate_skewness(const std::vector<double>& returns) {
    if (returns.size() < 3) {
        return 0.0;
    }
    
    double mean = 0.0;
    for (double r : returns) {
        mean += r;
    }
    mean /= returns.size();
    
    double variance = 0.0;
    for (double r : returns) {
        variance += (r - mean) * (r - mean);
    }
    variance /= returns.size();
    double sigma = std::sqrt(variance);
    
    if (sigma == 0.0) {
        return 0.0;
    }
    
    double skew = 0.0;
    for (double r : returns) {
        double diff = r - mean;
        skew += diff * diff * diff;
    }
    skew /= returns.size();
    skew /= (sigma * sigma * sigma);
    
    return skew;
}

double RiskEngine::_calculate_kurtosis(const std::vector<double>& returns) {
    if (returns.size() < 4) {
        return 0.0;
    }
    
    double mean = 0.0;
    for (double r : returns) {
        mean += r;
    }
    mean /= returns.size();
    
    double variance = 0.0;
    for (double r : returns) {
        variance += (r - mean) * (r - mean);
    }
    variance /= returns.size();
    double sigma = std::sqrt(variance);
    
    if (sigma == 0.0) {
        return 0.0;
    }
    
    double kurt = 0.0;
    for (double r : returns) {
        double diff = r - mean;
        kurt += diff * diff * diff * diff;
    }
    kurt /= returns.size();
    kurt /= (sigma * sigma * sigma * sigma);
    
    // Excess kurtosis (subtract 3 for normal distribution)
    return kurt - 3.0;
}

} // namespace quant_core
