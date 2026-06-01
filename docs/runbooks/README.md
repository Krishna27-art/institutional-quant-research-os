# On-Call Rotation and Runbooks
# 90-Day Plan Item #10 - Priority: P0 (Critical)

## On-Call Rotation Schedule

### Schedule Format
- **Rotation**: Weekly (Monday 00:00 to Sunday 23:59)
- **Primary**: First responder for all alerts
- **Secondary**: Backup if primary doesn't acknowledge in 10 minutes
- **Escalation**: If secondary doesn't acknowledge in 20 minutes, page on-call manager

### Current Rotation
| Week | Primary | Secondary | On-Call Manager |
|------|---------|-----------|-----------------|
| Week 1 | Engineer A | Engineer B | CTO |
| Week 2 | Engineer B | Engineer C | CTO |
| Week 3 | Engineer C | Engineer A | CTO |
| Week 4 | Engineer A | Engineer B | CTO |

### Handoff Procedure
1. **Sunday 23:00**: Primary and secondary discuss ongoing issues
2. **Sunday 23:30**: Primary documents handoff in #on-call-handoff channel
3. **Sunday 23:45**: Secondary acknowledges handoff
4. **Monday 00:00**: New rotation begins

## Alert Severity Levels

### P0 - Critical (Page Immediately)
**Definition**: System down, trading stopped, data feed failure, regulatory breach

**Examples**:
- Circuit breaker hit
- Exchange disconnect
- Redis down
- >5% data missing
- VaR breach > 3% of AUM
- Leverage > 4x
- Compliance violation

**Response Time**: < 5 minutes
**Escalation**: If not acknowledged in 10 minutes → secondary

### P1 - Urgent (Slack + SMS)
**Definition**: Degraded performance, strategy failure, risk limit warning

**Examples**:
- Sharpe < 0 for 3 days
- VaR > 3% of AUM
- Leverage > 3.5x
- Feature drift detected
- Model age > 150 days
- Low fill rate < 50%

**Response Time**: < 15 minutes
**Escalation**: If not acknowledged in 30 minutes → secondary

### P2 - Informational (Slack Only)
**Definition**: Non-critical issues, monitoring alerts

**Examples**:
- Feature drift warning
- Model age approaching 150 days
- Slightly elevated latency
- Minor data quality issues

**Response Time**: < 1 hour
**Escalation**: None

## Runbooks

### Runbook: Circuit Breaker Hit
**Severity**: P0

**Symptoms**:
- Daily PnL < -3% of AUM
- Trading stopped
- Alert: "Circuit breaker activated"

**Diagnosis**:
1. Check current daily PnL: `SELECT daily_pnl FROM portfolio_metrics WHERE date = CURRENT_DATE`
2. Check which strategy caused the loss: `SELECT strategy, pnl FROM strategy_pnl WHERE date = CURRENT_DATE`
3. Check if this is a legitimate market event or system error

**Fix**:
1. If market event: Allow circuit breaker to cool down (auto-reactivation in 5 days)
2. If system error: Identify root cause, fix, reset circuit breaker
3. If data error: Validate data quality, fix data pipeline

**Rollback**:
- Manual override: Set `circuit_breaker_active = False` in risk engine
- Requires: Risk manager approval

**Prevention**:
- Review position sizing
- Check risk limits
- Validate data quality

---

### Runbook: Exchange Disconnect
**Severity**: P0

**Symptoms**:
- WebSocket connection lost
- No new ticks
- Alert: "Exchange connection failed"

**Diagnosis**:
1. Check broker API status: `curl https://api.kite.trade/status`
2. Check network connectivity: `ping exchange.com`
3. Check broker credentials

**Fix**:
1. If broker outage: Switch to backup broker
2. If network issue: Check network, restart connection
3. If credentials issue: Refresh tokens

**Rollback**:
- Revert to primary broker when restored

**Prevention**:
- Monitor broker API status
- Implement automatic failover
- Keep backup broker pre-funded

---

### Runbook: Redis Down
**Severity**: P0

**Symptoms**:
- Feature pipeline fails
- Cannot cache features
- Alert: "Redis connection failed"

**Diagnosis**:
1. Check Redis status: `redis-cli ping`
2. Check Redis logs: `docker logs quant-redis`
3. Check Redis memory: `redis-cli info memory`

**Fix**:
1. If Redis crashed: Restart Redis container
2. If memory full: Clear old keys, increase memory
3. If network issue: Check network

**Rollback**:
- Use fallback: compute features on-demand until Redis restored

**Prevention**:
- Monitor Redis memory usage
- Set up Redis Sentinel for failover
- Implement key expiration

---

### Runbook: Data Quality Failure
**Severity**: P0

**Symptoms**:
- >5% missing ticks
- Negative prices
- Stuck prices
- Alert: "Data quality check failed"

**Diagnosis**:
1. Run data validation: `python data/data_validation.py`
2. Check data source: broker API or exchange feed
3. Check data pipeline logs

**Fix**:
1. If broker API issue: Switch to backup data source
2. If pipeline issue: Fix pipeline, reprocess data
3. If exchange issue: Wait for exchange to fix

**Rollback**:
- Use last known good data
- Pause trading until data quality restored

**Prevention**:
- Implement multiple data sources
- Add data validation at every stage
- Monitor data quality metrics

---

### Runbook: Strategy Sharpe Drop
**Severity**: P1

**Symptoms**:
- Rolling 20-day Sharpe < 0 for 3 days
- Alert: "Strategy health warning"

**Diagnosis**:
1. Check strategy health: `python monitoring/strategy_health.py`
2. Review recent trades
3. Check if market regime changed

**Fix**:
1. If regime change: Adjust strategy parameters or deactivate
2. If strategy decay: Auto-deactivate after 10 days of negative Sharpe
3. If temporary issue: Monitor, no action

**Rollback**:
- Reactivate strategy after 30 days if conditions improve

**Prevention**:
- Monitor strategy health daily
- Implement regime detection
- Regular strategy review

---

### Runbook: VaR Breach
**Severity**: P1

**Symptoms**:
- VaR > 3% of AUM
- Alert: "VaR limit warning"

**Diagnosis**:
1. Check current VaR: `python risk/institutional_risk_engine.py`
2. Check position sizes
3. Check market volatility

**Fix**:
1. If position too large: Reduce position size
2. If volatility high: Reduce leverage
3. If correlation spike: Diversify positions

**Rollback**:
- Restore previous position sizes if needed

**Prevention**:
- Monitor VaR in real-time
- Implement dynamic position sizing
- Use liquidity-adjusted VaR

---

### Runbook: Feature Drift
**Severity**: P2

**Symptoms**:
- PSI > 0.2 for any feature
- Alert: "Feature drift detected"

**Diagnosis**:
1. Check feature distributions: `python features/feature_pipeline.py`
2. Compare to training baseline
3. Check if market structure changed

**Fix**:
1. If drift minor: Monitor, no action
2. If drift major: Retrain model with new data
3. If feature obsolete: Remove feature from model

**Rollback**:
- Use previous model version

**Prevention**:
- Monitor feature distributions daily
- Implement automatic retraining
- Regular feature review

---

## Emergency Contacts

| Role | Name | Phone | Email |
|------|------|-------|-------|
| CTO | John Doe | +91-98765-43210 | cto@quant-research.os |
| Head of Risk | Jane Smith | +91-98765-43211 | risk@quant-research.os |
| Head of Trading | Bob Johnson | +91-98765-43212 | trading@quant-research.os |
| DevOps Lead | Alice Brown | +91-98765-43213 | devops@quant-research.os |

## Escalation Matrix

| Time Since Alert | Primary | Secondary | Manager |
|------------------|---------|-----------|---------|
| 0-10 min | Acknowledge | Standby | Standby |
| 10-30 min | Working | Acknowledge | Standby |
| 30-60 min | Working | Working | Page |
| >60 min | Escalate | Escalate | Escalate |

## Post-Incident Review

After any P0 or P1 incident, conduct a post-incident review within 48 hours:

1. **Timeline**: What happened, when
2. **Root Cause**: Why it happened
3. **Impact**: What was affected
4. **Resolution**: How it was fixed
5. **Prevention**: How to prevent recurrence
6. **Action Items**: Specific tasks with owners

Document in #post-incident-reviews channel and update relevant runbooks.
