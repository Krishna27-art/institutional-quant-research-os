# Disaster Recovery Plan
# 90-Day Plan Item #6 - Priority: P0 (Critical)

## Overview

**RTO (Recovery Time Objective)**: 4 hours  
**RPO (Recovery Point Objective)**: 1 hour

## Architecture

### Multi-AZ Deployment
- **Primary AZ**: ap-south-1a (Mumbai)
- **Secondary AZ**: ap-south-1b (Mumbai)
- **DR Region**: ap-south-2 (Chennai) - for catastrophic events

### Backup Broker
- **Primary Broker**: Zerodha Kite
- **Backup Broker**: Upstox
- **Failover**: Automatic if primary API down for >5 minutes

### Kill Switch
- **Web UI**: https://kill-switch.quant-research.os
- **API**: POST /api/kill-switch with authentication
- **Effect**: Cancel all open orders, stop new orders

## Components

### 1. Database Backup

#### PostgreSQL
- **Backup Frequency**: Every 15 minutes (WAL) + daily full backup
- **Retention**: 7 years (regulatory requirement)
- **Storage**: S3 with lifecycle policy
- **Recovery**: Point-in-time recovery (PITR) to any WAL

```bash
# Daily backup
pg_dump -h postgres-primary -U quant_user quantdb | gzip > s3://backups/postgres/daily/$(date +%Y%m%d).sql.gz

# WAL backup (continuous)
pg_receivewal -h postgres-primary -D /var/lib/postgresql/wal
aws s3 sync /var/lib/postgresql/wal s3://backups/postgres/wal/
```

#### ClickHouse
- **Backup Frequency**: Hourly
- **Retention**: 5 years raw, 10 years aggregated
- **Storage**: S3
- **Recovery**: ClickHouse backup tool

```bash
# Hourly backup
clickhouse-backup create hourly
clickhouse-backup upload s3://backups/clickhouse/hourly
```

#### Redis
- **Backup Frequency**: Every 5 minutes (RDB)
- **Retention**: 7 days
- **Storage**: EBS snapshots
- **Recovery**: Redis Sentinel failover

### 2. Application State

#### Configuration
- **Storage**: Git repository (versioned)
- **Backup**: S3 bucket
- **Recovery**: Clone from Git

#### Code
- **Storage**: GitHub
- **Backup**: S3 bucket (daily sync)
- **Recovery**: Clone from GitHub

#### Logs
- **Storage**: Loki + S3
- **Retention**: 90 days hot, 7 years cold
- **Recovery**: Query from Loki

### 3. Data Pipeline

#### Raw Ticks
- **Storage**: Parquet on S3 (partitioned by symbol/year/month)
- **Backup**: Cross-region replication
- **Recovery**: Restore from S3

#### Processed Data
- **Storage**: ClickHouse
- **Backup**: ClickHouse backups
- **Recovery**: Restore from backup or reprocess from raw ticks

## Disaster Scenarios

### Scenario 1: Single AZ Failure
**Severity**: High  
**Likelihood**: Low  
**RTO**: 1 hour  
**RPO**: 15 minutes

**Impact**:
- All services in affected AZ unavailable
- Trading stops temporarily

**Recovery Steps**:
1. Detect AZ failure (CloudWatch alarm)
2. Activate secondary AZ (auto-scaling group)
3. Redirect DNS to secondary AZ
4. Verify all services healthy
5. Resume trading

**Prevention**:
- Multi-AZ deployment
- Auto-scaling across AZs
- DNS failover

---

### Scenario 2: Region Failure
**Severity**: Critical  
**Likelihood**: Very Low  
**RTO**: 4 hours  
**RPO**: 1 hour

**Impact**:
- Entire Mumbai region unavailable
- Complete trading halt

**Recovery Steps**:
1. Detect region failure (CloudWatch alarm)
2. Activate DR region (Chennai)
4. Restore databases from latest backup
5. Restore data from S3
6. Verify all services healthy
7. Resume trading

**Prevention**:
- Cross-region replication
- DR region on standby
- Regular DR drills

---

### Scenario 3: Primary Broker Failure
**Severity**: High  
**Likelihood**: Medium  
**RTO**: 10 minutes  
**RPO**: 0 minutes

**Impact**:
- Cannot place orders via primary broker
- Trading may stop if no failover

**Recovery Steps**:
1. Detect broker API failure (health check)
2. Activate backup broker (automatic)
3. Sync positions to backup broker
4. Resume trading via backup broker

**Prevention**:
- Dual broker setup
- Automatic failover
- Keep backup broker pre-funded

---

### Scenario 4: Database Corruption
**Severity**: Critical  
**Likelihood**: Low  
**RTO**: 2 hours  
**RPO**: 15 minutes

**Impact**:
- Cannot read/write data
- Trading stops

**Recovery Steps**:
1. Detect corruption (health checks)
2. Stop application writes
3. Restore from latest backup
4. Replay WAL logs to PITR
5. Verify data integrity
6. Resume application

**Prevention**:
- Regular backups
- WAL archiving
- Database health checks

---

### Scenario 5: Ransomware Attack
**Severity**: Critical  
**Likelihood**: Low  
**RTO**: 24 hours  
**RPO**: 1 hour

**Impact**:
- Data encrypted
- Systems unavailable

**Recovery Steps**:
1. Isolate affected systems
2. Activate DR environment
3. Restore from immutable backups
4. Scan for malware
5. Verify clean restore
6. Resume operations

**Prevention**:
- Immutable backups
- Regular security audits
- Network segmentation

---

## Kill Switch Procedure

### Manual Kill Switch
**URL**: https://kill-switch.quant-research.os  
**Authentication**: MFA required

**Steps**:
1. Login with MFA
2. Click "Emergency Stop"
3. Confirm action
4. System cancels all open orders
5. System stops accepting new orders
6. Alert sent to on-call team

### API Kill Switch
```bash
curl -X POST https://api.quant-research.os/api/kill-switch \
  -H "Authorization: Bearer <token>" \
  -H "X-MFA: <mfa-code>"
```

### Kill Switch Effects
- Cancel all pending orders
- Stop new order generation
- Close all positions (optional)
- Alert on-call team
- Log event in audit trail

### Post-Kill Switch Recovery
1. Investigate root cause
2. Fix issue
3. Verify system health
4. Manual approval to restart
5. Gradual restart of trading

## Testing and Drills

### Monthly Tests
- Backup restoration test
- Failover to secondary AZ
- Kill switch test

### Quarterly Drills
- Full DR region activation
- Broker failover drill
- Ransomware simulation

### Annual Review
- Update DR plan
- Review RTO/RPO
- Update contact information

## Communication Plan

### Internal Communication
- **P0 Incidents**: Page on-call team immediately
- **P1 Incidents**: Slack #incidents channel
- **P2 Incidents**: Email update

### External Communication
- **Trading Halt**: Notify exchange within 30 minutes
- **Data Breach**: Notify regulators within 72 hours
- **System Outage**: Notify clients if >1 hour

## Contact Information

| Role | Name | Phone | Email |
|------|------|-------|-------|
| DR Coordinator | John Doe | +91-98765-43210 | dr@quant-research.os |
| CTO | Jane Smith | +91-98765-43211 | cto@quant-research.os |
| Head of Risk | Bob Johnson | +91-98765-43212 | risk@quant-research.os |
| DevOps Lead | Alice Brown | +91-98765-43213 | devops@quant-research.os |
| Primary Broker Contact | Zerodha Support | +91-80-4719-2020 | support@zerodha.com |
| Backup Broker Contact | Upstox Support | +91-22-6123-4567 | support@upstox.com |

## Appendix

### Backup Locations
- **PostgreSQL**: s3://quant-backups/postgres/
- **ClickHouse**: s3://quant-backups/clickhouse/
- **Redis**: s3://quant-backups/redis/
- **Logs**: s3://quant-backups/logs/
- **Code**: s3://quant-backups/code/

### Recovery Commands
```bash
# Restore PostgreSQL
aws s3 cp s3://quant-backups/postgres/daily/20240101.sql.gz - | gunzip | psql -h postgres-recovery -U quant_user quantdb

# Restore ClickHouse
clickhouse-backup restore s3://quant-backups/clickhouse/hourly/20240101-1200

# Restore Redis
aws s3 cp s3://quant-backups/redis/latest.rdb /var/lib/redis/dump.rdb
redis-server /etc/redis/redis.conf
```

### Monitoring
- **CloudWatch**: AWS infrastructure
- **Prometheus**: Application metrics
- **Grafana**: Dashboards
- **PagerDuty**: Alerting
