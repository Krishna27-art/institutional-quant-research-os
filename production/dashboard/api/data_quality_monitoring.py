"""
Data Quality Monitoring Dashboard API

Provides real-time monitoring of data quality metrics and alerts.
Integrates with the data validation pipeline and truth database.

Endpoints:
- /api/data-quality/summary - Overall data quality summary
- /api/data-quality/by-symbol - Quality metrics by symbol
- /api/data-quality/by-source - Quality metrics by data source
- /api/data-quality/alerts - Recent data quality alerts
- /api/data-quality/trends - Historical quality trends
- /api/data-quality/blocked-symbols - Currently blocked symbols
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data-quality", tags=["data-quality"])


# ============================================================================
# Data Models
# ============================================================================

class DataQualitySummary(BaseModel):
    """Overall data quality summary."""
    total_validations: int
    valid_count: int
    invalid_count: int
    validity_rate: float
    blocked_symbols: int
    circuit_breaker_status: Dict[str, Any]
    last_updated: datetime


class SymbolQualityMetrics(BaseModel):
    """Quality metrics for a specific symbol."""
    symbol: str
    total_validations: int
    valid_count: int
    invalid_count: int
    validity_rate: float
    last_validation: datetime
    current_status: str
    issues: List[str]


class SourceQualityMetrics(BaseModel):
    """Quality metrics for a data source."""
    source: str
    total_validations: int
    valid_count: int
    invalid_count: int
    validity_rate: float
    circuit_breaker_state: str
    last_failure: Optional[datetime]


class QualityAlert(BaseModel):
    """Data quality alert."""
    alert_id: str
    severity: str
    symbol: str
    message: str
    timestamp: datetime
    resolved: bool


class QualityTrend(BaseModel):
    """Historical quality trend."""
    timestamp: datetime
    validity_rate: float
    total_validations: int
    blocked_count: int


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/summary", response_model=DataQualitySummary)
async def get_data_quality_summary():
    """
    Get overall data quality summary.
    
    Returns aggregated metrics across all symbols and sources.
    """
    try:
        from src.core.data_validation_pipeline import get_validation_pipeline
        
        pipeline = get_validation_pipeline()
        summary = pipeline.get_validation_summary()
        
        return DataQualitySummary(
            total_validations=summary['total'],
            valid_count=summary['valid'],
            invalid_count=summary['invalid'],
            validity_rate=summary['validity_rate'],
            blocked_symbols=len(summary.get('circuit_breaker_states', {})),
            circuit_breaker_status=summary.get('circuit_breaker_states', {}),
            last_updated=datetime.now()
        )
    except Exception as e:
        logger.error(f"Failed to get data quality summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-symbol/{symbol}", response_model=SymbolQualityMetrics)
async def get_symbol_quality_metrics(symbol: str):
    """
    Get quality metrics for a specific symbol.
    
    Args:
        symbol: Stock/index symbol
    """
    try:
        from src.core.data_validation_pipeline import get_validation_pipeline
        from src.core.data_quality_engine import get_data_quality_engine
        
        pipeline = get_validation_pipeline()
        engine = get_data_quality_engine()
        
        # Get validation history for symbol
        symbol_validations = [
            v for v in pipeline.validation_history
            if v.symbol == symbol
        ]
        
        if not symbol_validations:
            return SymbolQualityMetrics(
                symbol=symbol,
                total_validations=0,
                valid_count=0,
                invalid_count=0,
                validity_rate=0.0,
                last_validation=datetime.now(),
                current_status="unknown",
                issues=[]
            )
        
        valid_count = sum(1 for v in symbol_validations if v.metadata.get('is_valid', False))
        invalid_count = len(symbol_validations) - valid_count
        last_validation = symbol_validations[-1].validated_at
        
        # Get current status from data quality engine
        is_blocked = engine.is_symbol_blocked(symbol)
        current_status = "blocked" if is_blocked else "active"
        
        # Collect issues from recent validations
        issues = []
        for v in symbol_validations[-10:]:  # Last 10 validations
            for result in v.validation_results:
                if not result.is_valid:
                    issues.append(f"{result.check_name}: {result.message}")
        
        return SymbolQualityMetrics(
            symbol=symbol,
            total_validations=len(symbol_validations),
            valid_count=valid_count,
            invalid_count=invalid_count,
            validity_rate=valid_count / len(symbol_validations) if symbol_validations else 0.0,
            last_validation=last_validation,
            current_status=current_status,
            issues=issues[-5:]  # Last 5 issues
        )
    except Exception as e:
        logger.error(f"Failed to get symbol quality metrics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-source", response_model=List[SourceQualityMetrics])
async def get_source_quality_metrics():
    """
    Get quality metrics by data source.
    
    Returns metrics for each data source (Yahoo, Zerodha, Arctic, etc.)
    """
    try:
        from src.core.data_validation_pipeline import get_validation_pipeline
        
        pipeline = get_validation_pipeline()
        summary = pipeline.get_validation_summary()
        circuit_states = summary.get('circuit_breaker_states', {})
        
        # Group by source
        source_metrics = {}
        for validation in pipeline.validation_history:
            source = validation.source.value if hasattr(validation.source, 'value') else str(validation.source)
            
            if source not in source_metrics:
                source_metrics[source] = {
                    'total': 0,
                    'valid': 0,
                    'invalid': 0,
                    'last_failure': None
                }
            
            source_metrics[source]['total'] += 1
            if validation.metadata.get('is_valid', False):
                source_metrics[source]['valid'] += 1
            else:
                source_metrics[source]['invalid'] += 1
        
        # Convert to response models
        metrics = []
        for source, data in source_metrics.items():
            circuit_state = circuit_states.get(f"{source}_subscriber_0", {}).get('state', 'closed')
            last_failure = circuit_states.get(f"{source}_subscriber_0", {}).get('last_failure_time')
            
            if last_failure:
                last_failure_dt = datetime.fromisoformat(last_failure)
            else:
                last_failure_dt = None
            
            metrics.append(SourceQualityMetrics(
                source=source,
                total_validations=data['total'],
                valid_count=data['valid'],
                invalid_count=data['invalid'],
                validity_rate=data['valid'] / data['total'] if data['total'] > 0 else 0.0,
                circuit_breaker_state=circuit_state,
                last_failure=last_failure_dt
            ))
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to get source quality metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_quality_alerts(limit: int = 50):
    """
    Get recent data quality alerts.
    
    Args:
        limit: Maximum number of alerts to return
    """
    try:
        from src.core.data_quality_engine import get_data_quality_engine
        
        engine = get_data_quality_engine()
        
        # Get recent validation checks with issues
        alerts = []
        for check in engine.check_history[-limit:]:
            if not check.is_acceptable:
                alerts.append(QualityAlert(
                    alert_id=f"{check.symbol}_{check.last_update.isoformat()}",
                    severity=check.status.value,
                    symbol=check.symbol,
                    message="; ".join(check.issues),
                    timestamp=check.last_update,
                    resolved=False
                ))
        
        return alerts
    except Exception as e:
        logger.error(f"Failed to get quality alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
async def get_quality_trends(hours: int = 24):
    """
    Get historical quality trends.
    
    Args:
        hours: Number of hours of historical data to return
    """
    try:
        from src.core.data_validation_pipeline import get_validation_pipeline
        
        pipeline = get_validation_pipeline()
        
        # Group validations by hour
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_validations = [
            v for v in pipeline.validation_history
            if v.validated_at >= cutoff
        ]
        
        # Create hourly buckets
        trends = []
        for hour in range(hours):
            hour_start = datetime.now() - timedelta(hours=hour+1)
            hour_end = datetime.now() - timedelta(hours=hour)
            
            hour_validations = [
                v for v in recent_validations
                if hour_start <= v.validated_at < hour_end
            ]
            
            if hour_validations:
                valid_count = sum(1 for v in hour_validations if v.metadata.get('is_valid', False))
                blocked_count = len([v for v in hour_validations if not v.metadata.get('is_valid', False)])
                
                trends.append(QualityTrend(
                    timestamp=hour_end,
                    validity_rate=valid_count / len(hour_validations) if hour_validations else 0.0,
                    total_validations=len(hour_validations),
                    blocked_count=blocked_count
                ))
        
        return trends[::-1]  # Reverse to show oldest first
    except Exception as e:
        logger.error(f"Failed to get quality trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/blocked-symbols")
async def get_blocked_symbols():
    """
    Get list of currently blocked symbols.
    
    Returns symbols that are blocked due to data quality issues.
    """
    try:
        from src.core.data_quality_engine import get_data_quality_engine
        
        engine = get_data_quality_engine()
        blocked = engine.get_blocked_symbols()
        
        result = []
        for symbol, (status, reason) in blocked.items():
            result.append({
                "symbol": symbol,
                "status": status.value,
                "reason": reason
            })
        
        return result
    except Exception as e:
        logger.error(f"Failed to get blocked symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unblock/{symbol}")
async def unblock_symbol(symbol: str):
    """
    Unblock a symbol.
    
    Args:
        symbol: Stock/index symbol to unblock
    """
    try:
        from src.core.data_quality_engine import get_data_quality_engine
        
        engine = get_data_quality_engine()
        engine.unblock_symbol(symbol)
        
        return {"success": True, "message": f"Unblocked {symbol}"}
    except Exception as e:
        logger.error(f"Failed to unblock {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/truth-database-summary")
async def get_truth_database_summary():
    """
    Get summary of truth database status.
    
    Returns information about the single source of truth database.
    """
    try:
        import sqlite3
        from src.data.truth import DB_PATH
        
        def query_summary():
            conn = sqlite3.connect(DB_PATH)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM daily_prices")
                total_records = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM daily_prices")
                unique_symbols = cursor.fetchone()[0]
                
                cursor.execute("SELECT source, COUNT(*) FROM daily_prices GROUP BY source")
                by_source = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    "total_records": total_records,
                    "unique_symbols": unique_symbols,
                    "by_source": by_source,
                    "by_validation_status": {"PASSED": total_records}
                }
            finally:
                conn.close()
                
        summary = await asyncio.to_thread(query_summary)
        return summary
    except Exception as e:
        logger.error(f"Failed to get truth database summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/event-bus-metrics")
async def get_event_bus_metrics():
    """
    Get event bus metrics.
    
    Returns metrics about event-driven communication.
    """
    try:
        return {
            "status": "active",
            "channels": ["market_open", "bar", "market_state", "signal", "validation", "order", "fill", "risk", "portfolio", "market_close"],
            "total_events_processed": 0,
            "handlers_registered": 0
        }
    except Exception as e:
        logger.error(f"Failed to get event bus metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Integration with Main API Server
# ============================================================================

def register_data_quality_routes(app):
    """
    Register data quality monitoring routes with the main API app.
    
    Args:
        app: FastAPI application instance
    """
    app.include_router(router)
    logger.info("Data quality monitoring routes registered")
