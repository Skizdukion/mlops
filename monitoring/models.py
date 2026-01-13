from datetime import datetime
from typing import Optional, Dict, Any
from sqlmodel import Field, SQLModel
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB


class MonitoringMetrics(SQLModel, table=True):
    __tablename__ = "monitoring_metrics"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_type: str = Field(index=True)

    # Drift Metrics
    prediction_drift: Optional[float] = None
    target_drift: Optional[float] = None
    data_drift: Optional[float] = None

    # Performance Metrics
    rmse: Optional[float] = None
    r2: Optional[float] = None
    mae: Optional[float] = None

    # Drifted Columns Details (stored as JSONB)
    drifted_columns: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSONB)
    )

    # Metadata
    report_name: Optional[str] = None
