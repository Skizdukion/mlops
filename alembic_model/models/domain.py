from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel, Relationship
import uuid


class Prediction(SQLModel, table=True):
    """Prediction model for both database and API."""

    __tablename__ = "nyc_duration_inferences"
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    tpep_pickup_datetime: datetime
    passenger_count: int
    pulocationid: int
    dolocationid: int
    model_type: str
    predicted_duration: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship to Feedback
    feedback: Optional["Feedback"] = Relationship(
        sa_relationship_kwargs={"uselist": False, "cascade": "all, delete-orphan"},
        back_populates="prediction",
    )


class Feedback(SQLModel, table=True):
    """Feedback model for both database and API."""

    __tablename__ = "nyc_duration_feedback"

    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    prediction_id: uuid.UUID = Field(foreign_key="nyc_duration_inferences.id")
    dolocationid: int
    duration: float
    is_current_train: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship to Prediction
    prediction: Optional[Prediction] = Relationship(back_populates="feedback")
