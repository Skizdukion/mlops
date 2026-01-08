from contextlib import contextmanager
from typing import Generator, Optional
import logging
from pathlib import Path
from sqlmodel import create_engine, Session, SQLModel, select, desc
from api_gateway.app.config import config
from api_gateway.app.models.domain import Prediction, Feedback
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)


class SqlLiteDatabase:
    def __init__(self):
        """Initialize database connection with SQLModel."""
        self.db_path = config.SQLITE_DB_PATH
        self._ensure_db_directory()

        # Create Engine with Connection Pooling
        # check_same_thread=False is needed for SQLite in multithreaded apps (FastAPI)
        self.engine = create_engine(
            self.db_path,
            connect_args={"check_same_thread": False},
            pool_size=20,  # Pool 20 connections
            max_overflow=10,  # Allow 10 more if pool is full
            pool_timeout=30,  # Wait 30s for a connection
            pool_recycle=1800,  # Recycle connections every 30 mins
        )

        # Create tables if they don't exist
        self._init_database()

    def _ensure_db_directory(self) -> None:
        """Ensure database directory exists."""
        # Extract path from sqlite:///path/to/db or just path/to/db
        if "sqlite:///" in self.db_path:
            file_path = self.db_path.replace("sqlite:///", "")
        else:
            file_path = self.db_path

        # Handle :memory: or relative paths carefully, but simplified here for typical file paths
        if file_path == ":memory:":
            return

        db_path = Path(file_path)
        parent_dir = db_path.parent.resolve()

        if parent_dir != Path(".").resolve() and not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Created directory for database: %s", parent_dir)
            except Exception as e:
                logger.error("Failed to create directory '%s': %s", parent_dir, e)
                raise

    def _init_database(self) -> None:
        """Initialize database tables."""
        try:
            SQLModel.metadata.create_all(self.engine)
            logger.info("Initialized database at %s", self.db_path)
        except Exception as e:
            logger.error("Failed to initialize database: %s", e)
            raise

    @contextmanager
    def get_db(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = Session(self.engine)
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_inference(self, prediction: Prediction) -> int:
        """Save prediction to database using SQLModel."""
        with self.get_db() as session:
            session.add(prediction)
            session.refresh(prediction)
            return prediction.id

    def save_feedback(self, feedback: Feedback):
        """Save feedback to database using SQLModel."""
        with self.get_db() as session:
            session.add(feedback)

    def get_prediction(self, prediction_id: int) -> Optional[Prediction]:
        """Get prediction by ID using SQLModel."""
        with self.get_db() as session:
            statement = select(Prediction).where(Prediction.id == prediction_id)
            result = session.exec(statement).first()
            return result

    def get_last_preds_with_feedback(self, row_count: int) -> list[Prediction]:
        with self.get_db() as session:
            statement = (
                select(Prediction)
                .where(Prediction.feedback is not None)  # Only works for one-to-one
                .options(selectinload(Prediction.feedback))
                .order_by(desc(Prediction.id))
                .limit(row_count)
            )
            results = session.exec(statement).all()

            return results


repo = SqlLiteDatabase()
