"""Create MIMIR database tables."""

from __future__ import annotations

import logging

from src.database.base import Base
from src.database.connection import engine
from src.database import models as _models

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    Base.metadata.create_all(bind=engine)
    logger.info("Created database tables: %s", sorted(Base.metadata.tables))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
