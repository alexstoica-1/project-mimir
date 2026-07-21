from sqlalchemy import text

from src.database.connection import engine


def test_connection() -> None:
    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT current_database(), current_user")
            )
            database, user = result.one()

        print(f"Connected to database: {database}")
        print(f"Connected as user: {user}")

    except Exception as exc:
        print(f"Database connection failed: {exc}")
        raise


if __name__ == "__main__":
    test_connection()