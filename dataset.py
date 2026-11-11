import csv
from pathlib import Path

from pydantic import ValidationError

from schemas import DatasetRowChurn

DATASET_PATH = Path(__file__).resolve().parent / "data" / "churn_dataset.csv"


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    """Read the CSV and validate each row before storing it as a dictionary."""
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        expected = set(DatasetRowChurn.model_fields)
        if (
            reader.fieldnames is None
            or set(reader.fieldnames) != expected
            or len(reader.fieldnames) != len(expected)
        ):
            raise ValueError("CSV должен содержать все 10 столбцов DatasetRowChurn без повторений")
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Лишние значения в строке CSV {line_number}")
            row = {
                key: None if isinstance(value, str) and not value.strip() else value
                for key, value in row.items()
            }
            try:
                validated = DatasetRowChurn.model_validate(row)
            except ValidationError as error:
                raise ValueError(f"Некорректные данные в строке CSV {line_number}") from error
            if validated.churn not in (0, 1):
                raise ValueError(f"churn должен быть 0 или 1 в строке CSV {line_number}")
            rows.append(validated.model_dump())
    return rows


def dataset_info(rows: list[dict]) -> dict:
    columns = list(DatasetRowChurn.model_fields)
    return {
        "row_count": len(rows),
        "column_count": len(columns),
        "columns": columns,
        "feature_names": [column for column in columns if column != "churn"],
        "churn_distribution": {
            "0": sum(row["churn"] == 0 for row in rows),
            "1": sum(row["churn"] == 1 for row in rows),
        },
    }
