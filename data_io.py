from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = PROJECT_ROOT / 'data'
OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / 'outputs'

EXERCISES_CSV: Final[Path] = DATA_DIR / 'exercises.csv'
TRAININGS_CSV: Final[Path] = DATA_DIR / 'trainings.csv'
MUSCLE_GROUPS_CSV: Final[Path] = DATA_DIR / 'muscle_groups.csv'

CSV_ENCODING: Final[str] = 'utf-8-sig'


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f'Не найден файл: {path}. '\
            'Положи CSV в папку data/ рядом со скриптами.'
        )
    return pd.read_csv(path, encoding=CSV_ENCODING)


def load_trainings(path: Path | str = TRAININGS_CSV) -> pd.DataFrame:
    return _read_csv(Path(path))


def load_exercises(path: Path | str = EXERCISES_CSV) -> pd.DataFrame:
    return _read_csv(Path(path))


def load_muscle_groups(path: Path | str = MUSCLE_GROUPS_CSV) -> pd.DataFrame:
    return _read_csv(Path(path))


def save_trainings(df: pd.DataFrame, path: Path | str = TRAININGS_CSV) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding=CSV_ENCODING)
    return output_path


def save_exercises(df: pd.DataFrame, path: Path | str = EXERCISES_CSV) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding=CSV_ENCODING)
    return output_path


def save_dataframe(df: pd.DataFrame, path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding=CSV_ENCODING)
    return output_path


def recompute_e1rm(weight: pd.Series, reps: pd.Series) -> pd.Series:
    weight_num = pd.to_numeric(weight, errors='coerce')
    reps_num = pd.to_numeric(reps, errors='coerce')
    return pd.Series(
        np.where(
            weight_num.notna() & reps_num.notna() & (reps_num > 0),
            weight_num * (1 + reps_num / 30.0),
            np.nan,
        ),
        index=weight.index,
        dtype='float64',
    )


def normalize_trainings(trainings: pd.DataFrame) -> pd.DataFrame:
    df = trainings.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df['sets'] = pd.to_numeric(df.get('sets', 1), errors='coerce').fillna(1.0)
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df['reps'] = pd.to_numeric(df['reps'], errors='coerce')
    df['exercise'] = df['exercise'].astype(str).str.strip()

    if '_row_id' not in df.columns:
        df['_row_id'] = np.arange(len(df), dtype=int)
    else:
        fallback = pd.Series(np.arange(len(df)), index=df.index)
        df['_row_id'] = pd.to_numeric(df['_row_id'], errors='coerce').fillna(fallback).astype(int)

    if 'e1rm' not in df.columns:
        df['e1rm'] = recompute_e1rm(df['weight'], df['reps'])
    else:
        df['e1rm'] = pd.to_numeric(df['e1rm'], errors='coerce')
        missing_mask = (
            df['e1rm'].isna()
            & df['weight'].notna()
            & df['reps'].notna()
            & (df['reps'] > 0)
        )
        df.loc[missing_mask, 'e1rm'] = recompute_e1rm(
            df.loc[missing_mask, 'weight'],
            df.loc[missing_mask, 'reps'],
        )

    return df
