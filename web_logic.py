from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final, Iterable, Sequence

import numpy as np
import pandas as pd

from data_io import normalize_trainings, recompute_e1rm


TRAINING_DISPLAY_COLUMNS: Final[tuple[str, ...]] = ('exercise', 'e1rm', 'sets', 'weight', 'reps')
EXERCISE_REQUIRED_COLUMNS: Final[set[str]] = {'exercise', 'difficulty_coeff', 'sum'}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    column: str
    is_e1rm: bool = False


EXERCISE_METRICS: Final[tuple[MetricSpec, ...]] = (
    MetricSpec(key='raw_volume', title='Сырой объём', column='raw_volume'),
    MetricSpec(key='smart_volume', title='Умный объём', column='smart_volume'),
    MetricSpec(key='best_e1rm', title='e1RM лучшего сета за день', column='best_e1rm', is_e1rm=True),
)

MUSCLE_METRICS: Final[dict[str, str]] = {
    'e1rm': 'E1RM',
    'tonnage': 'Умный объём',
    'rawton': 'Сырой объём',
    'fail': 'Fail score',
}

MUSCLE_MODES: Final[dict[str, str]] = {
    'big': 'Большие группы',
    'small': 'Маленькие мышцы',
}


@dataclass(frozen=True)
class FrontierResult:
    table: pd.DataFrame
    source_rows: int
    unique_performances: int
    hidden_rows: int


@dataclass(frozen=True)
class TrainingRowInput:
    sets: float
    weight: float
    reps: float
    exercise: str
    date: pd.Timestamp


def clean_names(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != 'nan':
            result.append(text)
    return sorted(set(result))


def format_number(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return ''

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if np.isclose(number, round(number)):
        return str(int(round(number)))

    return f'{number:.{digits}f}'.rstrip('0').rstrip('.')


def format_date(value: object, fmt: str = '%Y-%m-%d') -> str:
    if pd.isna(value):
        return ''
    parsed = pd.to_datetime(value, errors='coerce')
    if pd.isna(parsed):
        return str(value)
    return pd.Timestamp(parsed).strftime(fmt)


def format_ddmmyyyy(value: object) -> str:
    return format_date(value, '%d%m%Y')


def parse_html_date(value: str | None, fallback: pd.Timestamp | None = None) -> pd.Timestamp:
    if value:
        parsed = pd.to_datetime(value, errors='coerce')
        if not pd.isna(parsed):
            return pd.Timestamp(parsed).normalize()
    if fallback is not None:
        return pd.Timestamp(fallback).normalize()
    return pd.Timestamp.today().normalize()


def html_date(value: object) -> str:
    return format_date(value, '%Y-%m-%d')


def date_values_from_frame(df: pd.DataFrame) -> list[pd.Timestamp]:
    if df.empty or 'date' not in df.columns:
        return []
    dates = pd.to_datetime(df['date'], errors='coerce').dropna().dt.normalize().unique()
    return [pd.Timestamp(date).normalize() for date in sorted(dates)]


def display_table_rows(df: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        item: dict[str, str] = {}
        for column in columns:
            if column == 'date':
                item[column] = format_date(row.get(column))
            else:
                item[column] = format_number(row.get(column))
        rows.append(item)
    return rows


def recompute_e1rm_one(weight: float, reps: float) -> float:
    if pd.isna(weight) or pd.isna(reps) or reps <= 0:
        return float('nan')
    return float(weight) * (1.0 + float(reps) / 30.0)


def parse_ids(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []

    parts = re.split(r'[,\s;]+', text)
    result: list[int] = []

    for part in parts:
        if not part:
            continue
        if '-' in part:
            a_raw, b_raw = part.split('-', 1)
            a = int(a_raw)
            b = int(b_raw)
            lo, hi = min(a, b), max(a, b)
            result.extend(range(lo, hi + 1))
        else:
            result.append(int(part))

    return sorted(set(result))


def norm_line(text: str) -> str:
    return text.strip().replace('×', 'x').replace('*', 'x').replace('X', 'x').replace(' ', '')


def parse_bulk_line(line: str) -> list[tuple[float, float, float]] | None:
    normalized = norm_line(line).lower().replace(',', '.')
    if not normalized:
        return None

    # 5x(60x5) -> 5 подходов, 60 кг, 5 повторов
    match = re.fullmatch(r'(\d+(?:\.\d+)?)x\((\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)\)', normalized)
    if match:
        return [(float(match.group(1)), float(match.group(2)), float(match.group(3)))]

    # 3x85x2 -> 3 подхода, 85 кг, 2 повтора
    match = re.fullmatch(r'(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)', normalized)
    if match:
        return [(float(match.group(1)), float(match.group(2)), float(match.group(3)))]

    # 80x3 -> 1 подход, 80 кг, 3 повтора
    match = re.fullmatch(r'(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)', normalized)
    if match:
        return [(1.0, float(match.group(1)), float(match.group(2)))]

    return None


def normalize_trainings_for_editing(trainings: pd.DataFrame) -> pd.DataFrame:
    df = normalize_trainings(trainings)
    if '_row_id' not in df.columns:
        df['_row_id'] = np.arange(len(df), dtype=int)
    df['_row_id'] = pd.to_numeric(df['_row_id'], errors='coerce').fillna(0).astype(int)
    return df


def day_rows(trainings: pd.DataFrame, date_value: pd.Timestamp) -> pd.DataFrame:
    df = normalize_trainings_for_editing(trainings)
    selected_date = pd.Timestamp(date_value).normalize()
    out = df[df['date'] == selected_date].copy().sort_values('_row_id', ascending=False)
    out['shown_id'] = np.arange(len(out), dtype=int)
    return out


def append_training_rows(trainings: pd.DataFrame, rows: Sequence[TrainingRowInput]) -> pd.DataFrame:
    df = normalize_trainings_for_editing(trainings)
    if not rows:
        return df

    next_id = 0 if df.empty else int(pd.to_numeric(df['_row_id'], errors='coerce').max()) + 1
    new_rows: list[dict[str, object]] = []

    for row in rows:
        exercise = row.exercise.strip()
        if not exercise:
            continue

        weight = float(row.weight)
        reps = float(row.reps)
        sets = float(row.sets)
        new_rows.append(
            {
                '_row_id': next_id,
                'exercise': exercise,
                'date': pd.Timestamp(row.date).normalize(),
                'sets': sets,
                'weight': weight,
                'reps': reps,
                'e1rm': recompute_e1rm_one(weight, reps),
            }
        )
        next_id += 1

    if not new_rows:
        return df

    updated = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    updated['e1rm'] = recompute_e1rm(updated['weight'], updated['reps'])
    updated = updated.sort_values(['date', 'exercise', 'weight', 'reps', '_row_id']).reset_index(drop=True)
    return updated


def delete_training_row_ids(trainings: pd.DataFrame, row_ids: Sequence[int]) -> tuple[pd.DataFrame, int]:
    df = normalize_trainings_for_editing(trainings)
    ids = {int(row_id) for row_id in row_ids}
    if not ids:
        return df, 0
    keep_mask = ~df['_row_id'].isin(ids)
    deleted_count = int((~keep_mask).sum())
    return df.loc[keep_mask].copy().reset_index(drop=True), deleted_count


def exercise_muscle_columns(exercises: pd.DataFrame) -> list[str]:
    return [column for column in exercises.columns if column not in EXERCISE_REQUIRED_COLUMNS]


def exercise_options_from_data(trainings: pd.DataFrame, exercises: pd.DataFrame) -> list[str]:
    values = []
    if 'exercise' in exercises.columns:
        values.extend(exercises['exercise'].tolist())
    if 'exercise' in trainings.columns:
        values.extend(trainings['exercise'].tolist())
    return clean_names(values)


def save_exercise_row(
    exercises: pd.DataFrame,
    *,
    name: str,
    difficulty_coeff: float,
    muscles: dict[str, float],
) -> tuple[pd.DataFrame, str]:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError('Название упражнения пустое.')

    df = exercises.copy()
    missing = EXERCISE_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f'В exercises.csv не хватает колонок: {sorted(missing)}')

    muscle_cols = exercise_muscle_columns(df)
    row_data: dict[str, object] = {'exercise': clean_name}
    for column in muscle_cols:
        row_data[column] = float(muscles.get(column, 0.0))

    row_data['sum'] = float(sum(float(row_data[column]) for column in muscle_cols))
    row_data['difficulty_coeff'] = float(difficulty_coeff)

    mask = df['exercise'].astype(str).str.strip().str.lower() == clean_name.lower()
    if mask.any():
        idx = df.index[mask][0]
        for key, value in row_data.items():
            df.loc[idx, key] = value
        action = 'обновлено'
    else:
        df.loc[len(df)] = row_data
        action = 'добавлено'

    return df, action


def _prepare_exercise_rows(trainings: pd.DataFrame, exercise_name: str) -> pd.DataFrame:
    df = normalize_trainings(trainings)
    df = df[df['exercise'].astype(str).str.strip() == exercise_name].copy()

    if df.empty:
        return df

    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df['reps'] = pd.to_numeric(df['reps'], errors='coerce')
    df['sets'] = pd.to_numeric(df.get('sets', 1), errors='coerce').fillna(1.0)
    df['e1rm'] = pd.to_numeric(df.get('e1rm'), errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()

    df = df.dropna(subset=['weight', 'reps', 'date'])
    df = df[(df['weight'] > 0) & (df['reps'] > 0)]

    return df


def _collapse_duplicate_performances(exercise_df: pd.DataFrame) -> pd.DataFrame:
    if exercise_df.empty:
        return exercise_df.copy()

    sort_columns = ['weight', 'reps', 'e1rm', 'date', 'sets']
    ascending = [False, False, False, False, False]
    if '_row_id' in exercise_df.columns:
        sort_columns.append('_row_id')
        ascending.append(False)

    sorted_df = exercise_df.sort_values(sort_columns, ascending=ascending, na_position='last')
    return sorted_df.drop_duplicates(['weight', 'reps'], keep='first').copy()


def _keep_only_nondominated_performances(unique_df: pd.DataFrame) -> pd.DataFrame:
    if unique_df.empty:
        return unique_df.copy()

    weights = unique_df['weight'].to_numpy(dtype=float)
    reps = unique_df['reps'].to_numpy(dtype=float)

    dominates = (
        (weights[:, None] >= weights[None, :])
        & (reps[:, None] >= reps[None, :])
        & ((weights[:, None] > weights[None, :]) | (reps[:, None] > reps[None, :]))
    )
    dominated_mask = dominates.any(axis=0)

    return unique_df.loc[~dominated_mask].copy()


def build_best_performances(trainings: pd.DataFrame, exercise_name: str) -> FrontierResult:
    display_columns = ('weight', 'reps', 'date', 'sets', 'e1rm')
    exercise_df = _prepare_exercise_rows(trainings, exercise_name)
    unique_df = _collapse_duplicate_performances(exercise_df)
    frontier_df = _keep_only_nondominated_performances(unique_df)

    if not frontier_df.empty:
        frontier_df = frontier_df.sort_values(
            ['reps', 'weight', 'date'],
            ascending=[True, False, False],
        )
        frontier_df = frontier_df.loc[:, display_columns].reset_index(drop=True)

    return FrontierResult(
        table=frontier_df,
        source_rows=len(exercise_df),
        unique_performances=len(unique_df),
        hidden_rows=len(unique_df) - len(frontier_df),
    )


def _difficulty_by_exercise(exercises: pd.DataFrame) -> pd.DataFrame:
    if 'exercise' not in exercises.columns:
        raise ValueError('В exercises.csv нет колонки exercise.')

    ex = exercises.copy()
    ex['exercise'] = ex['exercise'].astype(str).str.strip()

    if 'difficulty_coeff' not in ex.columns:
        ex['difficulty_coeff'] = 1.0
    else:
        ex['difficulty_coeff'] = pd.to_numeric(ex['difficulty_coeff'], errors='coerce').fillna(1.0)

    return ex[['exercise', 'difficulty_coeff']].drop_duplicates('exercise', keep='first')


def build_daily_exercise_metrics(trainings: pd.DataFrame, exercises: pd.DataFrame) -> pd.DataFrame:
    train = normalize_trainings(trainings)
    train = train.dropna(subset=['date', 'exercise']).copy()
    train['exercise'] = train['exercise'].astype(str).str.strip()
    train = train[train['exercise'].ne('') & train['exercise'].str.lower().ne('nan')]

    columns = ['date', 'exercise', 'raw_volume', 'smart_volume', 'best_e1rm']
    if train.empty:
        return pd.DataFrame(columns=columns)

    difficulty = _difficulty_by_exercise(exercises)
    train = train.merge(difficulty, on='exercise', how='left')
    train['difficulty_coeff'] = pd.to_numeric(train['difficulty_coeff'], errors='coerce').fillna(1.0)

    sets = pd.to_numeric(train['sets'], errors='coerce').fillna(1.0)
    weight = pd.to_numeric(train['weight'], errors='coerce').fillna(0.0)
    reps = pd.to_numeric(train['reps'], errors='coerce').fillna(0.0)
    e1rm = pd.to_numeric(train['e1rm'], errors='coerce')

    train['raw_volume'] = weight * reps * sets
    train['smart_volume'] = train['raw_volume'] * train['difficulty_coeff']
    train['best_e1rm_source'] = e1rm

    daily = (
        train.groupby(['date', 'exercise'], as_index=False)
        .agg(
            raw_volume=('raw_volume', 'sum'),
            smart_volume=('smart_volume', 'sum'),
            best_e1rm=('best_e1rm_source', 'max'),
        )
        .sort_values(['date', 'exercise'])
        .reset_index(drop=True)
    )

    return daily[columns]


def calendar_smooth(dates: pd.Series, values: pd.Series, radius_days: int = 3) -> pd.Series:
    if dates.empty:
        return pd.Series(dtype='float64')

    normalized_dates = pd.to_datetime(dates, errors='coerce').dt.normalize()
    numeric_values = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
    day_values = normalized_dates.map(pd.Timestamp.toordinal).to_numpy(dtype=float)

    smoothed = np.empty(len(values), dtype=float)
    for idx, current_day in enumerate(day_values):
        mask = np.abs(day_values - current_day) <= radius_days
        window = numeric_values[mask]
        smoothed[idx] = float(np.nanmean(window)) if np.isfinite(window).any() else np.nan

    return pd.Series(smoothed, index=values.index)


def build_combined_volume_dataframe(
    trainings: pd.DataFrame,
    exercises: pd.DataFrame,
    muscle_groups: pd.DataFrame,
) -> pd.DataFrame:
    train = normalize_trainings(trainings)
    ex = exercises.copy()
    mg = muscle_groups.copy()

    train = train.dropna(subset=['date', 'exercise']).copy()
    if train.empty:
        return pd.DataFrame(columns=['date'])

    muscle_cols = [c for c in ex.columns if c not in {'exercise', 'difficulty_coeff', 'sum'}]
    available_cols = ['exercise', 'difficulty_coeff'] + muscle_cols
    tmp = train.merge(ex[available_cols], on='exercise', how='left')
    tmp['difficulty_coeff'] = pd.to_numeric(tmp['difficulty_coeff'], errors='coerce').fillna(0.0)

    tmp['w'] = pd.to_numeric(tmp['weight'], errors='coerce').fillna(0.0)
    tmp['r'] = pd.to_numeric(tmp['reps'], errors='coerce').fillna(0.0)
    tmp['s'] = pd.to_numeric(tmp['sets'], errors='coerce').fillna(1.0)
    tmp['diff'] = tmp['difficulty_coeff'].fillna(0.0)
    tmp['e'] = pd.to_numeric(tmp['e1rm'], errors='coerce').fillna(0.0)

    coef_small = tmp[muscle_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    tmp['score_e1rm'] = tmp['e'] * tmp['diff'] * tmp['s']
    tmp['score_tonnage'] = tmp['w'] * tmp['r'] * tmp['diff'] * tmp['s']
    tmp['score_rawton'] = tmp['w'] * tmp['r'] * tmp['s']

    ex_max = tmp.groupby('exercise')['e1rm'].transform('max')
    intensity = pd.Series(0.0, index=tmp.index)
    valid_max = ex_max > 0
    intensity.loc[valid_max] = tmp.loc[valid_max, 'e1rm'] / ex_max.loc[valid_max]

    threshold = 0.80
    power = 4
    fail_core = tmp['s'] * (intensity.clip(lower=0.0) - threshold).clip(lower=0.0).pow(power)
    tmp['score_fail'] = fail_core * tmp['diff']

    def daily_small_from_score(score_col: str) -> pd.DataFrame:
        small = coef_small.mul(tmp[score_col], axis=0)
        return small.groupby(tmp['date']).sum()

    daily_small_e1rm = daily_small_from_score('score_e1rm')
    daily_small_ton = daily_small_from_score('score_tonnage')
    daily_small_raw = daily_small_from_score('score_rawton')
    daily_small_fail = daily_small_from_score('score_fail')

    if {'muscle_big', 'muscle_small'} - set(mg.columns):
        raise ValueError('В muscle_groups.csv должны быть колонки muscle_small и muscle_big.')

    big_to_small = mg.groupby('muscle_big')['muscle_small'].apply(list).to_dict()
    big_groups = sorted(big_to_small.keys())

    def small_to_big(daily_small: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=daily_small.index)
        for big_group in big_groups:
            cols = [c for c in big_to_small[big_group] if c in daily_small.columns]
            out[big_group] = daily_small[cols].sum(axis=1) if cols else 0.0
        return out

    daily_big_e1rm = small_to_big(daily_small_e1rm)
    daily_big_ton = small_to_big(daily_small_ton)
    daily_big_raw = small_to_big(daily_small_raw)
    daily_big_fail = small_to_big(daily_small_fail)

    start = train['date'].min()
    end = train['date'].max()
    all_days = pd.date_range(start=start, end=end, freq='D')

    def reindex_all(df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(all_days, fill_value=0.0)

    daily_small_e1rm = reindex_all(daily_small_e1rm)
    daily_small_ton = reindex_all(daily_small_ton)
    daily_small_raw = reindex_all(daily_small_raw)
    daily_small_fail = reindex_all(daily_small_fail)

    daily_big_e1rm = reindex_all(daily_big_e1rm)
    daily_big_ton = reindex_all(daily_big_ton)
    daily_big_raw = reindex_all(daily_big_raw)
    daily_big_fail = reindex_all(daily_big_fail)

    combined = pd.DataFrame(index=all_days)
    combined['date'] = all_days

    def make_block(metric_name: str, daily_big: pd.DataFrame, daily_small: pd.DataFrame) -> pd.DataFrame:
        big_block = daily_big.add_prefix(f'{metric_name}__big__')
        small_block = daily_small.add_prefix(f'{metric_name}__small__')
        return pd.concat([big_block, small_block], axis=1)

    combined = pd.concat(
        [
            combined,
            make_block('e1rm', daily_big_e1rm, daily_small_e1rm),
            make_block('tonnage', daily_big_ton, daily_small_ton),
            make_block('rawton', daily_big_raw, daily_small_raw),
            make_block('fail', daily_big_fail, daily_small_fail),
        ],
        axis=1,
    ).reset_index(drop=True)

    return combined
