from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk
from typing import Iterable, Final

import numpy as np
import pandas as pd

from data_io import ensure_directories, load_trainings, normalize_trainings


DISPLAY_COLUMNS: Final[tuple[str, ...]] = ('weight', 'reps', 'date', 'sets', 'e1rm')
TREE_COLUMNS: Final[tuple[str, ...]] = ('weight', 'reps', 'date', 'sets', 'e1rm')


@dataclass(frozen=True)
class FrontierResult:
    table: pd.DataFrame
    source_rows: int
    unique_performances: int
    hidden_rows: int


def _clean_exercise_names(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != 'nan':
            result.append(text)
    return sorted(set(result))


def _format_number(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return ''

    number = float(value)
    if np.isclose(number, round(number)):
        return str(int(round(number)))

    formatted = f'{number:.{digits}f}'.rstrip('0').rstrip('.')
    return formatted.replace('.', ',')


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
    """Return rows that are not beaten by another row in both weight and reps.

    A row is hidden when there is another row with weight >= current weight and
    reps >= current reps, with at least one of these two values strictly better.
    """
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
    exercise_df = _prepare_exercise_rows(trainings, exercise_name)
    unique_df = _collapse_duplicate_performances(exercise_df)
    frontier_df = _keep_only_nondominated_performances(unique_df)

    if not frontier_df.empty:
        frontier_df = frontier_df.sort_values(
            ['reps', 'weight', 'date'],
            ascending=[True, False, False],
        )
        frontier_df = frontier_df.loc[:, DISPLAY_COLUMNS].reset_index(drop=True)

    return FrontierResult(
        table=frontier_df,
        source_rows=len(exercise_df),
        unique_performances=len(unique_df),
        hidden_rows=len(unique_df) - len(frontier_df),
    )


class ExerciseProgressWindow:
    def __init__(self, trainings: pd.DataFrame) -> None:
        self.trainings = normalize_trainings(trainings)
        self.exercises = _clean_exercise_names(self.trainings.get('exercise', pd.Series(dtype=object)))
        self.current_table = pd.DataFrame(columns=DISPLAY_COLUMNS)

        self.root = tk.Tk()
        self.root.title('MyTrainings — лучшие подходы по упражнению')
        self.root.geometry('760x540')
        self.root.minsize(620, 420)

        self.exercise_name = tk.StringVar(value=self.exercises[0] if self.exercises else '')
        self.status = tk.StringVar(value='')

        self._build_ui()
        self._refresh_table()

    @classmethod
    def from_csv(cls) -> ExerciseProgressWindow:
        ensure_directories()
        return cls(load_trainings())

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.grid(row=0, column=0, sticky='ew')
        top_frame.columnconfigure(1, weight=1)

        ttk.Label(top_frame, text='Упражнение:').grid(row=0, column=0, sticky='w')

        self.exercise_combo = ttk.Combobox(
            top_frame,
            textvariable=self.exercise_name,
            values=self.exercises,
            state='readonly' if self.exercises else 'disabled',
        )
        self.exercise_combo.grid(row=0, column=1, sticky='ew', padx=(8, 0))
        self.exercise_combo.bind('<<ComboboxSelected>>', lambda _: self._refresh_table())

        ttk.Button(top_frame, text='Обновить', command=self._refresh_from_csv).grid(
            row=0, column=2, sticky='ew', padx=(8, 0)
        )
        ttk.Button(top_frame, text='Копировать', command=self._copy_table_to_clipboard).grid(
            row=0, column=3, sticky='ew', padx=(8, 0)
        )

        table_frame = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        table_frame.grid(row=1, column=0, sticky='nsew')
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            table_frame,
            columns=TREE_COLUMNS,
            show='headings',
            selectmode='browse',
        )
        self.tree.grid(row=0, column=0, sticky='nsew')

        y_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        y_scroll.grid(row=0, column=1, sticky='ns')
        self.tree.configure(yscrollcommand=y_scroll.set)

        headings = {
            'weight': 'Вес',
            'reps': 'Повторы',
            'date': 'Дата',
            'sets': 'Подходы',
            'e1rm': 'e1RM',
        }
        widths = {
            'weight': 100,
            'reps': 100,
            'date': 130,
            'sets': 100,
            'e1rm': 100,
        }
        for column in TREE_COLUMNS:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], minwidth=70, anchor='center')

        bottom_frame = ttk.Frame(self.root, padding=10)
        bottom_frame.grid(row=2, column=0, sticky='ew')
        bottom_frame.columnconfigure(0, weight=1)

        explanation = (
            'Остаются только недоминированные подходы: строка скрывается, если есть другой подход '
            'с весом не меньше и повторами не меньше. При равных повторах остаётся больший вес.'
        )
        ttk.Label(bottom_frame, text=explanation, wraplength=720, foreground='#555').grid(
            row=0, column=0, sticky='ew'
        )
        ttk.Label(bottom_frame, textvariable=self.status, foreground='#555').grid(
            row=1, column=0, sticky='ew', pady=(6, 0)
        )

    def _refresh_from_csv(self) -> None:
        self.trainings = normalize_trainings(load_trainings())
        previous = self.exercise_name.get()
        self.exercises = _clean_exercise_names(self.trainings.get('exercise', pd.Series(dtype=object)))
        self.exercise_combo.configure(values=self.exercises, state='readonly' if self.exercises else 'disabled')

        if previous in self.exercises:
            self.exercise_name.set(previous)
        elif self.exercises:
            self.exercise_name.set(self.exercises[0])
        else:
            self.exercise_name.set('')

        self._refresh_table()

    def _refresh_table(self) -> None:
        for item_id in self.tree.get_children():
            self.tree.delete(item_id)

        exercise = self.exercise_name.get().strip()
        if not exercise:
            self.current_table = pd.DataFrame(columns=DISPLAY_COLUMNS)
            self.status.set('В trainings.csv нет упражнений.')
            return

        result = build_best_performances(self.trainings, exercise)
        self.current_table = result.table.copy()

        if result.table.empty:
            self.status.set(f'{exercise}: нет подходов с весом > 0 и повторами > 0.')
            return

        for _, row in result.table.iterrows():
            date = pd.Timestamp(row['date']).strftime('%Y-%m-%d') if pd.notna(row['date']) else ''
            values = (
                _format_number(row['weight']),
                _format_number(row['reps']),
                date,
                _format_number(row['sets']),
                _format_number(row['e1rm']),
            )
            self.tree.insert('', 'end', values=values)

        self.status.set(
            f'{exercise}: показано {len(result.table)} из {result.unique_performances} уникальных результатов. '
            f'Скрыто доминированных: {result.hidden_rows}. Исходных подходов: {result.source_rows}.'
        )

    def _copy_table_to_clipboard(self) -> None:
        if self.current_table.empty:
            self.status.set('Копировать нечего: таблица пустая.')
            return

        output = self.current_table.copy()
        output['date'] = pd.to_datetime(output['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        output = output.rename(
            columns={
                'weight': 'Вес',
                'reps': 'Повторы',
                'date': 'Дата',
                'sets': 'Подходы',
                'e1rm': 'e1RM',
            }
        )
        text = output.to_csv(sep='\t', index=False)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status.set('Таблица скопирована в буфер обмена.')

    def run(self) -> None:
        self.root.mainloop()


def main() -> ExerciseProgressWindow:
    app = ExerciseProgressWindow.from_csv()
    app.run()
    return app


if __name__ == '__main__':
    main()
