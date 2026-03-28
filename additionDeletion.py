from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import clear_output, display

from data_io import ensure_directories, load_exercises, load_trainings, normalize_trainings, save_trainings


DEFAULT_BODYWEIGHT = 69.0


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


def parse_bulk_line(line: str, *, bodyweight_mode: bool, bodyweight: float, add_w_default: float) -> list[tuple[float, float, float]] | None:
    normalized = norm_line(line).lower()
    if not normalized:
        return None

    if bodyweight_mode:
        match = re.fullmatch(r'\+(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)', normalized)
        if match:
            add_weight = float(match.group(1))
            reps = float(match.group(2))
            return [(1.0, bodyweight + add_weight, reps)]

        match = re.fullmatch(r'(\d+(?:\.\d+)?)x\(\+(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)\)', normalized)
        if match:
            sets = float(match.group(1))
            add_weight = float(match.group(2))
            reps = float(match.group(3))
            return [(sets, bodyweight + add_weight, reps)]

        match = re.fullmatch(r'(\d+(?:\.\d+)?)', normalized)
        if match:
            reps = float(match.group(1))
            return [(1.0, bodyweight + add_w_default, reps)]

        match = re.fullmatch(r'(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)', normalized)
        if match:
            sets = float(match.group(1))
            reps = float(match.group(2))
            return [(sets, bodyweight + add_w_default, reps)]

        return None

    match = re.fullmatch(r'(\d+(?:\.\d+)?)x\((\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)\)', normalized)
    if match:
        return [(float(match.group(1)), float(match.group(2)), float(match.group(3)))]

    match = re.fullmatch(r'(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)', normalized)
    if match:
        return [(float(match.group(1)), float(match.group(2)), float(match.group(3)))]

    match = re.fullmatch(r'(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)', normalized)
    if match:
        return [(1.0, float(match.group(1)), float(match.group(2)))]

    return None


def main(bodyweight: float = DEFAULT_BODYWEIGHT, trainings_csv: Path | None = None) -> widgets.VBox:
    ensure_directories()

    trainings_df = normalize_trainings(load_trainings() if trainings_csv is None else load_trainings(trainings_csv))
    exercises_df = load_exercises()

    df = trainings_df.copy()
    state: dict[str, object] = {'i': 0, 'deleted_stack': [], 'current_day_map': None}

    out = widgets.Output()
    status = widgets.HTML()
    header = widgets.HTML()

    prev_btn = widgets.Button(description='← Назад')
    next_btn = widgets.Button(description='Дальше →')
    undo_btn = widgets.Button(description='Undo', button_style='warning')
    save_btn = widgets.Button(description='Сохранить CSV', button_style='success')

    delete_input = widgets.Text(
        value='',
        placeholder='Например: 0,2,5 или 3-7',
        description='shown_id:',
        layout=widgets.Layout(width='420px'),
    )
    delete_btn = widgets.Button(description='Удалить выбранные', button_style='danger')

    date_picker = widgets.DatePicker(description='Date')
    exercise_dropdown = widgets.Dropdown(
        options=sorted(exercises_df['exercise'].dropna().astype(str).unique().tolist()),
        description='Exercise',
        layout=widgets.Layout(width='420px'),
    )
    exercise_new = widgets.Text(
        value='',
        placeholder='если новое — впиши',
        description='New',
        layout=widgets.Layout(width='420px'),
    )

    add_bodyweight = widgets.Checkbox(value=False, description=f'Add bodyweight ({bodyweight:g})')
    sets_in = widgets.FloatText(value=1.0, description='Sets', layout=widgets.Layout(width='180px'))
    weight_in = widgets.FloatText(value=0.0, description='Weight', layout=widgets.Layout(width='220px'))
    reps_in = widgets.FloatText(value=0.0, description='Reps', layout=widgets.Layout(width='220px'))

    bulk = widgets.Textarea(
        value='',
        placeholder=(
            'Bulk (каждая строка):\n'
            'Без bodyweight:\n'
            '80x3\n'
            '3x85x2\n'
            '5x(60x5)\n\n'
            'С галочкой Add bodyweight:\n'
            '12\n'
            '3x12\n'
            '+10x6\n'
            '2x(+10x6)\n'
            '(Weight сверху будет add-weight по умолчанию)'
        ),
        description='Bulk',
        layout=widgets.Layout(width='760px', height='170px'),
    )

    btn_add_one = widgets.Button(description='Добавить 1', button_style='success')
    btn_add_bulk = widgets.Button(description='Добавить Bulk', button_style='success')
    btn_show_day = widgets.Button(description='Показать выбранную дату')

    top_bar = widgets.HBox([prev_btn, next_btn, undo_btn, save_btn])
    del_bar = widgets.HBox([delete_input, delete_btn])
    add_bar_1 = widgets.HBox([date_picker, btn_show_day, add_bodyweight])
    add_bar_2 = widgets.HBox([exercise_dropdown, exercise_new])
    add_bar_3 = widgets.HBox([sets_in, weight_in, reps_in, btn_add_one, btn_add_bulk])

    display_cols = [c for c in ['exercise', 'e1rm', 'sets', 'weight', 'reps'] if c in df.columns]

    def get_dates() -> list[pd.Timestamp]:
        dates = df['date'].dropna()
        return list(pd.Series(dates.unique()).sort_values(ascending=False))

    def current_day_df() -> tuple[pd.DataFrame, pd.Timestamp | None]:
        dates = get_dates()
        if not dates:
            return pd.DataFrame(), None
        current_date = pd.Timestamp(dates[int(state['i'])])
        day_df = df[df['date'] == current_date].copy().sort_values(['_row_id'], ascending=False)
        day_df['shown_id'] = np.arange(len(day_df))
        state['current_day_map'] = day_df[['shown_id', '_row_id']].copy()
        return day_df, current_date

    def render(message: str = '') -> None:
        with out:
            clear_output(wait=True)

            dates = get_dates()
            if not dates:
                header.value = '<h3>Датасет пуст</h3>'
                status.value = message
                return

            day_df, current_date = current_day_df()
            header.value = f"<h3>[{int(state['i']) + 1}/{len(dates)}] {current_date.date()}</h3>"
            status.value = f"<span style='color:#888'>{message}</span>" if message else ''

            if day_df.empty:
                display(pd.DataFrame(columns=['shown_id'] + display_cols))
                return

            display(day_df[['shown_id'] + display_cols].reset_index(drop=True))

    def show_date(date_value: pd.Timestamp) -> None:
        with out:
            clear_output(wait=True)
            day_df = df[df['date'] == date_value.normalize()].copy().sort_values(['_row_id'], ascending=False)
            if day_df.empty:
                print('На эту дату записей нет.')
                return
            day_df['shown_id'] = np.arange(len(day_df))
            display(day_df[['shown_id'] + display_cols].reset_index(drop=True))

    def chosen_exercise() -> str:
        name = exercise_new.value.strip()
        return name if name else str(exercise_dropdown.value)

    def recompute_e1rm_all() -> None:
        nonlocal df
        df['e1rm'] = np.where(
            df['weight'].notna() & df['reps'].notna() & (df['reps'] > 0),
            df['weight'] * (1 + df['reps'] / 30.0),
            np.nan,
        )

    def add_rows(rows: list[tuple[float, float, float, str, pd.Timestamp]]) -> None:
        nonlocal df
        next_id = 0 if df.empty else int(df['_row_id'].max() + 1)

        new_rows: list[dict[str, object]] = []
        for sets, weight, reps, exercise_name, date_value in rows:
            new_rows.append(
                {
                    '_row_id': next_id,
                    'exercise': exercise_name,
                    'date': pd.to_datetime(date_value).normalize(),
                    'sets': float(sets),
                    'weight': float(weight),
                    'reps': float(reps),
                    'e1rm': recompute_e1rm_one(float(weight), float(reps)),
                }
            )
            next_id += 1

        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df = df.sort_values(['date', 'exercise', 'weight', 'reps', '_row_id']).copy()
        recompute_e1rm_all()

    def on_prev(_: widgets.Button) -> None:
        dates = get_dates()
        if not dates:
            return
        state['i'] = (int(state['i']) - 1) % len(dates)
        render()

    def on_next(_: widgets.Button) -> None:
        dates = get_dates()
        if not dates:
            return
        state['i'] = (int(state['i']) + 1) % len(dates)
        render()

    def on_undo(_: widgets.Button) -> None:
        nonlocal df
        deleted_stack = state['deleted_stack']
        assert isinstance(deleted_stack, list)

        if not deleted_stack:
            render('Нечего отменять')
            return

        restored = deleted_stack.pop()
        df = pd.concat([df, restored], ignore_index=True)
        df = df.sort_values(['date', 'exercise', 'weight', 'reps', '_row_id']).copy()
        render('Отменено')

    def on_delete(_: widgets.Button) -> None:
        nonlocal df
        try:
            shown_ids = parse_ids(delete_input.value)
        except Exception as exc:
            render(f'Ошибка: {exc}')
            return

        if not shown_ids:
            render('Не указаны shown_id')
            return

        current_map = state['current_day_map']
        if current_map is None or current_map.empty:
            render('Нечего удалять')
            return

        valid_rows = current_map[current_map['shown_id'].isin(shown_ids)]
        if valid_rows.empty:
            render('Нет таких shown_id')
            return

        row_ids = valid_rows['_row_id'].tolist()
        deleted = df[df['_row_id'].isin(row_ids)].copy()
        if deleted.empty:
            render('Уже удалено')
            return

        deleted_stack = state['deleted_stack']
        assert isinstance(deleted_stack, list)
        deleted_stack.append(deleted)

        df = df[~df['_row_id'].isin(row_ids)].copy()
        dates_now = get_dates()
        state['i'] = (int(state['i']) % len(dates_now)) if dates_now else 0
        delete_input.value = ''
        render(f'Удалено строк: {len(deleted)}')

    def on_add_one(_: widgets.Button) -> None:
        selected_date = date_picker.value
        if selected_date is None:
            status.value = '<b style="color:#d33">Выбери дату</b>'
            return

        exercise_name = chosen_exercise()
        sets = float(sets_in.value)
        reps = float(reps_in.value)
        input_weight = float(weight_in.value)
        weight = bodyweight + input_weight if add_bodyweight.value else input_weight

        add_rows([(sets, weight, reps, exercise_name, pd.Timestamp(selected_date))])
        status.value = f'<b style="color:#2a2">Добавлено:</b> {exercise_name}'
        show_date(pd.Timestamp(selected_date))

    def on_add_bulk(_: widgets.Button) -> None:
        selected_date = date_picker.value
        if selected_date is None:
            status.value = '<b style="color:#d33">Выбери дату</b>'
            return

        exercise_name = chosen_exercise()
        bodyweight_mode = bool(add_bodyweight.value)
        add_weight_default = float(weight_in.value) if bodyweight_mode else 0.0

        lines = [line for line in bulk.value.splitlines() if line.strip()]
        added = 0
        bad_lines: list[str] = []
        rows_to_add: list[tuple[float, float, float, str, pd.Timestamp]] = []

        for line in lines:
            parsed = parse_bulk_line(
                line,
                bodyweight_mode=bodyweight_mode,
                bodyweight=bodyweight,
                add_w_default=add_weight_default,
            )
            if parsed is None:
                bad_lines.append(line)
                continue
            for sets, weight, reps in parsed:
                rows_to_add.append((sets, weight, reps, exercise_name, pd.Timestamp(selected_date)))
                added += 1

        if rows_to_add:
            add_rows(rows_to_add)

        message = f'<b style="color:#2a2">Добавлено строк:</b> {added}'
        if bad_lines:
            message += f' | <b style="color:#d33">Не распарсилось:</b> {bad_lines}'
        status.value = message
        show_date(pd.Timestamp(selected_date))

    def on_show_day(_: widgets.Button) -> None:
        selected_date = date_picker.value
        if selected_date is None:
            status.value = '<b style="color:#d33">Выбери дату</b>'
            return
        show_date(pd.Timestamp(selected_date))

    def on_save(_: widgets.Button) -> None:
        save_path = save_trainings(df)
        render(f'Сохранено: {save_path}')

    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)
    undo_btn.on_click(on_undo)
    save_btn.on_click(on_save)
    delete_btn.on_click(on_delete)
    btn_add_one.on_click(on_add_one)
    btn_add_bulk.on_click(on_add_bulk)
    btn_show_day.on_click(on_show_day)

    app = widgets.VBox(
        [
            top_bar,
            header,
            out,
            del_bar,
            status,
            widgets.HTML('<hr><b>Добавление</b>'),
            add_bar_1,
            add_bar_2,
            add_bar_3,
            bulk,
        ]
    )

    display(app)
    render()
    return app


if __name__ == '__main__':
    main()
