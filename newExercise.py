from __future__ import annotations

import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import clear_output, display

from data_io import ensure_directories, load_exercises, save_exercises


REQUIRED_COLUMNS = {'exercise', 'difficulty_coeff', 'sum'}


def main() -> widgets.VBox:
    ensure_directories()
    exercises_df = load_exercises().copy()

    missing_required = REQUIRED_COLUMNS - set(exercises_df.columns)
    if missing_required:
        raise ValueError(f'В exercises.csv не хватает колонок: {missing_required}')

    muscle_cols = [c for c in exercises_df.columns if c not in REQUIRED_COLUMNS]

    exercise_name = widgets.Text(
        description='Exercise:',
        placeholder='Например: Пресс -20',
        layout=widgets.Layout(width='400px'),
    )

    difficulty_coeff_input = widgets.FloatText(
        value=1.0,
        description='difficulty:',
        layout=widgets.Layout(width='250px'),
    )

    muscle_inputs = {
        col: widgets.FloatText(
            value=0.0,
            description=col,
            layout=widgets.Layout(width='220px'),
        )
        for col in muscle_cols
    }

    save_button = widgets.Button(description='Сохранить упражнение', button_style='success', icon='save')
    load_button = widgets.Button(description='Загрузить существующее', button_style='info', icon='download')
    clear_button = widgets.Button(description='Очистить форму', button_style='warning', icon='eraser')
    sum_button = widgets.Button(description='Показать сумму мышц', icon='calculator')
    output_box = widgets.Output()

    def current_muscle_sum() -> float:
        return float(sum(float(widget.value or 0.0) for widget in muscle_inputs.values()))

    def fill_form_from_row(row: pd.Series) -> None:
        exercise_name.value = str(row['exercise'])
        difficulty_coeff_input.value = float(pd.to_numeric(row['difficulty_coeff'], errors='coerce') or 0.0)
        for col in muscle_cols:
            muscle_inputs[col].value = float(pd.to_numeric(row.get(col, 0.0), errors='coerce') or 0.0)

    def clear_form(_: widgets.Button | None = None) -> None:
        exercise_name.value = ''
        difficulty_coeff_input.value = 1.0
        for col in muscle_cols:
            muscle_inputs[col].value = 0.0

    def on_sum_clicked(_: widgets.Button) -> None:
        with output_box:
            clear_output()
            total = current_muscle_sum()
            print(f'Сумма мышечных коэффициентов: {total:.6f}')
            if not np.isclose(total, 1.0, atol=1e-6):
                print('Нормально, если это сделано специально. Но обычно удобно держать сумму около 1.0.')

    def on_load_clicked(_: widgets.Button) -> None:
        with output_box:
            clear_output()
            name = exercise_name.value.strip()
            if not name:
                print('Сначала введи название упражнения.')
                return

            mask = exercises_df['exercise'].astype(str).str.strip().str.lower() == name.lower()
            if not mask.any():
                print('Такого упражнения пока нет в exercises.csv.')
                return

            row = exercises_df.loc[mask].iloc[0]
            fill_form_from_row(row)
            print(f"Загружено существующее упражнение: {row['exercise']}")

    def on_save_clicked(_: widgets.Button) -> None:
        nonlocal exercises_df

        with output_box:
            clear_output()

            name = exercise_name.value.strip()
            if not name:
                print('Название упражнения пустое.')
                return

            diff = pd.to_numeric(difficulty_coeff_input.value, errors='coerce')
            if pd.isna(diff):
                print('difficulty_coeff должен быть числом.')
                return

            row_data: dict[str, float | str] = {'exercise': name}
            for col in muscle_cols:
                value = pd.to_numeric(muscle_inputs[col].value, errors='coerce')
                row_data[col] = 0.0 if pd.isna(value) else float(value)

            row_data['sum'] = float(sum(float(row_data[col]) for col in muscle_cols))
            row_data['difficulty_coeff'] = float(diff)

            mask = exercises_df['exercise'].astype(str).str.strip().str.lower() == name.lower()
            if mask.any():
                idx = exercises_df.index[mask][0]
                for key, value in row_data.items():
                    exercises_df.loc[idx, key] = value
                action = 'обновлено'
            else:
                exercises_df.loc[len(exercises_df)] = row_data
                action = 'добавлено'

            save_path = save_exercises(exercises_df)

            print(f'Упражнение {action}: {name}')
            print(f"difficulty_coeff = {float(row_data['difficulty_coeff']):.6f}")
            print(f"sum = {float(row_data['sum']):.6f}")
            print(f'Сохранено в: {save_path}')

            non_zero = {
                key: value
                for key, value in row_data.items()
                if key in muscle_cols and abs(float(value)) > 1e-12
            }
            if non_zero:
                print('\nНенулевые мышцы:')
                for key, value in sorted(non_zero.items(), key=lambda item: -float(item[1])):
                    print(f'  {key}: {float(value):.6f}')
            else:
                print('\nВсе мышечные коэффициенты сейчас нулевые.')

    clear_button.on_click(clear_form)
    save_button.on_click(on_save_clicked)
    load_button.on_click(on_load_clicked)
    sum_button.on_click(on_sum_clicked)

    muscle_grid = widgets.GridBox(
        children=[muscle_inputs[col] for col in muscle_cols],
        layout=widgets.Layout(grid_template_columns='repeat(3, 240px)', grid_gap='8px 8px'),
    )

    controls_row_1 = widgets.HBox([exercise_name, difficulty_coeff_input])
    controls_row_2 = widgets.HBox([save_button, load_button, clear_button, sum_button])

    app = widgets.VBox(
        [
            controls_row_1,
            controls_row_2,
            widgets.HTML('<b>Распределение по мышцам</b>'),
            muscle_grid,
            output_box,
        ]
    )
    display(app)
    return app


if __name__ == '__main__':
    main()
