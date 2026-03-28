from __future__ import annotations

import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import clear_output, display

from data_io import ensure_directories, load_trainings, normalize_trainings


def main() -> widgets.VBox:
    ensure_directories()
    progress_df = normalize_trainings(load_trainings())

    exercise_options = sorted(
        progress_df.loc[
            progress_df['exercise'].notna()
            & (progress_df['exercise'] != '')
            & (progress_df['exercise'] != 'nan'),
            'exercise',
        ].unique()
    )

    exercise_dropdown = widgets.Dropdown(
        options=exercise_options,
        description='Exercise:',
        layout=widgets.Layout(width='450px'),
    )
    result_output = widgets.Output()

    def show_best_reps_per_weight(exercise_name: str) -> None:
        with result_output:
            clear_output()

            exercise_df = progress_df[progress_df['exercise'] == exercise_name].copy()
            exercise_df = exercise_df.dropna(subset=['weight', 'reps'])

            if exercise_df.empty:
                print('Нет данных для этого упражнения.')
                return

            exercise_df = exercise_df.sort_values(
                by=['weight', 'reps', 'date', 'sets'],
                ascending=[True, False, False, False],
            )

            best_df = (
                exercise_df.groupby('weight', as_index=False)
                .first()[['weight', 'reps', 'date', 'sets', 'e1rm']]
                .rename(
                    columns={
                        'reps': 'best_reps',
                        'date': 'best_date',
                        'sets': 'best_sets',
                        'e1rm': 'best_e1rm',
                    }
                )
                .sort_values('weight')
                .reset_index(drop=True)
            )

            best_df['best_date'] = best_df['best_date'].dt.strftime('%Y-%m-%d')
            best_df['weight'] = best_df['weight'].round(2)
            best_df['best_e1rm'] = best_df['best_e1rm'].round(2)

            display(best_df)
            print(f'Упражнение: {exercise_name}')
            print(f'Всего записей: {len(exercise_df)}')
            print(f"Уникальных весов: {best_df['weight'].nunique()}")

    def on_exercise_change(change: dict[str, object]) -> None:
        if change['name'] == 'value':
            show_best_reps_per_weight(str(change['new']))

    exercise_dropdown.observe(on_exercise_change, names='value')

    app = widgets.VBox([exercise_dropdown, result_output])
    display(app)

    if exercise_options:
        show_best_reps_per_weight(exercise_options[0])
    else:
        with result_output:
            print('В trainings.csv нет упражнений.')
    return app


if __name__ == '__main__':
    main()
