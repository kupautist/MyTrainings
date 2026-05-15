from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import clear_output, display

from data_io import (
    OUTPUTS_DIR,
    ensure_directories,
    load_exercises,
    load_muscle_groups,
    load_trainings,
    normalize_trainings,
    save_dataframe,
)

OUTPUT_CSV = OUTPUTS_DIR / 'daily_muscle_volume_big_and_small_4metrics.csv'


def build_combined_volume_dataframe(
    trainings: pd.DataFrame,
    exercises: pd.DataFrame,
    muscle_groups: pd.DataFrame,
) -> pd.DataFrame:
    train = normalize_trainings(trainings)
    ex = exercises.copy()
    mg = muscle_groups.copy()

    train = train.dropna(subset=['date', 'exercise']).copy()

    missing = sorted(set(train['exercise']) - set(ex['exercise']))
    if missing:
        print('WARNING: эти упражнения отсутствуют в exercises.csv и не попадут в графики:')
        print(missing)

    muscle_cols = [c for c in ex.columns if c not in {'exercise', 'difficulty_coeff', 'sum'}]

    tmp = train.merge(ex[['exercise', 'difficulty_coeff'] + muscle_cols], on='exercise', how='left')
    tmp['difficulty_coeff'] = pd.to_numeric(tmp['difficulty_coeff'], errors='coerce').fillna(0.0)

    tmp['w'] = tmp['weight'].fillna(0.0)
    tmp['r'] = tmp['reps'].fillna(0.0)
    tmp['s'] = tmp['sets'].fillna(1.0)
    tmp['diff'] = tmp['difficulty_coeff'].fillna(0.0)
    tmp['e'] = tmp['e1rm'].fillna(0.0)

    coef_small = tmp[muscle_cols].fillna(0.0)

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


def build_widget_app(combined: pd.DataFrame) -> widgets.VBox:
    df = combined.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    if df.empty:
        raise ValueError('После обработки не осталось данных для визуализации.')

    date_values = pd.Series(df['date'].dt.normalize().unique()).sort_values().tolist()
    date_options = [(d.strftime('%Y-%m-%d'), d) for d in date_values]

    date_range = widgets.SelectionRangeSlider(
        options=date_options,
        index=(0, len(date_options) - 1),
        description='Dates',
        layout=widgets.Layout(width='900px'),
    )

    smooth = widgets.IntSlider(
        value=1,
        min=1,
        max=15,
        step=2,
        description='Smooth',
        continuous_update=False,
    )

    metric = widgets.ToggleButtons(
        options=[('E1RM', 'e1rm'), ('Tonnage', 'tonnage'), ('Raw tonnage', 'rawton'), ('Fail', 'fail')],
        value='e1rm',
        description='Metric',
    )

    mode = widgets.ToggleButtons(
        options=[('Big', 'big'), ('Small', 'small')],
        value='big',
        description='Mode',
    )

    btn_all = widgets.Button(description='Select all')
    btn_none = widgets.Button(description='None')

    out = widgets.Output()
    checks_box = widgets.Box()
    checks: dict[str, widgets.Checkbox] = {}

    metric_name = {'e1rm': 'E1RM', 'tonnage': 'Tonnage', 'rawton': 'Raw tonnage', 'fail': 'Fail'}
    mode_name = {'big': 'Big', 'small': 'Small'}

    def current_prefix() -> str:
        return f'{metric.value}__{mode.value}__'

    def plot(_: object | None = None) -> None:
        with out:
            clear_output(wait=True)

            start_d, end_d = date_range.value
            prefix = current_prefix()
            cols = [c for c in df.columns if c.startswith(prefix)]

            if not cols:
                print(f'Нет колонок для префикса: {prefix}')
                return

            selected = df[(df['date'] >= start_d) & (df['date'] <= end_d)][['date'] + cols].copy()
            chosen = [c for c in cols if checks.get(c) is not None and checks[c].value]

            if not chosen:
                print('Выключены все группы — включи хотя бы одну галочку.')
                return

            window = int(smooth.value)
            if window > 1:
                for col in chosen:
                    selected[col] = selected[col].rolling(window=window, center=True, min_periods=1).mean()

            plt.figure(figsize=(12, 5))
            for col in chosen:
                plt.plot(selected['date'], selected[col], label=col.replace(prefix, ''))

            plt.title(f'{metric_name[metric.value]} | {mode_name[mode.value]} (smooth={window})')
            plt.xlabel('date')
            plt.ylabel('score')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def build_checkboxes() -> None:
        checks.clear()
        prefix = current_prefix()
        cols = [c for c in df.columns if c.startswith(prefix)]

        if not cols:
            checks_box.children = (widgets.HTML(f'<b>Нет колонок для:</b> {prefix}'),)
            return

        for col in cols:
            label = col.replace(prefix, '')
            checkbox = widgets.Checkbox(value=True, description=label)
            checkbox.observe(plot, names='value')
            checks[col] = checkbox

        left = widgets.VBox([checks[col] for col in cols[::2]])
        right = widgets.VBox([checks[col] for col in cols[1::2]])
        checks_box.children = (widgets.HBox([left, right]),)

    def on_metric_or_mode(_: object) -> None:
        build_checkboxes()
        plot()

    def on_all(_: widgets.Button) -> None:
        for checkbox in checks.values():
            checkbox.unobserve(plot, names='value')
            checkbox.value = True
            checkbox.observe(plot, names='value')
        plot()

    def on_none(_: widgets.Button) -> None:
        for checkbox in checks.values():
            checkbox.unobserve(plot, names='value')
            checkbox.value = False
            checkbox.observe(plot, names='value')
        plot()

    btn_all.on_click(on_all)
    btn_none.on_click(on_none)

    build_checkboxes()
    date_range.observe(plot, names='value')
    smooth.observe(plot, names='value')
    metric.observe(on_metric_or_mode, names='value')
    mode.observe(on_metric_or_mode, names='value')

    app = widgets.VBox(
        [
            date_range,
            widgets.HBox([metric, mode, smooth, btn_all, btn_none]),
            checks_box,
            out,
        ]
    )
    plot()
    return app


def main(output_csv: Path = OUTPUT_CSV) -> tuple[pd.DataFrame, widgets.VBox]:
    ensure_directories()
    trainings = load_trainings()
    exercises = load_exercises()
    muscle_groups = load_muscle_groups()

    combined = build_combined_volume_dataframe(trainings, exercises, muscle_groups)
    saved_path = save_dataframe(combined, output_csv)
    print(f'Saved: {saved_path}')

    app = build_widget_app(combined)
    display(app)
    return combined, app


if __name__ == '__main__':
    main()
