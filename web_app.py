from __future__ import annotations

from dataclasses import dataclass
import base64
from io import BytesIO
import os
from typing import Sequence

import matplotlib

matplotlib.use('Agg')

import matplotlib.dates as mdates
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for

from data_io import (
    ensure_directories,
    load_exercises,
    load_muscle_groups,
    load_trainings,
    save_exercises,
    save_trainings,
)
from web_logic import (
    EXERCISE_METRICS,
    MUSCLE_METRICS,
    MUSCLE_MODES,
    TRAINING_DISPLAY_COLUMNS,
    TrainingRowInput,
    append_training_rows,
    build_best_performances,
    build_combined_volume_dataframe,
    build_daily_exercise_metrics,
    calendar_smooth,
    clean_names,
    date_values_from_frame,
    day_rows,
    delete_training_row_ids,
    display_table_rows,
    exercise_muscle_columns,
    exercise_options_from_data,
    format_date,
    format_ddmmyyyy,
    format_number,
    html_date,
    normalize_trainings_for_editing,
    parse_bulk_line,
    parse_html_date,
    save_exercise_row,
)


app = Flask(__name__)
app.secret_key = os.environ.get('MYTRAININGS_SECRET_KEY', 'mytrainings-local-dev')

COLORS: tuple[str, ...] = (
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#17becf',
    '#bcbd22',
    '#e377c2',
    '#7f7f7f',
    '#003f5c',
    '#ffa600',
)
LINESTYLES: tuple[str, ...] = ('-', '--', '-.', ':')


@dataclass(frozen=True)
class ChartResult:
    image: str | None
    message: str
    lines: int = 0


def _float_form(name: str, default: float = 0.0) -> float:
    raw = request.form.get(name, '').replace(',', '.').strip()
    if not raw:
        return default
    return float(raw)


def _int_arg(name: str, default: int) -> int:
    raw = request.args.get(name)
    if raw is None or raw == '':
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _selected_from_request(name: str, default: Sequence[str], *, submitted_name: str = 'submitted') -> list[str]:
    if request.args.get(submitted_name) != '1':
        return list(default)
    return request.args.getlist(name)


def _checkbox_bool(name: str, default: bool = False) -> bool:
    if request.args.get('submitted') != '1':
        return default
    return request.args.get(name) == '1'


def _training_form_state_from_request(date_value: pd.Timestamp) -> dict[str, str]:
    return {
        'date': html_date(date_value),
        'exercise': (request.form.get('exercise') or '').strip(),
        'new_exercise': (request.form.get('new_exercise') or '').strip(),
        'bulk': request.form.get('bulk') or '',
    }


def _training_form_state_for_render(
    *,
    selected_date: pd.Timestamp,
    exercises: Sequence[str],
    keep_previous: bool,
) -> dict[str, str]:
    previous = session.get('training_form_state', {}) if keep_previous else {}
    if not isinstance(previous, dict):
        previous = {}

    fallback_exercise = exercises[0] if exercises else ''
    return {
        'date': str(previous.get('date') or html_date(selected_date)),
        'exercise': str(previous.get('exercise') or fallback_exercise),
        'new_exercise': str(previous.get('new_exercise') or ''),
        'bulk': str(previous.get('bulk') or ''),
    }


def _date_slider_selection(dates: Sequence[pd.Timestamp]) -> tuple[int, int, pd.Timestamp | None, pd.Timestamp | None]:
    if not dates:
        return 0, 0, None, None

    max_idx = len(dates) - 1
    start_idx = min(max(_int_arg('start_i', 0), 0), max_idx)
    end_idx = min(max(_int_arg('end_i', max_idx), 0), max_idx)
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    return start_idx, end_idx, pd.Timestamp(dates[start_idx]).normalize(), pd.Timestamp(dates[end_idx]).normalize()


def _fig_to_base64(fig: Figure) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('ascii')


def _style_axis(fig: Figure) -> None:
    fig.patch.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
        ax.grid(True, color='#000000', alpha=0.12, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color('#111111')
        ax.tick_params(colors='#111111')
        ax.xaxis.label.set_color('#111111')
        ax.yaxis.label.set_color('#111111')
        ax.title.set_color('#111111')


def _line_style(index: int) -> tuple[str, str]:
    return COLORS[index % len(COLORS)], LINESTYLES[(index // len(COLORS)) % len(LINESTYLES)]


def _calendar_series(
    dates: pd.Series,
    values: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    calendar = pd.date_range(start=pd.Timestamp(start).normalize(), end=pd.Timestamp(end).normalize(), freq='D')
    series = pd.Series(fill_value, index=calendar, dtype='float64')

    parsed_dates = pd.to_datetime(dates, errors='coerce').dt.normalize()
    numeric_values = pd.to_numeric(values, errors='coerce')
    frame = pd.DataFrame({'date': parsed_dates, 'value': numeric_values}).dropna(subset=['date', 'value'])
    if not frame.empty:
        # На всякий случай склеиваем дубли по дню. В daily-таблицах дублей быть не должно.
        by_date = frame.groupby('date')['value'].max()
        common_dates = series.index.intersection(by_date.index)
        series.loc[common_dates] = by_date.loc[common_dates].astype(float)

    return pd.DataFrame({'date': calendar, 'value': series.to_numpy(dtype=float)})


def _plot_empty(message: str) -> str:
    fig = Figure(figsize=(10.5, 5.2), dpi=110)
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes, fontsize=13)
    ax.set_axis_off()
    return _fig_to_base64(fig)


def plot_exercise_metrics_chart(
    daily_metrics: pd.DataFrame,
    *,
    exercises: Sequence[str],
    metric_keys: Sequence[str],
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    show_points: bool,
    show_smooth: bool,
    show_mean: bool,
    smooth_days: int,
) -> ChartResult:
    if daily_metrics.empty or start is None or end is None:
        return ChartResult(image=_plot_empty('Нет данных для графика'), message='Нет данных.')

    metric_by_key = {metric.key: metric for metric in EXERCISE_METRICS}
    metrics = [metric_by_key[key] for key in metric_keys if key in metric_by_key]
    if not exercises:
        return ChartResult(image=_plot_empty('Выбери хотя бы одно упражнение'), message='Упражнения не выбраны.')
    if not metrics:
        return ChartResult(image=_plot_empty('Выбери хотя бы одну метрику'), message='Метрики не выбраны.')

    selected = daily_metrics[
        (daily_metrics['date'] >= start)
        & (daily_metrics['date'] <= end)
        & (daily_metrics['exercise'].isin(exercises))
    ].copy()
    if selected.empty:
        return ChartResult(image=_plot_empty('В выбранном диапазоне нет данных'), message='В выбранном диапазоне нет данных.')

    fig = Figure(figsize=(11.5, 5.7), dpi=110)
    ax = fig.add_subplot(111)
    line_count = 0
    smooth_days = max(int(smooth_days), 0)

    for exercise in exercises:
        exercise_df = selected[selected['exercise'] == exercise].sort_values('date')
        if exercise_df.empty:
            continue

        for metric in metrics:
            values = pd.to_numeric(exercise_df[metric.column], errors='coerce')
            if metric.is_e1rm:
                values = values.replace(0.0, np.nan)
            mask = values.notna()
            if not mask.any():
                continue

            dates = pd.to_datetime(exercise_df.loc[mask, 'date'], errors='coerce').dt.normalize()
            y = values.loc[mask].astype(float)
            color, linestyle = _line_style(line_count)
            label_base = f'{exercise} — {metric.title}'

            if show_points:
                ax.scatter(dates, y, color=color, s=24, alpha=0.82, label=f'{label_base} точки')

            if show_smooth and smooth_days > 0:
                calendar = _calendar_series(dates, y, start=start, end=end, fill_value=0.0)
                smooth = calendar_smooth(calendar['date'], calendar['value'], radius_days=smooth_days)
                ax.plot(
                    calendar['date'],
                    smooth,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.0,
                    label=f'{label_base} smooth ±{smooth_days} дн.',
                )
                if len(y) > 1:
                    ax.plot(dates, y, color=color, linestyle=':', linewidth=0.9, alpha=0.35)
                mean_source = calendar['value']
            else:
                ax.plot(dates, y, color=color, linestyle=linestyle, linewidth=1.7, label=label_base)
                mean_source = y

            if show_mean:
                mean_value = float(pd.to_numeric(mean_source, errors='coerce').mean())
                ax.axhline(
                    mean_value,
                    color=color,
                    linestyle='--',
                    linewidth=0.95,
                    alpha=0.50,
                    label=f'{label_base} среднее {format_number(mean_value)}',
                )

            line_count += 1

    if line_count == 0:
        return ChartResult(image=_plot_empty('Нет значений для выбранных настроек'), message='Нет значений для графика.')

    metric_titles = ', '.join(metric.title for metric in metrics)
    ax.set_title(f'{metric_titles}: {format_date(start, "%d.%m.%Y")} — {format_date(end, "%d.%m.%Y")}')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Значение')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    fig.autofmt_xdate()
    legend_columns = 1 if line_count < 8 else 2
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=7.5, ncol=legend_columns, frameon=True)
    _style_axis(fig)
    return ChartResult(
        image=_fig_to_base64(fig),
        message=f'Линий: {line_count}. Упражнений: {len(exercises)}. Метрик: {len(metrics)}. Smooth: ±{smooth_days} дн.',
        lines=line_count,
    )

def plot_muscle_chart(
    combined: pd.DataFrame,
    *,
    metric: str,
    mode: str,
    groups: Sequence[str],
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    smooth_window: int,
    show_mean: bool,
    show_points: bool,
) -> ChartResult:
    if combined.empty or start is None or end is None:
        return ChartResult(image=_plot_empty('Нет данных для графика'), message='Нет данных.')

    prefix = f'{metric}__{mode}__'
    columns = [column for column in combined.columns if column.startswith(prefix)]
    chosen_columns = [f'{prefix}{group}' for group in groups if f'{prefix}{group}' in columns]

    if not chosen_columns:
        return ChartResult(image=_plot_empty('Выбери хотя бы одну группу'), message='Группы не выбраны.')

    selected = combined[(combined['date'] >= start) & (combined['date'] <= end)][['date'] + chosen_columns].copy()
    if selected.empty:
        return ChartResult(image=_plot_empty('В выбранном диапазоне нет данных'), message='В выбранном диапазоне нет данных.')

    fig = Figure(figsize=(11.5, 5.7), dpi=110)
    ax = fig.add_subplot(111)
    line_count = 0
    smooth_window = max(int(smooth_window), 0)

    for column in chosen_columns:
        group_name = column.replace(prefix, '')
        dates = pd.to_datetime(selected['date'], errors='coerce').dt.normalize()
        values = pd.to_numeric(selected[column], errors='coerce').fillna(0.0).astype(float)
        nonzero_mask = values.ne(0.0)
        if not nonzero_mask.any():
            continue
        color, linestyle = _line_style(line_count)
        drew_group = False

        if show_points and nonzero_mask.any():
            ax.scatter(
                dates.loc[nonzero_mask],
                values.loc[nonzero_mask],
                color=color,
                s=22,
                alpha=0.78,
                label=f'{group_name} точки',
            )
            drew_group = True

        if smooth_window > 0:
            smooth = calendar_smooth(dates, values, radius_days=smooth_window)
            ax.plot(
                dates,
                smooth,
                color=color,
                linestyle=linestyle,
                linewidth=1.9,
                label=f'{group_name} smooth ±{smooth_window} дн.',
            )
            drew_group = True
        elif not show_points and nonzero_mask.any():
            ax.plot(
                dates.loc[nonzero_mask],
                values.loc[nonzero_mask],
                color=color,
                linestyle=linestyle,
                linewidth=1.7,
                label=group_name,
            )
            drew_group = True

        if show_mean and drew_group:
            mean_value = float(values.mean())
            ax.axhline(
                mean_value,
                color=color,
                linestyle=':',
                linewidth=0.95,
                alpha=0.55,
                label=f'{group_name} среднее {format_number(mean_value)}',
            )

        if drew_group:
            line_count += 1

    if line_count == 0:
        return ChartResult(image=_plot_empty('Нет ненулевых точек для выбранных групп'), message='Нет значений для графика.')

    smooth_label = f'±{smooth_window} дн' if smooth_window > 0 else 'выкл'
    ax.set_title(
        f'{MUSCLE_METRICS.get(metric, metric)} | {MUSCLE_MODES.get(mode, mode)} | '
        f'{format_date(start, "%d.%m.%Y")} — {format_date(end, "%d.%m.%Y")} | smooth={smooth_label}'
    )
    ax.set_xlabel('Дата')
    ax.set_ylabel('Score')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    fig.autofmt_xdate()
    legend_columns = 1 if line_count < 10 else 2
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=7.5, ncol=legend_columns, frameon=True)
    _style_axis(fig)
    return ChartResult(
        image=_fig_to_base64(fig),
        message=f'Линий: {line_count}. Групп: {len(groups)}. Smooth: {smooth_label}.',
        lines=line_count,
    )

@app.context_processor
def inject_globals() -> dict[str, object]:
    return {
        'nav_items': [
            ('overview_page', 'Обзор'),
            ('trainings_page', 'Тренировки'),
            ('exercise_editor_page', 'Упражнения'),
            ('best_sets_page', 'Лучшие подходы'),
            ('exercise_charts_page', 'Графики упражнений'),
            ('muscle_charts_page', 'Графики мышц'),
        ],
        'format_number': format_number,
        'format_date': format_date,
        'format_ddmmyyyy': format_ddmmyyyy,
    }


@app.route('/')
def overview_page() -> str:
    ensure_directories()
    trainings = normalize_trainings_for_editing(load_trainings())
    exercises = load_exercises()
    muscle_groups = load_muscle_groups()
    dates = date_values_from_frame(trainings)
    latest_date = dates[-1] if dates else None
    latest_rows = day_rows(trainings, latest_date) if latest_date is not None else pd.DataFrame()
    latest_columns = ['shown_id', *TRAINING_DISPLAY_COLUMNS]

    stats = {
        'Тренировочных строк': len(trainings),
        'Упражнений в справочнике': len(exercises),
        'Маленьких мышц': len(muscle_groups),
        'Дней с тренировками': len(dates),
        'Первый день': format_date(dates[0]) if dates else '—',
        'Последний день': format_date(dates[-1]) if dates else '—',
    }

    section_descriptions = [
        {
            'endpoint': 'trainings_page',
            'title': 'Тренировки',
            'text': 'Просмотр тренировочного дня, удаление строк и добавление новых подходов через bulk-формат.',
        },
        {
            'endpoint': 'exercise_editor_page',
            'title': 'Упражнения',
            'text': 'Справочник упражнений: difficulty_coeff и распределение нагрузки по маленьким мышцам.',
        },
        {
            'endpoint': 'best_sets_page',
            'title': 'Лучшие подходы',
            'text': 'Строка скрывается, если есть другой подход с весом не меньше и повторами не меньше. При равных повторах остаётся больший вес.',
        },
        {
            'endpoint': 'exercise_charts_page',
            'title': 'Графики упражнений',
            'text': 'Прогресс выбранных упражнений по сырому объёму, умному объёму и лучшему дневному e1RM.',
        },
        {
            'endpoint': 'muscle_charts_page',
            'title': 'Графики мышц',
            'text': 'Агрегация нагрузки по большим группам и маленьким мышцам с выбором метрик и сглаживания.',
        },
    ]

    return render_template(
        'overview.html',
        active='overview_page',
        title='Обзор',
        stats=stats,
        section_descriptions=section_descriptions,
        latest_date=latest_date,
        latest_columns=latest_columns,
        latest_rows=latest_rows.head(20).to_dict('records'),
    )


@app.route('/trainings', methods=['GET', 'POST'])
def trainings_page() -> str:
    ensure_directories()
    message = request.args.get('message', '')
    trainings = normalize_trainings_for_editing(load_trainings())
    exercises_df = load_exercises()
    exercises = exercise_options_from_data(trainings, exercises_df)

    if request.method == 'POST':
        action = request.form.get('action', '')
        date_value = parse_html_date(request.form.get('date'))
        if action == 'add_bulk':
            session['training_form_state'] = _training_form_state_from_request(date_value)

        try:
            if action == 'add_bulk':
                exercise_name = (request.form.get('new_exercise') or request.form.get('exercise') or '').strip()
                rows: list[TrainingRowInput] = []
                bad_lines: list[str] = []
                for line in (request.form.get('bulk') or '').splitlines():
                    if not line.strip():
                        continue
                    parsed = parse_bulk_line(line)
                    if parsed is None:
                        bad_lines.append(line)
                        continue
                    for sets, weight, reps in parsed:
                        rows.append(
                            TrainingRowInput(
                                sets=sets,
                                weight=weight,
                                reps=reps,
                                exercise=exercise_name,
                                date=date_value,
                            )
                        )
                updated = append_training_rows(trainings, rows)
                save_trainings(updated)
                message = f'Добавлено строк: {len(rows)}'
                if bad_lines:
                    message += f'. Не распарсилось: {bad_lines}'

            elif action == 'delete':
                row_ids = [int(value) for value in request.form.getlist('row_id')]
                updated, deleted_count = delete_training_row_ids(trainings, row_ids)
                save_trainings(updated)
                message = f'Удалено строк: {deleted_count}'

        except Exception as exc:  # noqa: BLE001 - выводим ошибку в интерфейс, не роняя сайт
            message = f'Ошибка: {exc}'

        return redirect(url_for('trainings_page', date=html_date(date_value), message=message))

    dates_desc = list(reversed(date_values_from_frame(trainings)))
    selected_date = parse_html_date(request.args.get('date'), dates_desc[0] if dates_desc else pd.Timestamp.today())
    current_rows = day_rows(trainings, selected_date)

    current_index = None
    for index, date_value in enumerate(dates_desc):
        if pd.Timestamp(date_value).normalize() == selected_date:
            current_index = index
            break

    prev_date = dates_desc[(current_index - 1) % len(dates_desc)] if current_index is not None and dates_desc else None
    next_date = dates_desc[(current_index + 1) % len(dates_desc)] if current_index is not None and dates_desc else None

    columns = ['shown_id', '_row_id', *TRAINING_DISPLAY_COLUMNS]
    form_state = _training_form_state_for_render(
        selected_date=selected_date,
        exercises=exercises,
        keep_previous=bool(message) and not message.startswith('Удалено'),
    )
    return render_template(
        'trainings.html',
        active='trainings_page',
        title='Тренировки',
        message=message,
        selected_date=selected_date,
        selected_date_html=html_date(selected_date),
        dates_count=len(dates_desc),
        current_index=current_index,
        prev_date=prev_date,
        next_date=next_date,
        columns=columns,
        rows=current_rows.to_dict('records'),
        exercises=exercises,
        form_state=form_state,
    )


@app.route('/exercises', methods=['GET', 'POST'])
def exercise_editor_page() -> str:
    ensure_directories()
    message = request.args.get('message', '')
    exercises_df = load_exercises()
    muscle_cols = exercise_muscle_columns(exercises_df)
    exercise_names = clean_names(exercises_df['exercise'].tolist()) if 'exercise' in exercises_df.columns else []

    selected_name = request.args.get('exercise', exercise_names[0] if exercise_names else '')
    selected_row: dict[str, object] | None = None
    if selected_name:
        mask = exercises_df['exercise'].astype(str).str.strip().str.lower() == selected_name.strip().lower()
        if mask.any():
            selected_row = exercises_df.loc[mask].iloc[0].to_dict()

    if request.method == 'POST':
        try:
            name = (request.form.get('exercise_name') or '').strip()
            difficulty = float((request.form.get('difficulty_coeff') or '1').replace(',', '.'))
            muscles = {
                column: float((request.form.get(f'muscle__{column}') or '0').replace(',', '.'))
                for column in muscle_cols
            }
            updated, action = save_exercise_row(
                exercises_df,
                name=name,
                difficulty_coeff=difficulty,
                muscles=muscles,
            )
            save_exercises(updated)
            total = sum(muscles.values())
            message = f'Упражнение {action}: {name}. Сумма мышц: {total:.6g}'
            return redirect(url_for('exercise_editor_page', exercise=name, message=message))
        except Exception as exc:  # noqa: BLE001
            message = f'Ошибка: {exc}'

    return render_template(
        'exercise_editor.html',
        active='exercise_editor_page',
        title='Упражнения',
        message=message,
        exercise_names=exercise_names,
        selected_name=selected_name,
        selected_row=selected_row,
        muscle_cols=muscle_cols,
    )


@app.route('/best-sets')
def best_sets_page() -> str:
    ensure_directories()
    trainings = normalize_trainings_for_editing(load_trainings())
    exercises_df = load_exercises()
    exercises = exercise_options_from_data(trainings, exercises_df)
    selected = request.args.get('exercise', exercises[0] if exercises else '')
    result = build_best_performances(trainings, selected) if selected else None
    columns = ['weight', 'reps', 'date', 'sets', 'e1rm']

    return render_template(
        'best_sets.html',
        active='best_sets_page',
        title='Лучшие подходы',
        exercises=exercises,
        selected=selected,
        result=result,
        columns=columns,
        rows=result.table.to_dict('records') if result is not None else [],
    )


@app.route('/exercise-charts')
def exercise_charts_page() -> str:
    ensure_directories()
    trainings = normalize_trainings_for_editing(load_trainings())
    exercises_df = load_exercises()
    exercises = exercise_options_from_data(trainings, exercises_df)
    daily = build_daily_exercise_metrics(trainings, exercises_df)
    dates = date_values_from_frame(daily)
    start_i, end_i, start, end = _date_slider_selection(dates)

    default_exercises = exercises[:3]
    selected_exercises = _selected_from_request('exercise', default_exercises)
    selected_metrics = _selected_from_request('metric', ['raw_volume'])
    show_points = _checkbox_bool('show_points', True)
    show_smooth = _checkbox_bool('show_smooth', True)
    show_mean = _checkbox_bool('show_mean', True)
    smooth_days = max(_int_arg('smooth', 3), 0)

    chart = plot_exercise_metrics_chart(
        daily,
        exercises=selected_exercises,
        metric_keys=selected_metrics,
        start=start,
        end=end,
        show_points=show_points,
        show_smooth=show_smooth,
        show_mean=show_mean,
        smooth_days=smooth_days,
    )

    return render_template(
        'exercise_charts.html',
        active='exercise_charts_page',
        title='Графики упражнений',
        exercises=exercises,
        selected_exercises=selected_exercises,
        metrics=EXERCISE_METRICS,
        selected_metrics=selected_metrics,
        dates_json=[format_date(date, '%d.%m.%Y') for date in dates],
        dates_count=len(dates),
        start_i=start_i,
        end_i=end_i,
        show_points=show_points,
        show_smooth=show_smooth,
        show_mean=show_mean,
        smooth_days=smooth_days,
        chart=chart,
    )


@app.route('/muscle-charts')
def muscle_charts_page() -> str:
    ensure_directories()
    trainings = normalize_trainings_for_editing(load_trainings())
    exercises_df = load_exercises()
    muscle_groups = load_muscle_groups()
    combined = build_combined_volume_dataframe(trainings, exercises_df, muscle_groups)
    dates = date_values_from_frame(combined)
    start_i, end_i, start, end = _date_slider_selection(dates)

    metric = request.args.get('metric', 'e1rm')
    if metric not in MUSCLE_METRICS:
        metric = 'e1rm'
    mode = request.args.get('mode', 'big')
    if mode not in MUSCLE_MODES:
        mode = 'big'

    prefix = f'{metric}__{mode}__'
    groups = [column.replace(prefix, '') for column in combined.columns if column.startswith(prefix)]
    default_groups = groups if mode == 'big' else groups[:8]
    selected_groups = _selected_from_request('group', default_groups)
    smooth_window = max(_int_arg('smooth', 3), 0)
    show_mean = _checkbox_bool('show_mean', False)
    show_points = _checkbox_bool('show_points', True)

    chart = plot_muscle_chart(
        combined,
        metric=metric,
        mode=mode,
        groups=selected_groups,
        start=start,
        end=end,
        smooth_window=smooth_window,
        show_mean=show_mean,
        show_points=show_points,
    )

    return render_template(
        'muscle_charts.html',
        active='muscle_charts_page',
        title='Графики мышц',
        muscle_metrics=MUSCLE_METRICS,
        muscle_modes=MUSCLE_MODES,
        metric=metric,
        mode=mode,
        groups=groups,
        selected_groups=selected_groups,
        dates_json=[format_date(date, '%d.%m.%Y') for date in dates],
        dates_count=len(dates),
        start_i=start_i,
        end_i=end_i,
        smooth_window=smooth_window,
        show_mean=show_mean,
        show_points=show_points,
        chart=chart,
    )


def main() -> None:
    ensure_directories()
    app.run(host='127.0.0.1', port=8000, debug=True)


if __name__ == '__main__':
    main()
