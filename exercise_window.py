from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk
from typing import Final, Iterable

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.figure import Figure

from data_io import ensure_directories, load_exercises, load_trainings, normalize_trainings


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    column: str


METRICS: Final[tuple[MetricSpec, ...]] = (
    MetricSpec(
        key='raw_volume',
        title='Сырой объём',
        column='raw_volume',
    ),
    MetricSpec(
        key='smart_volume',
        title='Умный объём',
        column='smart_volume',
    ),
    MetricSpec(
        key='best_e1rm',
        title='e1RM лучшего сета за день',
        column='best_e1rm',
    ),
)

DISPLAY_DATE_FORMAT: Final[str] = '%d%m%Y'
SMOOTH_RADIUS_DAYS: Final[int] = 3
PLOT_DEBOUNCE_MS: Final[int] = 180


def _clean_exercise_names(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != 'nan':
            result.append(text)
    return result


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


def _exercise_catalog(trainings: pd.DataFrame, exercises: pd.DataFrame) -> list[str]:
    known_exercises = _clean_exercise_names(exercises.get('exercise', pd.Series(dtype=object)))
    trained_exercises = _clean_exercise_names(trainings.get('exercise', pd.Series(dtype=object)))
    return sorted(set(known_exercises) | set(trained_exercises))


def _smooth_by_calendar_window(
    dates: pd.Series,
    values: pd.Series,
    *,
    radius_days: int = SMOOTH_RADIUS_DAYS,
) -> np.ndarray:
    series = pd.Series(
        pd.to_numeric(values, errors='coerce').to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(dates).dt.normalize()),
    ).dropna()

    if series.empty:
        return np.array([], dtype=float)

    series = series.sort_index()
    date_values = series.index.to_numpy(dtype='datetime64[ns]')
    numeric_values = series.to_numpy(dtype=float)
    radius = np.timedelta64(radius_days, 'D')

    smoothed = np.empty(len(series), dtype=float)
    for idx, date_value in enumerate(date_values):
        mask = (date_values >= date_value - radius) & (date_values <= date_value + radius)
        smoothed[idx] = float(np.nanmean(numeric_values[mask]))

    return smoothed


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
    )

    return daily[columns]


class ScrollableCheckFrame(ttk.Frame):
    def __init__(self, master: tk.Misc, *, height: int = 330) -> None:
        super().__init__(master)
        self.canvas = tk.Canvas(self, height=height, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor='nw')
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.scrollbar.grid(row=0, column=1, sticky='ns')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.inner.bind('<Configure>', self._update_scroll_region)
        self.canvas.bind('<Configure>', self._update_inner_width)
        self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)
        self.canvas.bind_all('<Button-4>', self._on_mousewheel_linux)
        self.canvas.bind_all('<Button-5>', self._on_mousewheel_linux)

    def _update_scroll_region(self, _: tk.Event[tk.Misc]) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _update_inner_width(self, event: tk.Event[tk.Misc]) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event: tk.Event[tk.Misc]) -> None:
        if not self.winfo_ismapped():
            return
        delta = int(-1 * (event.delta / 120))
        self.canvas.yview_scroll(delta, 'units')

    def _on_mousewheel_linux(self, event: tk.Event[tk.Misc]) -> None:
        if not self.winfo_ismapped():
            return
        delta = -1 if event.num == 4 else 1
        self.canvas.yview_scroll(delta, 'units')


class ExerciseMetricsWindow:
    def __init__(self, daily_metrics: pd.DataFrame, exercises: list[str]) -> None:
        self.daily_metrics = daily_metrics.copy()
        if not self.daily_metrics.empty:
            self.daily_metrics['date'] = pd.to_datetime(self.daily_metrics['date'], errors='coerce').dt.normalize()
            self.daily_metrics = self.daily_metrics.dropna(subset=['date']).sort_values(['date', 'exercise'])

        metric_exercises = _clean_exercise_names(self.daily_metrics.get('exercise', pd.Series(dtype=object)))
        self.exercises = sorted(set(exercises) | set(metric_exercises))

        if self.daily_metrics.empty:
            today = pd.Timestamp.today().normalize()
            self.min_date = today
            self.max_date = today
        else:
            self.min_date = pd.Timestamp(self.daily_metrics['date'].min()).normalize()
            self.max_date = pd.Timestamp(self.daily_metrics['date'].max()).normalize()

        self.date_span_days = max(0, int((self.max_date - self.min_date).days))

        self.root = tk.Tk()
        self.root.title('MyTrainings — прогресс по упражнениям')
        self.root.geometry('1360x820')
        self.root.minsize(1050, 650)

        self.start_offset = tk.IntVar(value=0)
        self.end_offset = tk.IntVar(value=self.date_span_days)
        self.start_date_label = tk.StringVar()
        self.end_date_label = tk.StringVar()
        self.date_range_label = tk.StringVar()
        self.status = tk.StringVar(value='Выбери упражнения и метрики, затем нажми «Построить».')

        self.show_average = tk.BooleanVar(value=True)
        self.show_smooth = tk.BooleanVar(value=True)
        self.show_points = tk.BooleanVar(value=True)

        self.exercise_vars: dict[str, tk.BooleanVar] = {
            exercise: tk.BooleanVar(value=True) for exercise in self.exercises
        }
        self.metric_vars: dict[str, tk.BooleanVar] = {
            metric.key: tk.BooleanVar(value=True) for metric in METRICS
        }

        self.figure = Figure(figsize=(10, 5.5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas: FigureCanvasTkAgg | None = None
        self.start_scale: ttk.Scale | None = None
        self.end_scale: ttk.Scale | None = None
        self._plot_after_id: str | None = None
        self._suppress_slider = False

        self._build_ui()
        self._update_date_labels()
        self.plot()

    @classmethod
    def from_csv(cls) -> ExerciseMetricsWindow:
        ensure_directories()
        trainings = load_trainings()
        exercises = load_exercises()
        daily_metrics = build_daily_exercise_metrics(trainings, exercises)
        return cls(daily_metrics, _exercise_catalog(trainings, exercises))

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.root, padding=10)
        controls.grid(row=0, column=0, sticky='nsew')
        controls.columnconfigure(0, weight=1)

        plot_frame = ttk.Frame(self.root, padding=(0, 10, 10, 10))
        plot_frame.grid(row=0, column=1, sticky='nsew')
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self._build_date_controls(controls)
        self._build_metric_controls(controls)
        self._build_graph_controls(controls)
        self._build_exercise_controls(controls)
        self._build_status_controls(controls)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

    def _build_date_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text='Диапазон дат', padding=8)
        frame.grid(row=0, column=0, sticky='ew')
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text='С:').grid(row=0, column=0, sticky='w')
        ttk.Label(frame, textvariable=self.start_date_label, width=12).grid(row=0, column=1, sticky='w')

        self.start_scale = ttk.Scale(
            frame,
            from_=0,
            to=self.date_span_days,
            orient='horizontal',
            variable=self.start_offset,
            command=self._on_start_slider,
        )
        self.start_scale.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(2, 6))

        ttk.Label(frame, text='По:').grid(row=2, column=0, sticky='w')
        ttk.Label(frame, textvariable=self.end_date_label, width=12).grid(row=2, column=1, sticky='w')

        self.end_scale = ttk.Scale(
            frame,
            from_=0,
            to=self.date_span_days,
            orient='horizontal',
            variable=self.end_offset,
            command=self._on_end_slider,
        )
        self.end_scale.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(2, 6))

        ttk.Label(frame, textvariable=self.date_range_label, foreground='#555').grid(
            row=4, column=0, columnspan=2, sticky='w'
        )
        ttk.Button(frame, text='Весь диапазон', command=self._select_full_date_range).grid(
            row=5, column=0, columnspan=2, sticky='ew', pady=(8, 0)
        )

    def _build_metric_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text='Метрики', padding=8)
        frame.grid(row=1, column=0, sticky='ew', pady=(10, 0))
        frame.columnconfigure(0, weight=1)

        for row, metric in enumerate(METRICS):
            ttk.Checkbutton(frame, text=metric.title, variable=self.metric_vars[metric.key]).grid(
                row=row, column=0, sticky='w'
            )

        buttons = ttk.Frame(frame)
        buttons.grid(row=len(METRICS), column=0, sticky='ew', pady=(8, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ttk.Button(buttons, text='All', command=lambda: self._set_metric_selection(True)).grid(
            row=0, column=0, sticky='ew'
        )
        ttk.Button(buttons, text='None', command=lambda: self._set_metric_selection(False)).grid(
            row=0, column=1, sticky='ew', padx=(6, 0)
        )

    def _build_graph_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text='График', padding=8)
        frame.grid(row=2, column=0, sticky='ew', pady=(10, 0))
        frame.columnconfigure(0, weight=1)

        ttk.Checkbutton(
            frame,
            text=f'Smooth ±{SMOOTH_RADIUS_DAYS} дня',
            variable=self.show_smooth,
            command=self.plot,
        ).grid(row=0, column=0, sticky='w')
        ttk.Checkbutton(
            frame,
            text='Средняя линия',
            variable=self.show_average,
            command=self.plot,
        ).grid(row=1, column=0, sticky='w')
        ttk.Checkbutton(
            frame,
            text='Исходные точки',
            variable=self.show_points,
            command=self.plot,
        ).grid(row=2, column=0, sticky='w')

    def _build_exercise_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text='Упражнения', padding=8)
        frame.grid(row=3, column=0, sticky='nsew', pady=(10, 0))
        parent.rowconfigure(3, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        buttons = ttk.Frame(frame)
        buttons.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ttk.Button(buttons, text='All', command=lambda: self._set_exercise_selection(True)).grid(
            row=0, column=0, sticky='ew'
        )
        ttk.Button(buttons, text='None', command=lambda: self._set_exercise_selection(False)).grid(
            row=0, column=1, sticky='ew', padx=(6, 0)
        )

        scroll = ScrollableCheckFrame(frame)
        scroll.grid(row=1, column=0, sticky='nsew')

        for row, exercise in enumerate(self.exercises):
            ttk.Checkbutton(scroll.inner, text=exercise, variable=self.exercise_vars[exercise]).grid(
                row=row, column=0, sticky='w', padx=(0, 8)
            )

    def _build_status_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=4, column=0, sticky='ew', pady=(10, 0))
        frame.columnconfigure(0, weight=1)

        ttk.Button(frame, text='Построить', command=self.plot).grid(row=0, column=0, sticky='ew')
        ttk.Label(frame, textvariable=self.status, wraplength=330, foreground='#555').grid(
            row=1, column=0, sticky='ew', pady=(8, 0)
        )

    def _date_from_offset(self, offset: int) -> pd.Timestamp:
        return self.min_date + pd.Timedelta(days=int(offset))

    def _read_slider_offsets(self) -> tuple[int, int]:
        if self.start_scale is None or self.end_scale is None:
            start = int(self.start_offset.get())
            end = int(self.end_offset.get())
        else:
            start = int(round(float(self.start_scale.get())))
            end = int(round(float(self.end_scale.get())))

        start = max(0, min(self.date_span_days, start))
        end = max(0, min(self.date_span_days, end))
        if start > end:
            start, end = end, start
        return start, end

    def _date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        start_offset, end_offset = self._read_slider_offsets()
        return self._date_from_offset(start_offset), self._date_from_offset(end_offset)

    def _set_slider_offsets(self, start: int, end: int, *, schedule_plot: bool = False) -> None:
        start = max(0, min(self.date_span_days, int(start)))
        end = max(0, min(self.date_span_days, int(end)))
        if start > end:
            end = start

        self._suppress_slider = True
        self.start_offset.set(start)
        self.end_offset.set(end)
        if self.start_scale is not None:
            self.start_scale.set(start)
        if self.end_scale is not None:
            self.end_scale.set(end)
        self._suppress_slider = False

        self._update_date_labels()
        if schedule_plot and self.canvas is not None:
            self._schedule_plot()

    def _on_start_slider(self, _: str) -> None:
        if self._suppress_slider:
            return
        start = int(round(float(self.start_scale.get()))) if self.start_scale is not None else 0
        end = int(round(float(self.end_scale.get()))) if self.end_scale is not None else self.date_span_days
        if start > end:
            end = start
        self._set_slider_offsets(start, end, schedule_plot=True)

    def _on_end_slider(self, _: str) -> None:
        if self._suppress_slider:
            return
        start = int(round(float(self.start_scale.get()))) if self.start_scale is not None else 0
        end = int(round(float(self.end_scale.get()))) if self.end_scale is not None else self.date_span_days
        if end < start:
            start = end
        self._set_slider_offsets(start, end, schedule_plot=True)

    def _update_date_labels(self) -> None:
        start, end = self._date_range()
        start_text = start.strftime(DISPLAY_DATE_FORMAT)
        end_text = end.strftime(DISPLAY_DATE_FORMAT)
        self.start_date_label.set(start_text)
        self.end_date_label.set(end_text)
        self.date_range_label.set(f'{start_text} — {end_text}')

    def _schedule_plot(self) -> None:
        if self._plot_after_id is not None:
            self.root.after_cancel(self._plot_after_id)
        self._plot_after_id = self.root.after(PLOT_DEBOUNCE_MS, self.plot)

    def _select_full_date_range(self) -> None:
        self._set_slider_offsets(0, self.date_span_days, schedule_plot=True)

    def _set_exercise_selection(self, selected: bool) -> None:
        for var in self.exercise_vars.values():
            var.set(selected)

    def _set_metric_selection(self, selected: bool) -> None:
        for var in self.metric_vars.values():
            var.set(selected)

    def _selected_exercises(self) -> list[str]:
        return [exercise for exercise, var in self.exercise_vars.items() if var.get()]

    def _selected_metrics(self) -> list[MetricSpec]:
        return [metric for metric in METRICS if self.metric_vars[metric.key].get()]

    def _filtered_data(self, start: pd.Timestamp, end: pd.Timestamp, exercises: list[str]) -> pd.DataFrame:
        if self.daily_metrics.empty:
            return self.daily_metrics.copy()

        mask = (
            (self.daily_metrics['date'] >= start)
            & (self.daily_metrics['date'] <= end)
            & (self.daily_metrics['exercise'].isin(exercises))
        )
        return self.daily_metrics.loc[mask].copy()

    def _draw_empty(self, message: str) -> None:
        self.ax.clear()
        self.ax.text(0.5, 0.5, message, ha='center', va='center', transform=self.ax.transAxes)
        self.ax.set_axis_off()
        if self.canvas is not None:
            self.canvas.draw_idle()

    def _series_for_plot(self, data: pd.DataFrame, metric: MetricSpec) -> pd.DataFrame:
        series = data[['date', metric.column]].copy()
        series[metric.column] = pd.to_numeric(series[metric.column], errors='coerce')
        series = series.dropna(subset=[metric.column]).sort_values('date')
        return series

    def _plot_series(self, series: pd.DataFrame, label_base: str, metric: MetricSpec) -> int:
        values = series[metric.column]
        if values.notna().sum() == 0:
            return 0

        dates = series['date']
        line_label = f'{label_base} — smooth' if self.show_smooth.get() else label_base

        if self.show_points.get():
            self.ax.scatter(
                dates,
                values,
                alpha=0.35,
                s=24,
                label='_nolegend_',
            )

        if self.show_smooth.get():
            smooth_values = _smooth_by_calendar_window(dates, values)
            line_dates = pd.Series(pd.to_datetime(dates).dt.normalize()).sort_values()
            line, = self.ax.plot(
                line_dates,
                smooth_values,
                marker='o',
                linewidth=1.8,
                markersize=3,
                label=line_label,
            )
        else:
            line, = self.ax.plot(
                dates,
                values,
                marker='o',
                linewidth=1.4,
                markersize=3,
                label=line_label,
            )

        if self.show_average.get():
            mean_value = float(values.mean())
            self.ax.axhline(
                mean_value,
                linestyle='--',
                linewidth=1.1,
                alpha=0.75,
                color=line.get_color(),
                label=f'{label_base} — среднее {mean_value:.2f}',
            )

        return 1

    def plot(self) -> None:
        self._plot_after_id = None
        self._update_date_labels()

        start, end = self._date_range()
        exercises = self._selected_exercises()
        metrics = self._selected_metrics()

        if not exercises:
            self.status.set('Не выбрано ни одного упражнения.')
            self._draw_empty('Выбери хотя бы одно упражнение')
            return

        if not metrics:
            self.status.set('Не выбрана ни одна метрика.')
            self._draw_empty('Выбери хотя бы одну метрику')
            return

        selected = self._filtered_data(start, end, exercises)
        if selected.empty:
            self.status.set('В выбранном диапазоне нет тренировок по выбранным упражнениям.')
            self._draw_empty('Нет тренировок в выбранном диапазоне')
            return

        self.ax.clear()
        self.ax.set_axis_on()
        plotted_series = 0

        for exercise in exercises:
            exercise_df = selected[selected['exercise'] == exercise].sort_values('date')
            if exercise_df.empty:
                continue

            for metric in metrics:
                series = self._series_for_plot(exercise_df, metric)
                if series.empty:
                    continue
                plotted_series += self._plot_series(series, f'{exercise} — {metric.title}', metric)

        if plotted_series == 0:
            self.status.set('Для выбранных упражнений и метрик нет значений.')
            self._draw_empty('Нет значений для графика')
            return

        metric_titles = ', '.join(metric.title for metric in metrics)
        start_text = start.strftime(DISPLAY_DATE_FORMAT)
        end_text = end.strftime(DISPLAY_DATE_FORMAT)
        self.ax.set_title(f'{metric_titles}\n{start_text} — {end_text}')
        self.ax.set_xlabel('Дата')
        self.ax.set_ylabel('Значение')
        self.ax.grid(True, alpha=0.3)
        self.ax.xaxis.set_major_locator(AutoDateLocator())
        self.ax.xaxis.set_major_formatter(DateFormatter(DISPLAY_DATE_FORMAT))
        self.figure.autofmt_xdate()

        legend_columns = 1 if plotted_series < 8 else 2
        self.ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8, ncol=legend_columns)
        self.figure.tight_layout()

        parts = [
            f'Серий: {plotted_series}',
            f'Упражнений: {len(exercises)}',
            f'Метрик: {len(metrics)}',
        ]
        if self.show_smooth.get():
            parts.append(f'smooth ±{SMOOTH_RADIUS_DAYS} дн.')
        if self.show_average.get():
            parts.append('среднее включено')
        self.status.set('. '.join(parts) + '.')

        if self.canvas is not None:
            self.canvas.draw_idle()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = ExerciseMetricsWindow.from_csv()
    app.run()


if __name__ == '__main__':
    main()
