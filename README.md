# MyTrainings
Этот проект собран из ноутбука `MyTrainings.ipynb`.

Основные изменения относительно Colab-версии:

- убрана зависимость от `google.colab.drive`;
- чтение идёт из локальных CSV-файлов в папке `data/`;
- сохранение идёт в локальные файлы проекта;
- логика разнесена по нескольким Python-файлам.

## Структура проекта

```text
training_project/
├── README.md
├── requirements.txt
├── data_io.py
├── visualization.py
├── additionDeletion.py
├── newExercise.py
├── exerciseProgress.py
├── data/
│   ├── exercises.csv
│   ├── trainings.csv
│   └── muscle_groups.csv
└── outputs/
```

### `visualization.py`

Позволяет смотреть графики объёма и интенсивности по большим и маленьким группам мышц.

### `additionDeletion.py`

Бывшая следующая большая ячейка с:

- просмотром тренировок по датам;
- Редактирование `data/trainings.csv`: удаление\добавление строк.

### `newExercise.py`

- редактирование/просмотр `data/exercises.csv`;
- позволяет открыть существующее упражнение;
- редактирует `difficulty_coeff` и коэффициенты по мышцам.

### `exerciseProgress.py`

Это отдельная промежуточная ячейка из ноутбука, которую я тоже вынес в модуль.

Она:

- показывает лучший результат по каждому весу для выбранного упражнения;
- ничего не изменяет в CSV.

## Запуск

Лучше запускать в Jupyter / VS Code Notebook / JupyterLab, потому что интерфейс основан на `ipywidgets`.

Примеры:

```python
import visualization
visualization.main()
```

```python
import additionDeletion
additionDeletion.main()
```

```python
import newExercise
newExercise.main()
```

```python
import exerciseProgress
exerciseProgress.main()
```

## Формула объёма по мышцам

Для каждого сета сначала считается `e1RM` по формуле Эпли:

```math
e1RM = weight \cdot \left(1 + \frac{reps}{30}\right)
```

Дальше считается score с учётом упражнения:

```math
score_{set} = e1RM \cdot difficulty\_coeff \cdot sets
```

Где:

- `difficulty_coeff` — коэффициент упражнения из `exercises.csv`
- `sets` — количество подходов

Потом score распределяется по мелким мышечным группам:

```math
score_{small\_muscle} = score_{set} \cdot muscle\_coef
```

Затем мелкие группы агрегируются в крупные через `muscle_groups.csv`.

## Что сохраняется после визуализации

Файл `outputs/daily_muscle_volume_big_and_small_4metrics.csv` содержит:

- `date`
- колонки `e1rm__big__...`
- колонки `e1rm__small__...`
- колонки `tonnage__big__...`
- колонки `tonnage__small__...`
- колонки `rawton__big__...`
- колонки `rawton__small__...`
- колонки `fail__big__...`
- колонки `fail__small__...`

Для всех календарных дней между минимальной и максимальной датой строится полный диапазон: если в день нет тренировки, в таблице стоят нули.

## Зависимости

См. `requirements.txt`.
