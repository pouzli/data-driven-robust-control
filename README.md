# Data-driven robust control of nonlinear systems

Минимальный воспроизводимый pipeline для дипломной работы:

> **Тема:** «Устойчивость и робастность управления нелинейной динамической системы, идентифицированной на основе данных».

## Реализовано

- Генерация синтетических траекторий для 4 нелинейных систем.
- Data-driven идентификация в виде `f_hat(x) = Theta(x) C` (least squares).
- Анализ residuals и оценка `epsilon` (95% квантиль нормы residual).
- Линеаризация в `x=0`, вычисление Jacobian.
- Решение уравнения Ляпунова `A^T P + P A = -Q`.
- Моделирование с ограниченной неопределенностью `||Delta(t)|| <= epsilon`.
- Добавление обратной связи `u = -Kx` (LQR на линейной части).
- Сравнение uncontrolled vs controlled траекторий.

## Структура проекта

```text
.
├── README.md
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── systems.py
│   ├── basis.py
│   ├── identification.py
│   ├── uncertainty.py
│   ├── lyapunov.py
│   ├── control.py
│   ├── simulation.py
│   ├── plots.py
│   └── utils.py
├── notebooks
│   └── synthetic_experiments.ipynb
├── data
│   └── synthetic
├── results
│   ├── figures
│   ├── tables
│   ├── metrics
│   └── summary
└── report
    └── synthetic_results.md
```

## Быстрый запуск

1. Создать окружение Python 3.11.
2. Установить зависимости:

```bash
pip install -r requirements.txt
```

3. Выполнить ноутбук:

```bash
jupyter notebook notebooks/synthetic_experiments.ipynb
```

или в batch-режиме:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/synthetic_experiments.ipynb
```

После запуска артефакты появятся в:
- `data/synthetic/`
- `results/figures/`
- `results/tables/`
- `results/metrics/`
- `results/summary/synthetic_experiments_summary.json`

## Исследуемые системы

1. `quadratic_oscillator`
2. `van_der_pol`
3. `cross_nonlinear`
4. `saturation_system`

## Генерируемые графики (для каждой системы)

- phase portrait (true)
- phase portrait (identified)
- residual histogram
- Lyapunov contours
- uncontrolled trajectories
- controlled trajectories
