Беляшов Артём

Московский институт электронной техники

2025

Библиотека data-generate-library позволяет создавать модели спинного мозга с учётом анатомической и функциональной организации а также проводить виртуальные эксперименты по записи локальных полевых потенциалов на эпидуральных электродах. Она включает следующие возможности:

Создание популяций нейронов (альфа-мотонейроны, C-волокна, интернейроны и др.). Генерация фоновой активности и болевых сигналов. Размещение виртуальных электродов для записи сигналов. Визуализация результатов симуляции. Библиотека использует фреймворк NetPyNE для моделирования нейронных сетей.

Установка

Перед использованием библиотеки установите необходимые зависимости. Вы можете использовать файл requirements.txt

pip install -r requirements.txt

После установки вы можете импортировать функции из библиотеки:

from data-generate-library import ( create_model, set_background_activity, set_pain, set_electrode, run_simulations, calculate_electrode_signal, plot_electrode_pain_signals )

Основные шаги для работы с библиотекой

Создайте модель спинного мозга : netParams = create_model(scale=1)

Задайте фоновую активность : netParams = set_background_activity(netParams, activity_scale=1)

Добавьте болевой сигнал : netParams, pain_info = set_pain(netParams, vertebrae=["L1"], pain_intensity=2, side="left")

Разместите электроды : electrode_positions = set_electrode(x=15000, y=50000, z=3500, num_electrodes=3, spacing=5000, axis="y")

Запустите симуляцию : run_simulations(netParams, num_runs=1, sim_duration=1000, record_step=0.025)

Вычислите сигналы на электродах : signals = calculate_electrode_signal(electrode_positions, power=2, save_to_csv=True, run_idx=1, pain_info=pain_info)

Визуализируйте результаты : plot_electrode_pain_signals(electrode_positions, signals, pain_info=pain_info)

Структура проекта

neural_sim/ │ ├── init.py # Экспорт функций пакета ├── simulation.py # Основные функции (create_model, set_background_activity, set_pain и др.) ├── requirements.txt # Список зависимостей └── README.md # Описание проекта и инструкции по использованию
