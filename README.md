# AIGate Eval

Самостоятельная проверка AIGate на своих данных: скрипт прогоняет размеченные наборы через ваш AIGate и считает, сколько персональных данных он скрыл и сколько промпт-инъекций остановил. Нужен Python 3.9+, сторонние пакеты не требуются.

Полная инструкция — в [README.html](README.html): настройки, политика, метрики, конвертер открытых наборов, формат своих файлов. Откройте файл в браузере после клонирования.

## Быстрый старт

```bash
git clone https://github.com/aitalon/aigate-eval.git
cd aigate-eval
cp config.example.ini config.ini   # укажите url AIGate и ключ приложения
python3 aigate_eval.py check       # проверка подключения и политики
python3 aigate_eval.py             # все проверки; отдельно: pii или injection
```

Свои наборы `*.jsonl` кладутся в `benchmarks/pii/` и `benchmarks/injection/`, для пробы там уже есть `example.jsonl`. Пробный прогон: `--limit 100`. Результаты — в `results/`.

## Открытые наборы

```bash
python3 -m venv .venv
.venv/bin/pip install -r converters/requirements.txt
.venv/bin/python converters/convert_open_datasets.py --list
.venv/bin/python converters/convert_open_datasets.py all
```
