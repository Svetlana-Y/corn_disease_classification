# corn_disease_classification


Репозиторий содержит минимальный MLOps-совместимый пайплайн для задачи классификации болезней кукурузы.

## Что внутри
- `corn_classifier/` — Python-пакет с кодом для загрузки данных, препроцессинга, обучения, экспорта и инференса.
- `configs/` — конфиги hydra.
- `dvc` проинициализирован, но удалённое хранилище не настроено — используется `download_data()` из Kaggle.
- `plots/` — папка для графиков (в `.gitignore` — но примеры графиков сохраняются локально в процессе работы).

## Setup

1. Клонирование репозитория:
```bash
git clone <URL репозитория> corn-classifier
cd corn-classifier
```

2. Создать виртуальное окружение и активировать его:
```bash
uv sync
source .venv/bin/activate
```

3. Установить `uv` (если не установлен) и зависимости из `pyproject.toml`:
```bash
pip install --upgrade pip
pip install uv
uv install
```
> Если `uv install` упадёт — можно использовать `pip install -e .[dev]`.

4. Установить pre-commit хуки и проверить их:
```bash
pre-commit install
pre-commit run -a
```

5. Получить данные (у вас нет доступа к удалённому dvc-remote — скачиваем с Kaggle):
   - Настройте Kaggle credentials: положите `kaggle.json` в `~/.kaggle/kaggle.json` (инструкция на kaggle.com).
   - Запустите:
```bash
python -c "from corn_classifier.data.download import download_data; download_data()"
```

## mlflow

Запуск MLflow-сервера (в отдельном терминале)

```bash
mlflow server --backend-store-uri file:./mlruns --default-artifact-root file:./mlartifacts --host 127.0.0.1 --port 8080
```

## Train

Запуск обучения (пример, быстро, 1 эпоха):

```bash
train
```

Экспорт модели в ONNX:
```bash
python -m corn_classifier.export
```

Запуск инференса на картинке:
```bash
python -m corn_classifier.infer infer.input=path/to/image.jpg
```

## Детали
См. разделы `configs/` для всех гиперпараметров (hydra). MLflow ожидается по адресу `http://127.0.0.1:8080` — если не поднят, логирование будет локальным, но скрипт не упадёт.
