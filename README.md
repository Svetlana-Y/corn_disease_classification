# corn_disease_classification

Репозиторий содержит пайплайн для задачи классификации болезней кукурузы.

## Что внутри

- `corn_classifier/` — Python-пакет с кодом для:
  - загрузки данных
  - препроцессинга
  - обучения
  - экспорта модели
  - инференса
- `configs/` — конфигурации Hydra
- `dvc` — используется для версионирования данных

## Setup

1. Клонирование репозитория:

```bash
git clone https://github.com/Svetlana-Y/corn_disease_classification.git
cd corn_disease_classification
```

2. Создать виртуальное окружение и активировать его:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Установить uv (если не установлен):

```bash
pip install --upgrade pip
pip install uv
```

4. Установить зависимости проекта:

```bash
uv pip install -e .
```

5. Установить pre-commit хуки и проверить их:

```bash
pre-commit install
pre-commit run -a
```

6. Получить данные (либо автоматически скачивается при запуске train):
   - Настройте dvc credentials: положите `dvc_credentials.json` (JSON с приватным ключом Service Account) в корень проекта.
   - Запустите:

```bash
dvc pull
```

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

MLflow ожидается по адресу `http://127.0.0.1:8080` — если не поднят, логирование будет локальным, но скрипт не упадёт.

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

См. разделы `configs/` для всех гиперпараметров (hydra)
