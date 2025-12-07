import subprocess
from pathlib import Path

DATA_DIR = Path("data/raw")


def download_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Замените owner/dataset на реальный, если нужно
    dataset = "ulaelg/corn-leaf-disease-data"
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(DATA_DIR), "--unzip"]
    print("Запускаю:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Данные скачаны в", DATA_DIR)


def dvc_pull_or_download():
    """Попытка dvc pull; если не удаётся — вызываем download_data"""
    try:
        subprocess.check_call(["dvc", "pull"])  # если нет remote — приведёт к ошибке
        print("dvc pull выполнен")
    except Exception:
        print("dvc pull не сработал — пробуем скачать напрямую через Kaggle")
        download_data()


if __name__ == "__main__":
    download_data()