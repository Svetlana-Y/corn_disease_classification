import subprocess
from pathlib import Path

DATA_DIR = Path("data/raw")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def download_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = "ulaelg/corn-leaf-disease-data"
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(DATA_DIR), "--unzip"]
    print("Запускаю:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Данные скачаны в", DATA_DIR)


def dvc_pull_or_download():
    try:
        subprocess.check_call(
            ["dvc", "pull"],
            cwd=PROJECT_ROOT,
        )
        print("dvc pull выполнен")
    except subprocess.CalledProcessError as e:
        print("dvc pull не сработал, код:", e.returncode)
        download_data()


if __name__ == "__main__":
    download_data()
