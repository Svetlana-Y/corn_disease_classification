import shutil
import subprocess
from pathlib import Path

DATA_DIR = Path("data/raw")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def download_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [
        "ulaelg/corn-leaf-disease-data",
        "smaranjitghose/corn-or-maize-leaf-disease-dataset",
    ]

    for ds in datasets:
        cmd = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            ds,
            "-p",
            str(DATA_DIR),
            "--unzip",
        ]
        print("Запускаю:", " ".join(cmd))
        subprocess.check_call(cmd)

    merged_dir = DATA_DIR / "merged"
    merged_dir.mkdir(exist_ok=True)

    for root in DATA_DIR.iterdir():
        if not root.is_dir():
            continue
        if root.name == "merged":
            continue

        for class_dir in root.iterdir():
            if not class_dir.is_dir():
                continue

            target_class_dir = merged_dir / class_dir.name
            target_class_dir.mkdir(exist_ok=True)

            for img in class_dir.iterdir():
                if img.is_file():
                    shutil.copy(img, target_class_dir / img.name)

    for item in DATA_DIR.iterdir():
        if item.name != "merged":
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    print("Загрузка завершена")


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
