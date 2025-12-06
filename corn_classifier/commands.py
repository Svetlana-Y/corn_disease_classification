from typing import Any


def noop(*args: Any, **kwargs: Any) -> None:
    print("use train / infer / export entrypoints")


if __name__ == "__main__":
    noop()
