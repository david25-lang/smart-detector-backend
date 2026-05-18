from pathlib import Path


def ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def relative_to_base(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()
