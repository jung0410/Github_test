# generate_tree.py
import os
from pathlib import Path

# 폴더/파일 제외 규칙 (필요하면 더 추가)
EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".idea", ".vscode", "venv", ".venv", "dist", "build"
}
EXCLUDE_FILES = {
    ".DS_Store"
}
EXCLUDE_SUFFIXES = {".pkl", ".npy", ".npz", ".jpg", ".jpeg", ".png", ".mp4", ".avi"}

def print_tree(
    root=".",
    max_depth=4,           # 깊이 제한 (README용이면 2~3 추천)
    max_files_per_dir=50,  # 폴더당 파일 너무 많으면 컷
):
    root = Path(root).resolve()

    def _walk(path: Path, prefix: str, depth: int):
        if depth > max_depth:
            return

        items = []
        for p in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if p.is_dir() and p.name in EXCLUDE_DIRS:
                continue
            if p.is_file():
                if p.name in EXCLUDE_FILES:
                    continue
                if p.suffix.lower() in EXCLUDE_SUFFIXES:
                    continue
            items.append(p)

        # 파일 개수 제한
        if len(items) > max_files_per_dir:
            items = items[:max_files_per_dir] + ["__TRUNCATED__"]

        for i, p in enumerate(items):
            last = (i == len(items) - 1)
            connector = "└── " if last else "├── "

            if p == "__TRUNCATED__":
                print(prefix + connector + "… (truncated)")
                continue

            print(prefix + connector + p.name)

            if p.is_dir():
                extension = "    " if last else "│   "
                _walk(p, prefix + extension, depth + 1)

    print(".")
    _walk(root, "", 1)

if __name__ == "__main__":
    print_tree(root=".", max_depth=3, max_files_per_dir=30)
