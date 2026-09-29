"""Execute a notebook atomically with nbclient from the project environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--kernel", default="alpha-search-venv")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    path = args.notebook.resolve()
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name=args.kernel,
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    temporary = path.with_name(f"{path.stem}.executed.tmp.ipynb")
    nbformat.write(notebook, temporary)
    temporary.replace(path)
    print(path)


if __name__ == "__main__":
    main()
