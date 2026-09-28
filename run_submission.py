"""Последовательно выполняет кодовые ячейки submission.ipynb из корня проекта.

Обучение и поиск используют локальные Parquet и кэш эмбеддингов. Итоговый CSV
проверяется последней ячейкой ноутбука перед сохранением в answer.csv.
"""

import json
from pathlib import Path

from IPython.display import display


def main():
    notebook = json.loads(Path("submission.ipynb").read_text(encoding="utf-8"))
    scope = {"__name__": "__main__", "display": display}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell_id = cell.get("id", "unnamed")
        print(f"Cell {cell_id}", flush=True)
        code = "".join(cell["source"])
        exec(compile(code, f"submission:{cell_id}", "exec"), scope)


if __name__ == "__main__":
    main()
