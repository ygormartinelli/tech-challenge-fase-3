"""Valida a estrutura do artefato executado sem exigir CSVs na CI."""

from pathlib import Path

import nbformat


def test_committed_notebook_contains_executed_results() -> None:
    """Protege contra publicação acidental de notebook sem resultados."""
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(
        root / "notebooks/01_eda_medical_abstracts.ipynb", as_version=4
    )
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code)
    outputs = [output for cell in code for output in cell.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    assert sum("image/png" in output.get("data", {}) for output in outputs) == 7
