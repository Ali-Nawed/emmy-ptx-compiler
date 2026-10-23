import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    from pathlib import Path
    from dataclasses import fields, is_dataclass
    from collections import Counter
    from pprint import pformat
    from emmy.compiler.graph import Graph
    from emmy.compiler.ir.kernel.ir import KernelOp
    return Counter, Graph, KernelOp, Path, fields, is_dataclass, json, mo, pformat


@app.cell
def _(mo):
    mo.md("""
    # Explore Emmy's Kernel IR

    Select a problem below. Emmy's **built-in parser** turns its JSON into typed
    Python objects. Cells are editable: try `graph.nodes`, `kernel.body`, or
    `tile.axes` to explore further. This notebook inspects IR; it does not generate PTX.
    """)
    return


@app.cell
def _(Path, mo):
    repo = Path(__file__).resolve().parent
    problem_picker = mo.ui.dropdown(
        options=sorted(p.name for p in (repo / "problems").iterdir() if (p / "meta.json").exists()),
        value="p00_scalar_add", label="Problem",
    )
    problem_picker
    return problem_picker, repo


@app.cell
def _(json, problem_picker, repo, mo):
    problem_dir = repo / "problems" / problem_picker.value
    metadata = json.loads((problem_dir / "meta.json").read_text())
    kernel_picker = mo.ui.dropdown(
        options=[k["entry"] for k in metadata["kernels"]],
        value=metadata["kernels"][0]["entry"], label="Kernel",
    )
    mo.vstack([mo.md(f"**Computation:** `{metadata['code']}`"), kernel_picker])
    return kernel_picker, metadata, problem_dir


@app.cell
def _(Graph, KernelOp, json, kernel_picker, metadata, problem_dir):
    ir_path = problem_dir / f"{kernel_picker.value}.ir.json"
    raw_ir = json.loads(ir_path.read_text())
    graph = Graph.from_dict(raw_ir)
    kernel_node = next(n for n in graph.nodes.values()
                       if isinstance(n.op, KernelOp) and n.op.name == kernel_picker.value)
    kernel = kernel_node.op
    launch = next(k for k in metadata["kernels"] if k["entry"] == kernel.name)
    return graph, ir_path, kernel, kernel_node, launch, raw_ir


@app.cell
def _(graph, launch, mo):
    mo.vstack([
        mo.md("## Graph nodes and launch contract\nParameter order comes from `meta.json`."),
        mo.ui.table([{"node": name, "operation": type(node.op).__name__,
                      "inputs": str(node.inputs), "outputs": str(node.outputs)}
                     for name, node in graph.nodes.items()]),
        mo.json(launch),
    ])
    return


@app.cell
def _(ir_path, mo):
    mo.vstack([mo.md("## Readable IR"),
               mo.ui.code_editor(value=ir_path.with_suffix(".txt").read_text(), language="plaintext", disabled=True)])
    return


@app.cell
def _(kernel, mo, pformat):
    tile = kernel.body[0]
    mo.vstack([mo.md("## Parsed body\nThese are Python objects, not the serialized body string."),
               mo.md("```python\n" + pformat(kernel.body, width=95) + "\n```")])
    return (tile,)


@app.cell
def _(Counter, fields, is_dataclass, kernel, mo):
    def walk_ir(value, path="body"):
        if is_dataclass(value) and not isinstance(value, type):
            yield path, value
            for field in fields(value):
                yield from walk_ir(getattr(value, field.name), f"{path}.{field.name}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                yield from walk_ir(item, f"{path}[{index}]")

    parsed_nodes = list(walk_ir(kernel.body))
    node_counts = Counter(type(value).__name__ for _, value in parsed_nodes)
    mo.vstack([mo.md("## Object inventory\nIncludes statements, expressions, axes and type metadata."),
               mo.ui.table([{"type": name, "count": count}
                            for name, count in sorted(node_counts.items())])])
    return parsed_nodes, walk_ir


@app.cell
def _(mo, parsed_nodes):
    node_picker = mo.ui.dropdown(
        options={f"{path}: {type(value).__name__}": index
                 for index, (path, value) in enumerate(parsed_nodes)},
        value=f"{parsed_nodes[0][0]}: {type(parsed_nodes[0][1]).__name__}", label="Inspect an object",
    )
    node_picker
    return (node_picker,)


@app.cell
def _(fields, mo, node_picker, parsed_nodes):
    selected_path, selected_node = parsed_nodes[node_picker.value]
    mo.vstack([mo.md(f"**Path:** `{selected_path}`\n\n**Class:** `{type(selected_node).__module__}.{type(selected_node).__name__}`"),
               mo.ui.table([{"field": field.name, "value": repr(getattr(selected_node, field.name))}
                            for field in fields(selected_node)])])
    return selected_node, selected_path


@app.cell
def _(mo, tile):
    mo.md(f"""
    ## Try your own expressions

    The first statement is **{type(tile).__name__}**. Add a Python cell and try:

    ```python
    tile.axes           # Logical work dimensions
    tile.body[0]        # First nested statement
    type(tile.body[0])  # Its actual Emmy class
    selected_node       # Object chosen above
    ```

    For p00, look for `a0` with extent 64 and accesses at `a0 * 2 + 0/1`.
    Its launch has 256 threads, so the future emitter must guard inactive workers.
    """)
    return


if __name__ == "__main__":
    app.run()
