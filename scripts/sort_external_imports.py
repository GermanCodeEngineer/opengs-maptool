import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Helper to determine if an import is external (not relative or from opengs_maptool)
def is_external_import(node):
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.startswith("opengs_maptool"):
                return False
        return True
    elif isinstance(node, ast.ImportFrom):
        if node.level > 0:
            return False  # relative import
        if node.module and node.module.startswith("opengs_maptool"):
            return False
        return True
    return False

def process_file(filepath: Path):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        source = ''.join(lines)
    tree = ast.parse(source)

    # Find all import lines and their line numbers
    import_lines: List[Tuple[int, str, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            lineno = node.lineno - 1  # 0-based
            import_lines.append((lineno, lines[lineno], node))

    # Separate external and internal imports
    external = [(i, l, n) for i, l, n in import_lines if is_external_import(n)]
    internal = [(i, l, n) for i, l, n in import_lines if not is_external_import(n)]

    # Sort external imports by root package name
    def root_name(node):
        if isinstance(node, ast.Import):
            return node.names[0].name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            return (node.module or '').split('.')[0]
        return ''
    external_sorted = sorted(external, key=lambda x: root_name(x[2]).lower())

    # Build new lines
    new_lines = lines[:]
    for (old, _, _), (_, new_line, _) in zip(external, external_sorted):
        new_lines[old] = new_line

    # Write back only if changed
    if [l for i, l, n in external] != [l for i, l, n in external_sorted]:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Sorted external imports in {filepath}")


def process_folder(folder: Path):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.py'):
                process_file(Path(root) / file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sort_external_imports.py <folder>")
        sys.exit(1)
    process_folder(Path(sys.argv[1]))
