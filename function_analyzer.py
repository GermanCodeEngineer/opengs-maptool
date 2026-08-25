import argparse
import ast
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class FunctionAnalyzer(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.file_content = file_content.splitlines()
        self.definitions: List[str] = []
        # Maps formatted type strings -> list of parameter names (or '<return>')
        self.type_to_params: Dict[str, List[str]] = defaultdict(list)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_function(node)
        self.generic_visit(node)

    def _process_function(self, node: ast.AST) -> None:
        # Extract the exact header definition line(s) from original source code
        start_line = node.lineno - 1
        header_lines = []
        for i in range(start_line, len(self.file_content)):
            line = self.file_content[i]
            header_lines.append(line.strip())
            if line.rstrip().endswith(":"):
                break
        
        full_def = " ".join(header_lines)
        self.definitions.append(full_def)

        # 1. Extract parameter type annotations
        args_node = node.args
        all_args = (
            args_node.posonlyargs
            + args_node.args
            + args_node.kwonlyargs
        )
        if args_node.vararg:
            all_args.append(args_node.vararg)
        if args_node.kwarg:
            all_args.append(args_node.kwarg)

        for arg in all_args:
            if arg.annotation:
                param_name = arg.arg
                type_str = ast.unparse(arg.annotation)
                self.type_to_params[type_str].append(param_name)

        # 2. Extract return type annotation if present
        if node.returns:
            return_type_str = ast.unparse(node.returns)
            self.type_to_params[return_type_str].append("<return>")


def analyze_directory(target_dir: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    all_definitions: List[str] = []
    aggregated_types: Dict[str, List[str]] = defaultdict(list)

    for py_file in target_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(py_file))
            analyzer = FunctionAnalyzer(content)
            analyzer.visit(tree)

            all_definitions.extend(analyzer.definitions)
            for type_str, usage_locations in analyzer.type_to_params.items():
                aggregated_types[type_str].extend(usage_locations)
        except (SyntaxError, UnicodeDecodeError):
            continue

    return all_definitions, aggregated_types


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Python function definitions, parameter types, and return types in a directory."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Target directory path to search for Python files",
    )
    args = parser.parse_args()

    target_dir: Path = args.directory.resolve()
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: Directory '{target_dir}' does not exist or is not a directory.")
        return

    definitions, type_data = analyze_directory(target_dir)

    print("==================================================")
    print(f" FUNCTION DEFINITIONS REPORT ({len(definitions)} found)")
    print("==================================================\n")
    
    for def_line in definitions:
        print(def_line)

    print("\n==================================================")
    print(" TYPE ANNOTATION FREQUENCY & USAGE ANALYSIS")
    print("==================================================\n")

    if not type_data:
        print("No type annotations found in function signatures.")
        return

    # Sort types by frequency descending
    sorted_types = sorted(type_data.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"{'Type Annotation':<35} | {'Count':<6} | Top Parameter Names / Usage")
    print("-" * 75)

    for type_str, usage in sorted_types:
        count = len(usage)
        usage_counts = Counter(usage)
        common_usage = ", ".join(
            f"{name} ({c})" for name, c in usage_counts.most_common(3)
        )
        print(f"{type_str:<35} | {count:<6} | {common_usage}")


if __name__ == "__main__":
    main()
