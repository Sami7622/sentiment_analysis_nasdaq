#!/usr/bin/env python3
"""
Script to execute notebook cells sequentially
"""
import json
import sys
import subprocess
from pathlib import Path

def execute_notebook_cells(notebook_path, start_cell=24):
    """Execute notebook cells starting from a specific cell index"""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Extract code from cells
    cells_to_run = []
    for i, cell in enumerate(nb['cells']):
        if i >= start_cell and cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if source.strip() and not source.strip().startswith('!'):
                cells_to_run.append((i, source))
            elif source.strip().startswith('!'):
                # Handle shell commands
                cmd = source.strip()[1:].strip()
                print(f"\n[Cell {i}] Executing: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                print(result.stdout)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
    
    # Execute Python code cells
    code = '\n\n'.join([f"# Cell {i}\n{source}" for i, source in cells_to_run])
    
    if code.strip():
        print(f"\nExecuting {len(cells_to_run)} cells...")
        exec(code, {'__name__': '__main__'})

if __name__ == '__main__':
    notebook_path = Path(__file__).parent / 'eda.ipynb'
    execute_notebook_cells(notebook_path, start_cell=24)

