import os
import sys
import time
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from solver import SudokuSolver

def read_grid_from_txt(filepath):
    grid = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                row = [int(x) for x in line.split()]
                if len(row) == 9:
                    grid.append(row)
    if len(grid) != 9:
        raise ValueError(f"Invalid grid size in {filepath}")
    return grid

def write_grid_to_txt(grid, filepath):
    with open(filepath, 'w') as f:
        for r in range(9):
            row_str = " ".join(str(val) for val in grid[r])
            f.write(row_str + "\n")

def print_grid(grid):
    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("-" * 21)
        row_str = ""
        for c in range(9):
            if c % 3 == 0 and c != 0:
                row_str += "| "
            val = str(grid[r][c]) if grid[r][c] != 0 else "."
            row_str += val + " "
        print(row_str)

def main():
    input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/sudoku_texts'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/output/phase3'))
    
    os.makedirs(output_dir, exist_ok=True)
    
    txt_files = glob.glob(os.path.join(input_dir, '*.txt'))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
        
    for txt_file in sorted(txt_files):
        filename = os.path.basename(txt_file)
        print(f"\n--- Processing {filename} ---")
        
        try:
            puzzle = read_grid_from_txt(txt_file)
        except Exception as e:
            print(f"Error reading file: {e}")
            continue
            
        print("Initial Puzzle:")
        print_grid(puzzle)
        
        solver = SudokuSolver(puzzle)
        
        start_time = time.time()
        success = solver.solve()
        end_time = time.time()
        
        print(f"\nSolved: {success}")
        print(f"Time taken: {end_time - start_time:.5f} seconds")
        
        if success:
            solved_grid = solver.get_grid()
            print("\nSolution:")
            print_grid(solved_grid)
            
            output_filepath = os.path.join(output_dir, f"solved_{filename}")
            write_grid_to_txt(solved_grid, output_filepath)
            print(f"Saved solution to {output_filepath}")
        else:
            print("\nPuzzle could not be solved (invalid or no solution).")
            output_filepath = os.path.join(output_dir, f"failed_{filename}")
            with open(output_filepath, 'w') as f:
                f.write("UNSOLVABLE OR INVALID PUZZLE\n")
            print(f"Saved failure state to {output_filepath}")

if __name__ == "__main__":
    main()
