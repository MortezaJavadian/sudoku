import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from solver import SudokuSolver

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
    # A known 'hard' Sudoku puzzle
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    print("Initial Puzzle:")
    print_grid(puzzle)
    
    solver = SudokuSolver(puzzle)
    
    start_time = time.time()
    success = solver.solve()
    end_time = time.time()
    
    print(f"\nSolved: {success}")
    print(f"Time taken: {end_time - start_time:.5f} seconds")
    
    if success:
        print("\nSolution:")
        print_grid(solver.get_grid())
    else:
        print("\nPuzzle could not be solved.")
        
    # Test Invalid Puzzle
    invalid_puzzle = [row[:] for row in puzzle]
    invalid_puzzle[0][2] = 5 # Place '5' in the same row as the first '5'
    
    solver_invalid = SudokuSolver(invalid_puzzle)
    success_invalid = solver_invalid.solve()
    assert success_invalid == False, "Solver should fail on invalid puzzles"
    print("\nInvalid puzzle correctly identified as unsolvable.")

if __name__ == "__main__":
    main()
