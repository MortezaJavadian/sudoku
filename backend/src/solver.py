import copy
from typing import List, Tuple, Optional

class SudokuSolver:
    """
    A Sudoku Solver implementing Backtracking with Minimum Remaining Values (MRV) heuristic
    inspired by LiorSinai/SudokuSolver-Python and aima-python.
    """
    def __init__(self, grid: List[List[int]]):
        self.grid = copy.deepcopy(grid)
        self.n = 9
        
    def solve(self) -> bool:
        """
        Solves the sudoku in-place. Returns True if solved, False if unsolvable.
        """
        # First check if the initial grid is valid
        if not self._is_initial_grid_valid():
            return False
            
        return self._solve_backtrack()
        
    def get_grid(self) -> List[List[int]]:
        return self.grid

    def _solve_backtrack(self) -> bool:
        empty = self._find_empty_mrv()
        if not empty:
            return True # Solved!
            
        row, col = empty
        for num in range(1, 10):
            if self._is_valid(row, col, num):
                self.grid[row][col] = num
                
                if self._solve_backtrack():
                    return True
                    
                self.grid[row][col] = 0 # Backtrack
                
        return False
        
    def _find_empty_mrv(self) -> Optional[Tuple[int, int]]:
        """
        Finds the empty cell with the minimum remaining valid values (MRV heuristic).
        This drastically speeds up solving compared to simple linear search.
        """
        min_options = 10
        best_cell = None
        
        for r in range(self.n):
            for c in range(self.n):
                if self.grid[r][c] == 0:
                    options = sum(1 for num in range(1, 10) if self._is_valid(r, c, num))
                    if options < min_options:
                        min_options = options
                        best_cell = (r, c)
                        if min_options == 1:
                            return best_cell # Can't do better than 1 option
                            
        return best_cell

    def _is_valid(self, row: int, col: int, num: int) -> bool:
        # Check row
        for i in range(self.n):
            if self.grid[row][i] == num:
                return False
                
        # Check column
        for i in range(self.n):
            if self.grid[i][col] == num:
                return False
                
        # Check 3x3 box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        for i in range(3):
            for j in range(3):
                if self.grid[box_row + i][box_col + j] == num:
                    return False
                    
        return True
        
    def _is_initial_grid_valid(self) -> bool:
        """
        Checks if the starting grid violates any basic Sudoku rules.
        """
        for r in range(self.n):
            for c in range(self.n):
                val = self.grid[r][c]
                if val != 0:
                    self.grid[r][c] = 0
                    if not self._is_valid(r, c, val):
                        self.grid[r][c] = val
                        return False
                    self.grid[r][c] = val
        return True
