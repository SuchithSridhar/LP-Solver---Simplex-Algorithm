"""
Program to solve linear programming problems using SymPy.
This program uses the Simplex Algorithm to find
the optimal solution to the LP while maintaining fractions.
In its current state, there are inputs for which
the algorithm will misbehave as no anti-cycling rules
have been put in place.

@author Suchith Sridhar Khajjayam
@date 27 Sep 2024
"""

import sympy as sp
from sympy import Matrix, Rational, zeros, eye, simplify

SLACK_VAR = "y"
REAL_VAR = "x"
S_VAR = "s"
EMPTY_VAR = " "

# Control the amount of Tableaus printed
PRINT = True


class Tableau:
    def __init__(self, A, b, c):
        # Convert inputs to SymPy matrices and rational numbers
        self.num_constraints, self.num_variables = A.shape
        A = Matrix(A).applyfunc(Rational)
        b = Matrix(b).applyfunc(Rational)
        c = Matrix(c).applyfunc(Rational)

        # Create the tableau
        self.tableau, self.var_types = self._create_tableau(A, b, c)
        self.var_indices = [i for i in range(len(self.var_types))]

    def _create_tableau(self, A, b, c):
        identity = eye(self.num_constraints)
        b_col = b.reshape(self.num_constraints, 1)
        tableau = b_col.row_join(identity).row_join(A)

        zeros_row = zeros(1, self.num_constraints + 1)
        c_row = c.reshape(1, self.num_variables)
        c_row = zeros_row.row_join(c_row)
        tableau = tableau.col_join(c_row)

        var_types = (
            [EMPTY_VAR]
            + [SLACK_VAR] * self.num_constraints
            + [REAL_VAR] * self.num_variables
        )

        return tableau, var_types

    @property
    def A(self):
        """
        Get the non-identity part of the tableau.
        """
        return self.tableau[:-1, self.num_constraints + 1 :]

    @property
    def b(self):
        """
        Get the first column except last row,
        basically b without -d at the end.
        """
        return self.tableau[:-1, 0]

    @property
    def c(self):
        """
        Get the last row for constants, but exclude -d from the start.
        """
        return self.tableau[-1, 1:]

    @property
    def d(self):
        """
        Get the negative of the value stored in the bottom left corner.
        This is the objective function value.
        """
        return -self.tableau[-1, 0]

    def scale_row(self, row_index, scale_factor):
        self.tableau.row_op(row_index, lambda x, _: x * scale_factor)
        return self.tableau

    def add_rows(self, target_row, source_row, scale_factor=1):
        self.tableau.row_op(
            target_row, lambda x, j: x + scale_factor * self.tableau[source_row, j]
        )
        return self.tableau

    def swap_row(self, target_row, source_row):
        self.tableau = self.tableau.elementary_row_op(
            "n<->m", n=target_row, m=source_row
        )
        return self.tableau

    def scale_col(self, col_index, scale_factor):
        self.tableau[:, col_index] = self.tableau[:, col_index] * scale_factor
        return self.tableau

    def add_cols(self, target_col, source_col, scale_factor=1):
        self.tableau[:, target_col] += scale_factor * self.tableau[:, source_col]
        return self.tableau

    def swap_cols(self, target_col, source_col):
        # Also keeps track of where the variables are going so that
        # later we can identify which columns are real and slack variables
        self.tableau.col_swap(target_col, source_col)
        self.var_types[target_col], self.var_types[source_col] = (
            self.var_types[source_col],
            self.var_types[target_col],
        )
        self.var_indices[target_col], self.var_indices[source_col] = (
            self.var_indices[source_col],
            self.var_indices[target_col],
        )

    def is_basic_feasible(self):
        return all(bi >= 0 for bi in self.b)

    def normalize(self, row, col):
        # Make a particular position 1 and 0 for all other rows in this column
        pivot_value = self.tableau[row, col]
        self.scale_row(row, Rational(1, pivot_value))
        for row_idx in range(self.tableau.rows):
            if row_idx != row:
                scale_factor = -self.tableau[row_idx, col]
                self.add_rows(row_idx, row, scale_factor)

    def pivot(self, col_non_basic, col_basic):
        row = col_basic  # Since identity matrix
        self.swap_cols(col_non_basic + 1, col_basic + 1)  # +1 for b column
        self.normalize(row, col_basic + 1)
        if PRINT:
            print(
                "\n\n"
                f"Pivoting: moving variable in column {col_non_basic} into the"
                f" basis and moving variable in column {col_basic} out."
                "\n\n"
            )
            print(self.latex_string())

    def solve_lp(self):
        # This assumes you have a basic feasible solution
        assert self.is_basic_feasible()

        while any(ci > 0 for ci in self.c):
            # Pick the first index in c that's positive
            selected_var = next(idx for idx, ci in enumerate(self.c) if ci > 0)
            var_col = self.tableau[:-1, selected_var + 1]

            # Filter out non-positive entries in var_col
            valid_indices = [i for i, val in enumerate(var_col) if val > 0]

            if not valid_indices:
                raise Exception("Problem is unbounded.")

            ratios = [(self.b[i] / var_col[i], i) for i in valid_indices]

            # Find the index of the minimum ratio
            var_leave = min(ratios)[1]

            self.pivot(selected_var, var_leave)

    def make_basic_feasible(self):
        if self.is_basic_feasible():
            return self.tableau

        # Add an auxiliary column for handling feasibility
        rows, cols = self.tableau.shape
        save_c = self.c.copy()
        save_indices = self.var_indices.copy()

        s_col = -sp.ones(rows, 1)
        self.tableau = self.tableau.row_join(s_col)
        self.var_types.append(S_VAR)
        s_index = self.var_indices[-1] + 1
        self.var_indices.append(s_index)

        # Set the objective function value for the auxiliary problem
        self.tableau[-1, :] = zeros(1, self.tableau.shape[1])
        self.tableau[-1, -1] = -1  # Set value for the auxiliary variable

        if PRINT:
            print("\nAuxiliary Tableau:\n")
            print(self.latex_string())

        # Find the row with the most negative value in b (to be the entering row)
        min_b_value = min(self.b)
        min_b_idx = [i for i, bi in enumerate(self.b) if bi == min_b_value][0]

        # Swap columns to make s a basic variable
        self.swap_cols(min_b_idx + 1, -1)  # Swap with last column (s variable)
        self.normalize(min_b_idx, min_b_idx + 1)

        if PRINT:
            print("\nAuxiliary Tableau with BFS:\n")
            print(self.latex_string())

        self.solve_lp()

        if PRINT:
            print("\nAuxiliary Tableau Solved:\n")
            print(self.latex_string())

        # Remove the auxiliary variable
        s_index = self.var_types.index(S_VAR)
        if s_index <= self.num_constraints:
            # s is a basic variable, need to pivot it out
            non_basic_indices = [
                i for i in range(1, len(self.var_types)) if self.var_types[i] != S_VAR
            ]
            for idx in non_basic_indices:
                if self.tableau[s_index - 1, idx] != 0:
                    self.pivot(idx - 1, s_index - 1)
                    break
            if PRINT:
                print("INFO: s was in basic variables and had to be moved out.")

        s_index = self.var_types.index(S_VAR)
        self.tableau.col_del(s_index)
        del self.var_types[s_index]
        del self.var_indices[s_index]

        # Restore the original objective function
        self.tableau[-1, :] = zeros(1, self.tableau.shape[1])
        for i in range(1, len(save_indices)):
            col_pos = self.var_indices.index(save_indices[i])
            coef = save_c[i - 1]
            if col_pos >= 0:
                self.tableau[-1, col_pos] = coef
            else:
                print("INFO: column missing from coefficients?")

        # Adjust the last row to make coefficients of basic variables zero
        for i in range(self.num_constraints):
            pos_val = self.tableau[-1, i + 1]
            if pos_val != 0:
                self.add_rows(-1, i, -pos_val)

    def display(self):
        """
        Displays the variables along with their indices and the tableau.
        """

        formatted_vars = []

        for var_type, var_index in zip(self.var_types, self.var_indices):
            if var_type == SLACK_VAR:
                formatted_vars.append(f"y{var_index}")
            elif var_type == REAL_VAR:
                formatted_vars.append(f"x{var_index - self.num_constraints}")
            elif var_type == S_VAR:
                formatted_vars.append("s")
            else:
                formatted_vars.append("")

        col_width = 10

        # Create the formatted header row for the tableau
        header = "".join(f"{var:>{col_width}}" for var in formatted_vars)
        print(header)

        for row in self.tableau.tolist():
            # Convert each value in the row to a string to avoid formatting errors
            formatted_row = "".join(
                f"{str(simplify(value)):>{col_width}}" for value in row
            )
            print(formatted_row)
        print()

    def latex_string(self):
        """
        Converts the tableau to a LaTeX formatted string.
        Returns a LaTeX string that can be used to display the tableau.
        """
        # Header of the LaTeX table
        latex_str = (
            "\\begin{gather}\n\\begin{array}{|r|"
            + "r" * (self.tableau.rows - 1)
            + "|"
            + "r" * (-self.tableau.rows + self.tableau.cols)
            + "|}\n\\hline\n"
        )

        # Format the variable names for the LaTeX header row
        header_vars = []
        for var_type, var_index in zip(self.var_types, self.var_indices):
            if var_type == SLACK_VAR:
                header_vars.append(f"y_{{{var_index}}}")
            elif var_type == REAL_VAR:
                header_vars.append(f"x_{{{var_index - self.num_constraints}}}")
            elif var_type == S_VAR:
                header_vars.append("s")
            else:
                header_vars.append(" ")

        latex_str += " & ".join(header_vars) + " \\\\\n\\hline\n"

        # Populate the table rows
        rows = self.tableau.tolist()
        for i, row in enumerate(rows):
            row_str = " & ".join(str(simplify(value)) for value in row)
            latex_str += row_str + " \\\\\n"
            if (i == len(rows)-2):
                latex_str += "\\hline\n"

        # Footer of the LaTeX table
        latex_str += "\\hline\n\\end{array}\n\\end{gather}\n"

        return latex_str
