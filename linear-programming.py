"""Program to solve linear programming problems. This program uses the Simplex
Algorithm to find the optimal solution to the LP. In its current state, there
are inputs for which the algorithm will misbehave as no anti-cycling rules have
been put in place.

@author Suchith Sridhar Khajjayam
@date 27 Sep 2024
"""

from sympy import Matrix
from tableau import Tableau


A = [[4, -1, -2, 2], [-3, -1, -1, 1], [1, 2, 2, -2], [-1, -2, -2, 2], [4, 1, 3, -3]]
b = [20, -1, -8, 8, 21]
c = [-3, 4, -2, 2]

A = Matrix(A)
b = Matrix(b)
c = Matrix(c)
"""A = np.array([[-1, -2, -2], [-3, -2, -1], [1, 0, 3], [1, 1, 1]]) b =
np.array([-3, -5, 10, 9]) c = np.array([1, -2, 1])"""

tableau = Tableau(A, b, c)
print("\\subsection{Simplex Initialization}\n")
print("\n\nInitital Tableau:\n\n")
print(tableau.latex_string())

print()
print("Is inital BFS feasible: ", tableau.is_basic_feasible())
print()
tableau.make_basic_feasible()
print("\n\nNew equivalent tableau with feasible BFS:\n\n")
print(tableau.latex_string())

print("\n\\subsection{Simplex Optimization}\n")

tableau.solve_lp()
print("\n\nSolved Tableau:\n\n")
print(tableau.latex_string())
