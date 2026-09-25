\ Constrained least squares
Minimize
 obj: - 6 x1 - 4 x2 + [ 4 x1 ^ 2 + 2 x1 * x2 + 4 x2 ^ 2 ] / 2
Subject To
 c1: x1 + x2 <= 2
Bounds
 0 <= x1
 0 <= x2
End
