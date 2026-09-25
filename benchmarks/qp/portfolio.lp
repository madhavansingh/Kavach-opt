\ Portfolio optimization with 3 assets
Minimize
 obj: - 0.1 x1 - 0.15 x2 - 0.12 x3 + [ 2 x1 ^ 2 + 1.2 x1 * x2 + 2 x2 ^ 2 + 0.8 x2 * x3 + 2 x3 ^ 2 ] / 2
Subject To
 budget: x1 + x2 + x3 = 1
Bounds
 0 <= x1 <= 1
 0 <= x2 <= 1
 0 <= x3 <= 1
End
