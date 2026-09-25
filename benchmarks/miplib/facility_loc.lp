\ Uncapacitated facility location MILP instance
Minimize
 cost: 100 y1 + 120 y2 + 150 y3 + 2 x11 + 4 x12 + 5 x21 + 2 x22 + 3 x31 + 3 x32
Subject To
 dem1: x11 + x21 + x31 = 10
 dem2: x12 + x22 + x32 = 15
 cap1: x11 + x12 - 25 y1 <= 0
 cap2: x21 + x22 - 25 y2 <= 0
 cap3: x31 + x32 - 25 y3 <= 0
Binaries
 y1 y2 y3
Bounds
 0 <= x11
 0 <= x12
 0 <= x21
 0 <= x22
 0 <= x31
 0 <= x32
End
