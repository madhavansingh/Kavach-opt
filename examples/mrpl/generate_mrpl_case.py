#!/usr/bin/env python3
"""
MRPL Industrial Demonstration Model Generator.

DATA STATUS:
All numeric values in this demonstration are illustrative/synthetic.
They are not proprietary MRPL operational data and must not be
interpreted as actual MRPL process parameters.
"""

import sys
import os

DISCLOSURE = """\\ ==============================================================================
\\ MRPL Industrial Case Study: Refinery Crude Blending and Unit Allocation
\\
\\ DATA STATUS:
\\ All numeric values in this demonstration are illustrative/synthetic.
\\ They are not proprietary MRPL operational data and must not be interpreted
\\ as actual MRPL process parameters.
\\ ==============================================================================
"""

def generate_lp(output_path="examples/mrpl/mrpl_blending_capacity.lp"):
    content = f"""{DISCLOSURE}
Maximize
 obj: 110 prod_P1 + 95 prod_P2 + 65 prod_P3 - 80 x_A_P1 - 80 x_A_P2 - 80 x_A_P3 - 72 x_B_P1 - 72 x_B_P2 - 72 x_B_P3 - 60 x_C_P1 - 60 x_C_P2 - 60 x_C_P3 - 48 x_D_P1 - 48 x_D_P2 - 48 x_D_P3 - 2.5 t_U1 - 3.5 t_U2 - 4.5 t_U3 - 500 y_U1 - 600 y_U2 - 400 y_U3

Subject To
 \\ Product pool mass balance
 bal_P1: x_A_P1 + x_B_P1 + x_C_P1 + x_D_P1 - prod_P1 = 0
 bal_P2: x_A_P2 + x_B_P2 + x_C_P2 + x_D_P2 - prod_P2 = 0
 bal_P3: x_A_P3 + x_B_P3 + x_C_P3 + x_D_P3 - prod_P3 = 0

 \\ Crude availability limits
 avail_A: x_A_P1 + x_A_P2 + x_A_P3 <= 120
 avail_B: x_B_P1 + x_B_P2 + x_B_P3 <= 100
 avail_C: x_C_P1 + x_C_P2 + x_C_P3 <= 140
 avail_D: x_D_P1 + x_D_P2 + x_D_P3 <= 80

 \\ Unit throughput definitions
 def_t_U1: x_A_P1 + x_A_P2 + x_A_P3 + x_B_P1 + x_B_P2 + x_B_P3 - t_U1 = 0
 def_t_U2: x_C_P1 + x_C_P2 + x_C_P3 + x_D_P1 + x_D_P2 + x_D_P3 - t_U2 = 0
 def_t_U3: x_C_P1 + x_C_P2 + x_D_P1 + x_D_P2 - t_U3 = 0

 \\ Unit capacity limits (minimum turndown and maximum capacity)
 cap_max_U1: t_U1 - 200 y_U1 <= 0
 cap_min_U1: t_U1 - 50 y_U1 >= 0
 cap_max_U2: t_U2 - 180 y_U2 <= 0
 cap_min_U2: t_U2 - 60 y_U2 >= 0
 cap_max_U3: t_U3 - 120 y_U3 <= 0
 cap_min_U3: t_U3 - 30 y_U3 >= 0

 \\ Product quality: Sulfur upper bounds (Linearized: sum_c (sulfur_c - max_s_p) * x_c_p <= 0)
 q_sulfur_P1: - 0.3 x_A_P1 + 0.3 x_B_P1 + 1.6 x_C_P1 + 2.8 x_D_P1 <= 0
 q_sulfur_P2: - 0.8 x_A_P2 - 0.2 x_B_P2 + 1.1 x_C_P2 + 2.3 x_D_P2 <= 0
 q_sulfur_P3: - 2.5 x_A_P3 - 1.9 x_B_P3 - 0.6 x_C_P3 + 0.6 x_D_P3 <= 0

 \\ Product quality: Density bounds (Linearized: sum_c (density_c - bound_p) * x_c_p <= / >= 0)
 q_dens_max_P1: - 20 x_A_P1 + 10 x_B_P1 + 35 x_C_P1 + 70 x_D_P1 <= 0
 q_dens_max_P2: - 45 x_A_P2 - 15 x_B_P2 + 10 x_C_P2 + 45 x_D_P2 <= 0
 q_dens_min_P3: - 55 x_A_P3 - 25 x_B_P3 + 35 x_D_P3 >= 0

Bounds
 0 <= prod_P1 <= 140
 0 <= prod_P2 <= 160
 0 <= prod_P3 <= 90
 0 <= t_U1 <= 200
 0 <= t_U2 <= 180
 0 <= t_U3 <= 120

Binaries
 y_U1 y_U2 y_U3

End
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Generated MRPL model: {output_path}")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "examples/mrpl/mrpl_blending_capacity.lp"
    generate_lp(out)
