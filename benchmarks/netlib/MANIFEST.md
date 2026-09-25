# Netlib Benchmark Corpus Manifest

This directory contains 13 authentic Netlib Linear Programming benchmark instances retrieved directly from the official Netlib repository (`https://netlib.org/lp/data/`) and expanded using the standard Netlib decompression tool `emps.c`.

## Official Published Metadata

The following table lists the official metadata published in the Netlib `readme` table:

| Instance | Netlib Rows (incl. Obj) | Netlib Columns | Total Nonzeros (incl. Obj) | Features | Optimal Objective Value |
|---|---|---|---|---|---|
| `afiro.mps` | 28 | 32 | 88 | Fixed MPS | -464.75314286 |
| `adlittle.mps` | 57 | 97 | 465 | Fixed MPS | 225494.96316 |
| `blend.mps` | 75 | 83 | 521 | Unlabelled RHS | -30.812149846 |
| `sc50a.mps` | 51 | 48 | 131 | Fixed MPS | -64.575077059 |
| `sc50b.mps` | 51 | 48 | 119 | Fixed MPS | -70.000000000 |
| `share2b.mps` | 97 | 79 | 730 | Fixed MPS | -415.73224074 |
| `beaconfd.mps` | 174 | 262 | 3476 | Large LP | 33592.485807 |
| `sc105.mps` | 106 | 103 | 281 | Fixed MPS | -52.202061212 |
| `lotfi.mps` | 154 | 308 | 1086 | Fixed MPS | -25.264706062 |
| `share1b.mps` | 118 | 225 | 1182 | Fixed MPS | -76589.318579 |
| `kb2.mps` | 44 | 41 | 291 | Empty RHS, BOUNDS | -1749.9001299 |
| `stocfor1.mps` | 118 | 111 | 474 | Stochastic LP | -41131.976219 |
| `boeing2.mps` | 167 | 143 | 1339 | BOUNDS & RANGES | -315.01872802 |

## Canonical Model Mapping

In Kavach-Opt's `CanonicalModel` representation [A]:
1. **Variables (`num_variables`)**: Equals Netlib Columns count.
2. **Objective (`objective`)**: The objective row (row type `N`, e.g. `COST` or first `N` row) is separated from constraints and stored in `model.objective()`.
3. **Constraints (`num_constraints`)**:
   - For instances without RANGES: `CanonicalModel::num_constraints() == Netlib_Rows - 1`.
   - For instances with RANGES (e.g. `boeing2`): ranged rows are canonically expanded into dual bounding constraints ($a^T x \le b_u$ and $a^T x \ge b_l$), preserving the exact feasible region without slacks.
4. **Total Nonzeros**: Sum of constraint nonzeros (`CanonicalModel::num_nonzeros()`) plus objective linear terms nonzeros (`CanonicalModel::objective().linear_terms.num_terms()`) equals Netlib Total Nonzeros.
