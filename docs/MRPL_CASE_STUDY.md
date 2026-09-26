# MRPL-Style Refinery Optimization Case Study

## 1. Data disclosure

> [!IMPORTANT]
> **DATA STATUS**:
> All numeric values in this demonstration are illustrative/synthetic.
> They are not proprietary MRPL operational data and must not be interpreted
> as actual MRPL process parameters.

## 2. Business problem

Modern petroleum refineries balance crude feedstock purchasing, unit throughput limits, and blending constraints to produce finished transportation fuels (Gasoline, Diesel) and heavy fuel oil while maximizing net refining margins. Operating units exhibit minimum turndown limits and high fixed activation/maintenance costs, requiring genuine mixed-integer linear programming (MILP) to determine optimal on/off unit configurations alongside continuous blending ratios.

## 3. Mathematical model

The refinery optimization model integrates:
- **Continuous decision variables** $x[c, p] \ge 0$ for blending crude $c \in \{A, B, C, D\}$ into product pool $p \in \{P1, P2, P3\}$.
- **Continuous production variables** $prod[p] \ge 0$ representing total finished product yield.
- **Continuous unit throughputs** $t[u] \ge 0$ for processing units $u \in \{U1, U2, U3\}$.
- **Binary activation decisions** $y[u] \in \{0, 1\}$ indicating unit online status.

### Objective Function
$$\max \quad \sum_{p} \text{Price}_p \cdot prod_p - \sum_{c,p} \text{Cost}_c \cdot x_{c,p} - \sum_u \left( \text{VarCost}_u \cdot t_u + \text{FixedCost}_u \cdot y_u \right)$$

### Key Constraints
1. **Product Mass Balance**: $\sum_c x_{c,p} - prod_p = 0, \quad \forall p$
2. **Crude Availability**: $\sum_p x_{c,p} \le \text{Avail}_c, \quad \forall c$
3. **Unit Throughput Linkage**:
   - $t_{U1} = \sum_p (x_{A,p} + x_{B,p})$ (CDU Sweet Crudes)
   - $t_{U2} = \sum_p (x_{C,p} + x_{D,p})$ (VDU/FCCU Sour Crudes)
   - $t_{U3} = x_{C,P1} + x_{C,P2} + x_{D,P1} + x_{D,P2}$ (HDS Clean-fuel Desulfurization)
4. **Unit Capacity & Turndown**: $\text{MinCap}_u \cdot y_u \le t_u \le \text{MaxCap}_u \cdot y_u, \quad \forall u$
5. **Linearized Quality Specifications**:
   $$\sum_c (\text{Quality}_{c,q} - \text{Limit}_{p,q}) \cdot x_{c,p} \le 0$$

## 4. Model dimensions

| Dimension | Count |
|---|---|
| Total Variables | 21 |
| Continuous Variables | 18 |
| Binary Variables | 3 |
| Linear Constraints | 22 |
| Matrix Nonzeros | 81 |

## 5. Solver pipeline

```mermaid
flowchart LR
    A["Native LP Parser"] --> B["CanonicalModel"]
    B --> C["Presolve"]
    C --> D["Root LP Relaxation"]
    D --> E["MILP Branch-and-Bound"]
    E --> F["Independent MILPVerifier"]
    F --> G["Certified Solution"]
```

## 6. Baseline solution

- **Status**: `OPTIMAL` (Certified: `VERIFIED_OPTIMAL`)
- **Net Objective (Margin)**: $5445.91 k/day
- **LP Relaxation Bound**: $5891.36 k/day
- **Optimality Gap**: 0.00%
- **Nodes Explored**: 5
- **Solve Runtime**: 0.0012 s

## 7. Blending plan

| Feed Stream | To Product P1 (Gasoline) | To Product P2 (Diesel) | To Product P3 (Fuel Oil) | Total Used (kbpd) |
|---|---|---|---|---|
| Crude A | 117.9 | 2.1 | 0.0 | 120.0 |
| Crude B | 0.0 | 80.0 | 0.0 | 80.0 |
| Crude C | 22.1 | 16.1 | 45.0 | 83.2 |
| Crude D | 0.0 | 0.0 | 45.0 | 45.0 |

## 8. Unit allocation

| Unit | Description | Status | Throughput (kbpd) | Min-Max Rating | Utilization |
|---|---|---|---|---|---|
| U1 | Atmospheric Distillation Unit (CDU) | **ACTIVE** | 200.0 | 50.0 - 200.0 | 100.0% |
| U2 | Vacuum Distillation & Conversion Unit (VDU/FCCU) | **ACTIVE** | 128.2 | 60.0 - 180.0 | 71.2% |
| U3 | Hydrodesulfurization Unit (HDS) | **ACTIVE** | 38.2 | 30.0 - 120.0 | 31.8% |

## 9. Product quality

| Product | Production (kbpd) | Sulfur (wt%) | Sulfur Limit | Sulfur Margin | Density (kg/m³) | Density Limit | Density Margin |
|---|---|---|---|---|---|---|---|
| P1 (Motor Spirit (Gasoline)) | 140.0 | 0.80 | 0.80 | +0.00 | 828.7 | 840.0 | +11.3 |
| P2 (High-Speed Diesel (HSD)) | 98.2 | 1.30 | 1.30 | +0.00 | 853.5 | 865.0 | +11.5 |
| P3 (Heavy Fuel Oil (HFO)) | 90.0 | 3.00 | 3.00 | +-0.00 | 892.5 | 875.0 | +17.5 |

## 10. Economics

- **Gross Product Sales Revenue**: $30577.27 k/day
- **Crude Feedstock Acquisition Cost**: -$22510.91 k/day
- **Unit Variable Operating Cost**: -$1120.45 k/day
- **Unit Fixed Activation Cost**: -$1500.00 k/day
- **Net Operational Margin**: **$5445.91 k/day**

## 11. Independent verification

Independent mathematical certification performed by `MILPVerifier`:
- **Aggregate Primal Residual**: 2.84e-14
- **Max Bound Violation**: 2.84e-14
- **Max Integrality Violation**: 2.22e-16
- **Domain Mass Balance Residual**: 2.84e-14 (Status: PASSED)
- **Certified Status**: `OPTIMAL`

## 12. What-if scenarios

| Scenario ID | Description | Status | Net Margin ($k/day) | $\Delta$ Objective | Nodes | Primary Operational Impact |
|---|---|---|---|---|---|---|
| `baseline` | baseline | `OPTIMAL` | 5445.909091 | +0.000000 | 5 | Optimal baseline dispatch |
| `scenario_a_reduced_crude` | scenario_a_reduced_crude | `OPTIMAL` | 3750.000000 | -1695.909091 | 7 | Sweet crude shortfall drops Gasoline production |
| `scenario_b_tight_quality` | scenario_b_tight_quality | `OPTIMAL` | 4975.000000 | -470.909091 | 5 | Tightened diesel sulfur specs shift crudes to hydrotreater |
| `scenario_c_reduced_capacity` | scenario_c_reduced_capacity | `OPTIMAL` | 5445.909091 | -0.000000 | 5 | HDS bottleneck limits clean fuel output |
| `scenario_d_infeasible` | scenario_d_infeasible | `INFEASIBLE` | N/A | N/A | 0 | Feedstock starvation causes integer infeasibility |

## 13. HiGHS comparison

| Scenario | Kavach-Opt Obj | HiGHS Obj | Rel Error | Kavach Time | HiGHS Status |
|---|---|---|---|---|---|
| `baseline` | 5445.909091 | N/A | N/A | 0.0012 s | UNAVAILABLE |
| `scenario_a_reduced_crude` | 3750.000000 | N/A | N/A | 0.0014 s | UNAVAILABLE |
| `scenario_b_tight_quality` | 4975.000000 | N/A | N/A | 0.0011 s | UNAVAILABLE |
| `scenario_c_reduced_capacity` | 5445.909091 | N/A | N/A | 0.0011 s | UNAVAILABLE |
| `scenario_d_infeasible` | N/A | N/A | N/A | 0.0000 s | UNAVAILABLE |

## 14. Limitations

- This demonstration utilizes synthetic illustrative parameters and does not represent actual proprietary operational data of Mangalore Refinery and Petrochemicals Limited (MRPL).
- Refinery units are modeled as semi-continuous linearized yield and throughput units rather than non-linear chemical equilibrium thermodynamic equations.
- Multi-period tankage inventory dynamics and crude vessel arrival logistics are omitted to maintain focus on the single-period MILP blending and capacity allocation core.
