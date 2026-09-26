#!/usr/bin/env python3
import json
import os
import subprocess
import sys

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kavach_bin = os.path.join(repo_root, "build-release", "kavach-opt")
    if not os.path.exists(kavach_bin):
        sys.exit(f"Error: binary {kavach_bin} not found. Build release first.")

    models = [
        # Netlib
        ("afiro", "benchmarks/netlib/afiro.mps"),
        ("adlittle", "benchmarks/netlib/adlittle.mps"),
        ("blend", "benchmarks/netlib/blend.mps"),
        ("sc50a", "benchmarks/netlib/sc50a.mps"),
        ("sc50b", "benchmarks/netlib/sc50b.mps"),
        ("kb2", "benchmarks/netlib/kb2.mps"),
        ("share2b", "benchmarks/netlib/share2b.mps"),
        ("sc105", "benchmarks/netlib/sc105.mps"),
        ("stocfor1", "benchmarks/netlib/stocfor1.mps"),
        ("share1b", "benchmarks/netlib/share1b.mps"),
        ("lotfi", "benchmarks/netlib/lotfi.mps"),
        ("beaconfd", "benchmarks/netlib/beaconfd.mps"),
        ("boeing2", "benchmarks/netlib/boeing2.mps"),
        # MIPLIB
        ("knapsack", "benchmarks/miplib/knapsack.lp"),
        ("facility_loc", "benchmarks/miplib/facility_loc.lp"),
        ("infeasible_milp", "benchmarks/miplib/infeasible_milp.lp"),
        # QP
        ("portfolio", "benchmarks/qp/portfolio.lp"),
        ("constrained_ls", "benchmarks/qp/constrained_ls.lp"),
        # MRPL
        ("mrpl_blending_capacity", "examples/mrpl/mrpl_blending_capacity.lp"),
    ]

    decisions = []

    for name, rel_path in models:
        full_path = os.path.join(repo_root, rel_path)
        if not os.path.exists(full_path):
            print(f"Skipping {name}: {rel_path} not found")
            continue

        cmd = [kavach_bin, "solve", full_path, "--json"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
        if res.returncode != 0:
            print(f"Error solving {name}: {res.stderr}")
            continue

        try:
            data = json.loads(res.stdout)
            strat = data.get("strategy", {})
            sol = data.get("result", {})
            entry = {
                "instance_name": name,
                "instance_path": rel_path,
                "strategy_version": strat.get("strategy_version", "1.0.0"),
                "problem_class": strat.get("problem_class", "UNKNOWN"),
                "decision": {
                    "presolve_policy": strat.get("presolve_policy"),
                    "scaling_enabled": strat.get("scaling_enabled"),
                    "scaling_policy": strat.get("scaling_policy"),
                    "numerical_risk_profile": strat.get("numerical_risk_profile"),
                    "lp_refactorization_interval": strat.get("lp_refactorization_interval"),
                    "lp_force_bland_rule": strat.get("lp_force_bland_rule"),
                    "recommended_backend": strat.get("recommended_backend"),
                    "recommended_cpu_threads": strat.get("recommended_cpu_threads"),
                    "gpu_eligible_for_batched_spmv": strat.get("gpu_eligible_for_batched_spmv"),
                    "gpu_routing_reason": strat.get("gpu_routing_reason"),
                },
                "rules_fired": strat.get("rules_fired", []),
                "rule_explanations": strat.get("rule_explanations", []),
                "human_readable_rationale": strat.get("human_readable_rationale", ""),
                "solve_summary": {
                    "status": sol.get("status"),
                    "objective": sol.get("objective"),
                    "iterations": sol.get("iterations"),
                    "nodes": sol.get("nodes"),
                    "solve_time_seconds": sol.get("solve_time_seconds"),
                    "verified": sol.get("verified"),
                },
                "strategy_time_ms": strat.get("total_strategy_time_ms", 0.0)
            }
            decisions.append(entry)
            print(f"Captured strategy for {name} ({strat.get('problem_class')}) - {len(strat.get('rules_fired', []))} rules")
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON for {name}: {e}\nOutput was: {res.stdout}")

    out_dir = os.path.join(repo_root, "benchmark_results", "strategy")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "strategy_decisions.json")

    catalog = {
        "benchmark_suite": "Kavach-Opt Solver Strategy Corpus",
        "version": "1.0.0",
        "description": "Exhaustive deterministic solver strategy decisions across standard corpora",
        "total_instances": len(decisions),
        "decisions": decisions
    }

    with open(out_file, "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"\nSuccessfully generated {out_file} with {len(decisions)} instance decisions.")

if __name__ == "__main__":
    main()
