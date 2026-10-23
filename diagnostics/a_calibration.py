"""Section 4: calibrator comparison on cross-fitted OOF predictions (train only)."""
from common import *   # noqa: F401,F403

Z, yr, ya, yg, fold = load_cache(13)
out = {"route": {}, "approve": {}, "grounded": {}}

print("=== ROUTE (top-label ECE as in selfcheck; Brier skill vs class prior)")
print("%-14s %7s %7s %7s %8s %7s %8s %7s" % ("method", "ECE15", "ECEeqm", "Brier", "BrSkill", "NLL", "risk@80", "AbsPt"))
for kind in ("raw", "temperature", "ovr_platt", "ovr_isotonic"):
    P = crossfit_route_any(Z[:, :5], yr, kind)
    m = route_metrics(P, yr)
    out["route"][kind] = m
    out["route"][kind]["reliability_top_label"] = reliability(P.max(1), (P.argmax(1) == yr).astype(float))
    print("%-14s %7.4f %7.4f %7.4f %8.4f %7.4f %8.4f %7.3f   acc %.4f auroc(conf) %.3f"
          % (kind, m["ece15_top"], m["ece15_equal_mass"], m["brier"], m["brier_skill"], m["nll"],
             m["risk80"], m["abstention_part"], m["acc"], m["auroc_conf_vs_correct"]))

for name, z, y, current in (("approve", Z[:, 5], ya, "isotonic"), ("grounded", Z[:, 6], yg, "platt")):
    print("\n=== %s   (current deployed calibrator: %s)" % (name.upper(), current))
    print("%-12s %7s %7s %7s %8s %7s %7s %8s %7s %7s" %
          ("method", "ECE15", "ECEeqm", "Brier", "BrSkill", "NLL", "AUROC", "risk@80", "AbsPt", "uniqP"))
    for kind in ("raw", "temperature", "platt", "isotonic"):
        p = crossfit_binary_any(z, y, kind)
        m = binary_metrics(p, y)
        m["reliability"] = reliability(p, y.astype(float))
        out[name][kind] = m
        print("%-12s %7.4f %7.4f %7.4f %8.4f %7.4f %7.4f %8.4f %7.3f %7d%s"
              % (kind, m["ece15"], m["ece15_equal_mass"], m["brier"], m["brier_skill"], m["nll"],
                 m["auroc_label"], m["risk80"], m["abstention_part"], m["unique_p"],
                 "  <- current" if kind == current else ""))

print("\nReliability bins, CURRENT calibrators (cross-fitted):")
for name, z, y, kind in (("approve", Z[:, 5], ya, "isotonic"), ("grounded", Z[:, 6], yg, "platt")):
    print(" " + name)
    for b in out[name][kind]["reliability"]:
        print("   %-8s n=%4d  mean_p=%.3f  observed=%.3f  gap=%+.3f" %
              (b["bin"], b["n"], b["mean_p"], b["observed"], b["observed"] - b["mean_p"]))
print(" route (top label)")
for b in out["route"]["temperature"]["reliability_top_label"]:
    print("   %-8s n=%4d  mean_conf=%.3f  accuracy=%.3f  gap=%+.3f" %
          (b["bin"], b["n"], b["mean_p"], b["observed"], b["observed"] - b["mean_p"]))
save_json("calibration_comparison.json", out)
