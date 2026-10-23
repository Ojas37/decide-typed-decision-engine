"""Section 5: score decomposition and what-if simulations, OOF on train only, selfcheck formulas."""
from common import *   # noqa: F401,F403

WEIGHTS = SC.WEIGHTS
GRID_A = np.round(np.arange(0.50, 0.951, 0.01), 2)
GRID_G = np.round(np.arange(0.30, 0.701, 0.01), 2)


def cov_scale(c):
    return min(1.0, c / 0.8)


def score_from(P, yr, pa, ya, pg, yg, r_abst, a_abst, g_abst, a_thr, g_thr):
    """selfcheck's scoring on OOF predictions (latency = full marks assumed => +20)."""
    n = len(yr)
    top = P.argmax(1); conf = P.max(1); corr = (top == yr).astype(float)
    truth = [Q[i] for i in yr]
    ans = ~r_abst
    pairs = [(truth[i], Q[top[i]]) for i in range(n) if ans[i]]
    maj = Counter_most(pairs)
    f1 = SC.macro_f1(pairs, Q); triv = SC.macro_f1([(t, maj) for t, _ in pairs], Q)
    q_r = SC.skill(f1, triv) * cov_scale(ans.mean())
    q_a = T.approve_quality(ya, pa, a_thr, a_abst)
    q_g = T.grounded_quality(yg, pg, g_thr, g_abst)
    quality = (q_r + q_a + q_g) / 3
    # calibration
    rm = route_metrics(P, yr); am = binary_metrics(pa, ya); gm = binary_metrics(pg, yg)
    mean_ece = (rm["ece15_top"] + am["ece15"] + gm["ece15"]) / 3
    skills = [rm["brier_skill"], am["brier_skill"], gm["brier_skill"]]
    calibration = 0.5 * max(0.0, 1 - mean_ece / 0.2) + 0.5 * sum(skills) / 3
    abst = (rm["abstention_part"] + am["abstention_part"] + gm["abstention_part"]) / 3
    total = 100 * (WEIGHTS["quality"] * quality + WEIGHTS["calibration"] * calibration
                   + WEIGHTS["abstention"] * abst + WEIGHTS["latency"] * 1.0)
    return {"total": total, "quality": quality, "calibration": calibration, "abstention": abst,
            "q": [q_r, q_a, q_g], "ece": [rm["ece15_top"], am["ece15"], gm["ece15"]], "skill": skills,
            "abst_parts": [rm["abstention_part"], am["abstention_part"], gm["abstention_part"]],
            "cov": [float(ans.mean()), float((~a_abst).mean()), float((~g_abst).mean())]}


def Counter_most(pairs):
    from collections import Counter
    return Counter(t for t, _ in pairs).most_common(1)[0][0]


def quantile_abstain(score_low_is_uncertain, cov):
    thr = np.quantile(score_low_is_uncertain, 1 - cov)
    return score_low_is_uncertain < thr


def pipeline(Z, yr, ya, yg, seed=13, cov=0.83, route_cal="temperature", app_cal="isotonic", gr_cal="platt",
             app_rule="min_conf", app_thr=None, cov_route=None, cov_app=None, cov_gr=None, P_override=None):
    cov_route = cov if cov_route is None else cov_route
    cov_app = cov if cov_app is None else cov_app
    cov_gr = cov if cov_gr is None else cov_gr
    P = crossfit_route_any(Z[:, :5], yr, route_cal, seed=seed) if P_override is None else P_override
    pa = crossfit_binary_any(Z[:, 5], ya, app_cal, seed=seed)
    pg = crossfit_binary_any(Z[:, 6], yg, gr_cal, seed=seed)
    r_abst = quantile_abstain(P.max(1), cov_route)
    if app_rule == "min_conf":
        a_abst = quantile_abstain(np.maximum(pa, 1 - pa), cov_app)
    elif app_rule == "max_risk":
        a_abst = quantile_abstain(-M.approve_risk(pa), cov_app)
    elif app_rule == "band_below_thr":       # abstain just under/around the decision threshold, nearest-first
        thr0 = app_thr if app_thr else 0.75
        a_abst = quantile_abstain(-np.abs(pa - (thr0 - 0.12)), cov_app)
    if app_thr is None:
        full = np.array([T.approve_cost(ya, pa >= t) for t in GRID_A])
        app_thr = T.pick_threshold(GRID_A, -full, 0.75)
    g_abst = quantile_abstain(np.maximum(pg, 1 - pg), cov_gr)
    q_g = np.array([T.grounded_quality(yg, pg, float(t), g_abst) for t in GRID_G])
    g_thr = T.pick_threshold(GRID_G, q_g, 0.5)
    out = score_from(P, yr, pa, ya, pg, yg, r_abst, a_abst, g_abst, app_thr, g_thr)
    out["app_thr"] = app_thr
    return out


def show(tag, r, base=None):
    d = "" if base is None else "  (%+.2f vs base)" % (r["total"] - base["total"])
    print("%-46s total %.2f | Q %.3f C %.3f A %.3f | qual[r/a/g] %.3f/%.3f/%.3f | abst[r/a/g] %.3f/%.3f/%.3f%s"
          % (tag, r["total"], r["quality"], r["calibration"], r["abstention"], *r["q"], *r["abst_parts"], d))


if __name__ == "__main__":
    out = {}
    print("=== 1. METHOD CHECK + SEED NOISE (deployed recipe; latency assumed full => +20)")
    res = []
    for seed in (13, 14, 15):
        Z, yr, ya, yg, fold = load_cache(seed)
        r = pipeline(Z, yr, ya, yg, seed=seed)
        res.append(r)
        show("seed %d" % seed, r)
    tot = np.array([r["total"] for r in res])
    print("OOF total over seeds: mean %.2f  std %.2f  range [%.2f, %.2f]   <- public-dev result you reported: 82.07" % (tot.mean(), tot.std(ddof=1), tot.min(), tot.max()))
    out["seed_totals"] = tot.tolist()

    Z, yr, ya, yg, fold = load_cache(13)
    base = pipeline(Z, yr, ya, yg)
    print("\n=== 2. WHERE THE POINTS ARE (seed 13).  weight x axis = points; headroom = (1-axis) x weight")
    for k, w in WEIGHTS.items():
        v = {"quality": base["quality"], "calibration": base["calibration"], "abstention": base["abstention"], "latency": 1.0}[k]
        print("  %-11s axis %.3f  x %2.0f = %5.2f pts   headroom %.2f pts" % (k, v, 100 * w, 100 * w * v, 100 * w * (1 - v)))
    print("  quality parts  route %.3f  approve %.3f  grounded %.3f  -> headroom pts: %.2f / %.2f / %.2f (each worth 40/3)"
          % (*base["q"], *[40 / 3 * (1 - q) for q in base["q"]]))
    print("  calibration: mean ECE parts %s  Brier skills %s" % (np.round(base["ece"], 4), np.round(base["skill"], 3)))
    print("  abstention parts route/approve/grounded: %s  (each worth 5 pts x part)" % np.round(base["abst_parts"], 3))

    print("\n=== 3. WHAT-IFS on the SAME OOF methodology (not dev claims)")
    show("deployed recipe (cov .83, T / iso / platt)", base)
    for c in (0.75, 0.78, 0.80, 0.85, 0.90, 1.00):
        show("coverage target %.2f (all three)" % c, pipeline(Z, yr, ya, yg, cov=c), base)
    for rc in ("raw", "ovr_platt", "ovr_isotonic"):
        show("route calibrator = %s" % rc, pipeline(Z, yr, ya, yg, route_cal=rc), base)
    for ac in ("raw", "temperature", "platt"):
        show("approve calibrator = %s" % ac, pipeline(Z, yr, ya, yg, app_cal=ac), base)
    for gc in ("raw", "temperature", "isotonic"):
        show("grounded calibrator = %s" % gc, pipeline(Z, yr, ya, yg, gr_cal=gc), base)
    for rule in ("max_risk", "band_below_thr"):
        show("approve abstain rule = %s" % rule, pipeline(Z, yr, ya, yg, app_rule=rule), base)
    for t in (0.50, 0.60, 0.70, 0.73, 0.75, 0.78, 0.80):
        show("approve decision threshold = %.2f" % t, pipeline(Z, yr, ya, yg, app_thr=t), base)
    show("coverage route .80 / approve .83 / grounded .83", pipeline(Z, yr, ya, yg, cov_route=.80), base)
    show("route .83 / approve .80 / grounded .83", pipeline(Z, yr, ya, yg, cov_app=.80), base)
    show("route .83 / approve .83 / grounded .80", pipeline(Z, yr, ya, yg, cov_gr=.80), base)
    show("coverage .80 on all three", pipeline(Z, yr, ya, yg, cov=0.80), base)

    print("\n=== 4. UPPER-BOUND SIMULATIONS (oracle edits to OOF predictions; NOT achievable promises)")
    rows = load_train()
    P0 = crossfit_route_any(Z[:, :5], yr, "temperature")
    import re
    mk = re.compile("Priority for me is:|My real question is this:|The part I need you to action is:|The main thing I need resolved is:|What I actually need help with is this:|Most importantly:|Less important, but while I am writing:|Some context first:|For background,|Separately, and this is less urgent,|On a related note,|Also worth mentioning,")
    has_mk = np.array([bool(mk.search(r["state"]["body"])) for r in rows])
    wrong = (P0.argmax(1) != yr) & has_mk
    P1 = P0.copy()
    for i in np.where(wrong)[0]:           # give the TRUE class the confidence the model had in its wrong answer
        top = P1[i].argmax(); P1[i, [top, yr[i]]] = P1[i, [yr[i], top]]
    print("   marker-row route errors flipped to correct (%d of %d total errors)" % (wrong.sum(), (P0.argmax(1) != yr).sum()))
    show("ORACLE: fix all marker-row route errors", pipeline(Z, yr, ya, yg, P_override=P1), base)
    # same, but ALSO assume the model's confidence in those rows rises to the typical correct level (0.95)
    P2 = P1.copy()
    for i in np.where(has_mk & (P0.argmax(1) == yr))[0]:
        pass
    # grounded: remove half of the errors, smallest-confidence first stays; simulate by flipping 50% of FP/FN logits
    Zg = Z.copy()
    pg0 = crossfit_binary_any(Z[:, 6], yg, "platt")
    err_idx = np.where((pg0 >= 0.5) != yg)[0]
    rng = np.random.RandomState(0)
    fix = rng.choice(err_idx, size=len(err_idx) // 2, replace=False)
    Zg[fix, 6] = np.where(yg[fix], 4.0, -4.0)
    show("ORACLE: halve grounded errors", pipeline(Zg, yr, ya, yg), base)
    Zg2 = Z.copy(); Zg2[err_idx, 6] = np.where(yg[err_idx], 4.0, -4.0)
    show("ORACLE: remove ALL grounded errors", pipeline(Zg2, yr, ya, yg), base)
    save_json("score_whatifs.json", out)
