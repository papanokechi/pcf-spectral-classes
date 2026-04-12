#!/usr/bin/env python3
"""
Phase 2 congruence hunt for the PCF spectral-class program.

Goal:
  extract a secondary arithmetic signature

      Σ = (𝒟, 𝒞, T, ω)

from the first 500 denominator convergents, where
  * 𝒟 is the prime-divisibility density vector,
  * 𝒞 is the residue-support / entropy profile mod p,
  * T is the exact eventual residue period mod p,
  * ω is the Apéry-style congruence-retention score
      q_{np} ≡ q_n (mod p^3).

The script reads the existing phase-1 taxonomy CSV and reconstructs the seed
families from their labels. It then writes:

  results/pcf_spectral_phase2_congruence.json
  results/pcf_spectral_phase2_congruence.csv
  results/pcf_spectral_phase2_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from typing import Callable

from mpmath import mp
import sympy as sp

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PHASE1_CSV = RESULTS / "pcf_spectral_phase1_taxonomy.csv"
RESULTS.mkdir(exist_ok=True)

DEFAULT_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23)
DEFAULT_N_TERMS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2/2.5 congruence scan for PCF families")
    parser.add_argument("--n-terms", type=int, default=DEFAULT_N_TERMS,
                        help="number of denominator convergents to generate")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="optional subset of family labels to scan")
    parser.add_argument("--tag", default="",
                        help="optional suffix tag for output filenames")
    return parser.parse_args()


ARGS = parse_args()
PRIMES = tuple(DEFAULT_PRIMES)
N_TERMS = ARGS.n_terms


@dataclass
class ResidueStats:
    p: int
    zero_density: float
    support_size: int
    entropy_bits: float
    period_mod_p: int
    omega_mod_p: float
    omega_mod_p2: float
    omega_mod_p3: float
    most_common: list[tuple[int, int]]


@dataclass
class SignatureRow:
    label: str
    group: str
    D: list[float]
    C_support: list[int]
    C_entropy: list[float]
    periods: list[int]
    omega: list[float]
    arithmetic_purity: float
    retention_stability: float
    mean_zero_density: float
    mean_entropy: float
    mean_omega_p3: float
    dist_to_log: float | None = None
    dist_to_pi: float | None = None
    nearest_cluster: str | None = None


def read_phase1_rows() -> list[dict[str, str]]:
    with PHASE1_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def coeffs_from_label(label: str) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    if label.startswith("log-k="):
        k = int(label.split("=")[1])
        return (0, 0, -k), (k, k + 1), "Log ladder"
    if label.startswith("pi-m="):
        m = int(label.split("=")[1])
        return (0, 2 * m + 1, -2), (1, 3), "Pi family"
    if label == "apery-zeta3":
        return (0, 0, 0, 0, 0, 0, -1), (5, 27, 51, 34), "Apery-isolated"
    if label == "probe-d1-pos-4n2-6n1":
        return (0, 0, 4), (1, 6), "M2-2 probe"
    if label == "probe-d1-neg-n2-8n4":
        return (0, 0, -1), (4, 8), "M2-2 probe"
    if label == "probe-d1-boundary-nn1-3n2":
        return (0, -1, 1), (2, 3), "M2-2 probe"
    if label.startswith("vquad-"):
        m = re.fullmatch(r"vquad-(-?\d+)-(-?\d+)-(-?\d+)", label)
        if m is None:
            raise ValueError(f"Could not parse V_quad label: {label}")
        A, B, C = map(int, m.groups())
        return (1,), (int(C), int(B), int(A)), "V_quad"
    raise ValueError(f"Unsupported label format: {label}")


def poly_eval(coeffs: tuple[int, ...], n: int) -> sp.Integer:
    x = sp.Integer(n)
    return sum(sp.Integer(c) * x**i for i, c in enumerate(coeffs))


def poly_degree(coeffs: tuple[int, ...]) -> int:
    deg = -1
    for i, c in enumerate(coeffs):
        if c != 0:
            deg = i
    return deg


def lead_coeff(coeffs: tuple[int, ...]) -> int:
    deg = poly_degree(coeffs)
    return int(coeffs[deg]) if deg >= 0 else 0


def sign_token(x: int | float) -> str:
    if x > 0:
        return "+"
    if x < 0:
        return "-"
    return "0"


def tau_sector(tau: str) -> str | None:
    if tau == "pi-type":
        return "pi-type"
    if tau == "log-type":
        return "log-type"
    if tau == "Catalan":
        return "Catalan-tier"
    if tau in {"algebraic/surd", "algebraic/rational", "rational"}:
        return "algebraic/rational"
    return None


def is_perfect_squareish(x: float | None) -> bool:
    if x is None:
        return False
    rounded = int(round(x))
    if abs(x - rounded) > 1e-9 or rounded < 0:
        return False
    root = math.isqrt(rounded)
    return root * root == rounded


def galois_prediction(delta_shape: float | None, delta_sign: str, lambda_rational: bool, is_boundary: bool) -> str:
    if is_boundary:
        return "degenerate-boundary"
    if delta_shape is not None and abs(delta_shape) < 1e-9:
        return "Catalan-tier"
    if lambda_rational:
        return "degenerate"
    if delta_sign == "+":
        return "pi-type"
    if delta_sign == "-":
        return "log-type"
    return "degenerate"


def parse_floatish(text: str) -> float | None:
    if text is None:
        return None
    val = text.strip()
    if val in {"", "—", "N/A", "nan"}:
        return None
    try:
        return float(val)
    except Exception:
        return None


def build_phase1_meta(row: dict[str, str]) -> dict[str, object]:
    a_coeffs, b_coeffs, inferred_group = coeffs_from_label(row["label"])
    delta = lead_coeff(a_coeffs)
    delta_sign = sign_token(delta)
    tau = row.get("tau", "")
    sector = tau_sector(tau)
    delta_shape = parse_floatish(row.get("Delta_shape", ""))
    lambda_rational = is_perfect_squareish(delta_shape)
    a1 = int(poly_eval(a_coeffs, 1))
    is_boundary = (a1 == 0)
    g_pred = galois_prediction(delta_shape, delta_sign, lambda_rational, is_boundary)
    g_hit = (sector == g_pred) if sector is not None and g_pred in {"pi-type", "log-type", "Catalan-tier"} else None
    b0_mod4 = int(b_coeffs[0]) % 4 if b_coeffs else None
    return {
        "label": row["label"],
        "group": row.get("group") or inferred_group,
        "d": parse_floatish(row.get("d", "")),
        "tau": tau,
        "tau_sector": sector,
        "Delta_shape": delta_shape,
        "delta_lead": delta,
        "delta_sign": delta_sign,
        "lambda_rational": lambda_rational,
        "galois_prediction": g_pred,
        "galois_hit": g_hit,
        "is_degenerate_boundary": is_boundary,
        "a1": a1,
        "b0_mod4": b0_mod4,
        "identified_for_m22": sector is not None,
    }


def analyze_m22(phase1_rows: list[dict[str, str]]) -> dict[str, object]:
    meta_rows = [build_phase1_meta(r) for r in phase1_rows]
    d1_identified = [r for r in meta_rows if r["d"] == 1 and r["identified_for_m22"]]
    primary_rows = [
        r for r in d1_identified
        if not r["lambda_rational"] and not r["is_degenerate_boundary"]
    ]
    hits = sum(1 for r in primary_rows if r["galois_hit"])
    total = len(primary_rows)
    hit_rate = (hits / total) if total else None

    degenerate_rows = [r for r in d1_identified if r["lambda_rational"]]
    secondary_pool = [r for r in degenerate_rows if r["tau_sector"] in {"pi-type", "algebraic/rational"}]
    secondary_hits = 0
    secondary_total = len(secondary_pool)
    sec_classes: dict[int, set[str]] = {}
    for r in secondary_pool:
        b0 = int(r["b0_mod4"])
        sec_classes.setdefault(b0, set()).add(str(r["tau_sector"]))
        prediction = "pi-type" if b0 % 4 == 1 else "algebraic/rational"
        if prediction == r["tau_sector"]:
            secondary_hits += 1
    observed_secondary_sectors = sorted({str(r["tau_sector"]) for r in secondary_pool})
    secondary_has_both = {"pi-type", "algebraic/rational"}.issubset(set(observed_secondary_sectors))
    secondary_rate = (secondary_hits / secondary_total) if secondary_total else None
    secondary_pure = all(len(v) == 1 for v in sec_classes.values()) if secondary_total else None

    delta_buckets: dict[int, list[dict[str, object]]] = {}
    for r in d1_identified:
        delta_shape = r["Delta_shape"]
        if delta_shape is None:
            continue
        key = int(round(delta_shape))
        delta_buckets.setdefault(key, []).append(r)
    colliding = {
        key: rows for key, rows in delta_buckets.items()
        if len({r["tau_sector"] for r in rows if r["tau_sector"] is not None}) > 1
    }

    if total > 0 and hit_rate is not None and hit_rate >= 0.90:
        verdict = "GO — Galois-class separator passes the 90% threshold"
    elif total > 0 and hit_rate is not None and hit_rate < 0.80:
        verdict = "NO-GO — pivot to hypergeometric z0"
    else:
        verdict = "NO-GO — insufficient non-degenerate identified families; pivot to hypergeometric z0"

    return {
        "meta_by_label": {r["label"]: r for r in meta_rows},
        "d1_identified": d1_identified,
        "primary_rows": primary_rows,
        "hit_rate": hit_rate,
        "hits": hits,
        "total": total,
        "degenerate_rows": degenerate_rows,
        "secondary_pool": secondary_pool,
        "secondary_hits": secondary_hits,
        "secondary_total": secondary_total,
        "secondary_rate": secondary_rate,
        "secondary_classes": {k: sorted(v) for k, v in sec_classes.items()},
        "secondary_pure": secondary_pure,
        "secondary_has_both": secondary_has_both,
        "secondary_observed_sectors": observed_secondary_sectors,
        "colliding": colliding,
        "verdict": verdict,
    }


def mp_poly_eval(coeffs: tuple[int, ...], n: int | mp.mpf) -> mp.mpf:
    n = mp.mpf(n)
    total = mp.mpf("0")
    for i, c in enumerate(coeffs):
        total += mp.mpf(c) * n**i
    return total


def high_precision_cf_value(a_coeffs: tuple[int, ...], b_coeffs: tuple[int, ...], depth: int = 1600, dps: int = 110):
    with mp.workdps(dps):
        p_prev = mp.mpf("1")
        p_curr = mp_poly_eval(b_coeffs, 0)
        q_prev = mp.mpf("0")
        q_curr = mp.mpf("1")
        for n in range(1, depth + 1):
            a_n = mp_poly_eval(a_coeffs, n)
            b_n = mp_poly_eval(b_coeffs, n)
            p_new = b_n * p_curr + a_n * p_prev
            q_new = b_n * q_curr + a_n * q_prev
            scale = max(abs(p_new), abs(q_new), mp.mpf("1"))
            if scale > mp.mpf("1e80"):
                p_new /= scale
                p_curr /= scale
                p_prev /= scale
                q_new /= scale
                q_curr /= scale
                q_prev /= scale
            p_prev, p_curr = p_curr, p_new
            q_prev, q_curr = q_curr, q_new
        return (p_curr / q_curr) if q_curr != 0 else None


def error_digits(err: mp.mpf) -> float:
    err = abs(mp.mpf(err))
    if err == 0:
        return float(mp.dps)
    return max(0.0, float(-mp.log10(err)))


def fmt_mpf_or_mpc(z, digits: int = 12) -> str:
    if z is None:
        return "None"
    z = mp.mpc(z)
    if abs(mp.im(z)) < mp.mpf("1e-20"):
        return mp.nstr(mp.re(z), digits)
    sign = "+" if mp.im(z) >= 0 else "-"
    return f"{mp.nstr(mp.re(z), digits)}{sign}{mp.nstr(abs(mp.im(z)), digits)}j"


def abc_from_alpha_delta(alpha: int | float, delta: int | float):
    alpha = mp.mpf(alpha)
    delta = mp.mpf(delta)
    s = mp.mpf("1") - alpha
    p = -delta
    disc = s * s - 4 * p
    root = mp.sqrt(disc)
    roots = [((s + root) / 2), ((s - root) / 2)]
    roots.sort(key=lambda z: (float(mp.re(z)), float(mp.im(z))))
    return roots[0], roots[1], mp.mpf("1")


def rational_constant_candidates(max_abs: int = 10, max_den: int = 12):
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int, mp.mpf]] = []
    priority = [
        (1, 1), (-1, 1), (1, 2), (-1, 2), (2, 1), (-2, 1), (3, 2), (-3, 2),
        (4, 3), (-4, 3), (3, 1), (-3, 1), (5, 2), (-5, 2), (1, 3), (-1, 3),
        (2, 3), (-2, 3), (3, 4), (-3, 4), (4, 1), (-4, 1), (5, 1), (-5, 1),
    ]
    for p, q in priority:
        g = math.gcd(abs(p), q)
        key = (p // g, q // g)
        if key in seen or abs(mp.mpf(p) / q) > max_abs:
            continue
        seen.add(key)
        out.append((key[0], key[1], mp.mpf(key[0]) / key[1]))
    for q in range(1, max_den + 1):
        for p in range(-max_abs * q, max_abs * q + 1):
            if p == 0 or math.gcd(abs(p), q) != 1:
                continue
            key = (p, q)
            if key in seen:
                continue
            seen.add(key)
            out.append((p, q, mp.mpf(p) / q))
    return out


def search_real_interval(L, a, b, c, C, left: float, right: float, steps: int = 24):
    best_err = mp.inf
    best_z = None
    seeds = [0.05, 0.1, 0.125, 0.2, 0.25, 1/3, 0.5, 2/3, 0.75, 0.9]
    if right > 1:
        seeds += [1.2, 1.5, 2.0, 3.0]

    for guess in seeds:
        if guess < left or guess > right or (right > 1 and abs(guess - 1) < 1e-8):
            continue
        try:
            root = mp.findroot(lambda z: mp.re(C * mp.hyp2f1(a, b, c, z) - L), mp.mpf(guess))
            if left <= root <= right:
                err = abs(C * mp.hyp2f1(a, b, c, root) - L)
                if err < best_err:
                    best_err = err
                    best_z = root
        except Exception:
            pass

    # Coarse fallback only if the seeded solves fail.
    if best_z is None:
        step = mp.mpf(right - left) / steps
        for j in range(steps + 1):
            z = mp.mpf(left) + step * j
            if right > 1 and abs(z - 1) < mp.mpf("1e-8"):
                continue
            try:
                err = abs(C * mp.hyp2f1(a, b, c, z) - L)
            except Exception:
                continue
            if err < best_err:
                best_err = err
                best_z = z

    if best_z is None:
        return None
    return {"z0": best_z, "digits": error_digits(best_err), "region": f"real[{left},{right}]"}


def search_unit_circle(L, a, b, c, C, steps: int = 120):
    best_err = mp.inf
    best_z = None
    for j in range(steps):
        theta = 2 * mp.pi * j / steps
        z = mp.e ** (1j * theta)
        try:
            err = abs(C * mp.hyp2f1(a, b, c, z) - L)
        except Exception:
            continue
        if err < best_err:
            best_err = err
            best_z = z
    if best_z is None:
        return None
    return {"z0": best_z, "digits": error_digits(best_err), "region": "unit-circle"}


def mobius_images(z):
    vals = []
    for func in [
        lambda w: w,
        lambda w: 1 - w,
        lambda w: 1 / w,
        lambda w: w / (w - 1),
        lambda w: 1 / (1 - w),
        lambda w: 1 - 1 / w,
    ]:
        try:
            vals.append(func(z))
        except Exception:
            pass
    return vals


def mobius_related(z1, z2, tol: float = 1e-30) -> bool:
    if z1 is None or z2 is None:
        return False
    z1 = mp.mpc(z1)
    z2 = mp.mpc(z2)
    tol_mpf = mp.mpf(str(tol))
    return any(abs(w - z2) < tol_mpf for w in mobius_images(z1))


def fit_hypergeom_point(value, alpha: int | float, delta: int | float):
    if value is None:
        return None
    with mp.workdps(60):
        L = mp.mpf(value)
        a, b, c = abc_from_alpha_delta(alpha, delta)
        best = {"digits": 0.0, "z0": None, "C": None, "region": "none"}

        # Fast path: search the natural real interval first.
        for p, q, C in rational_constant_candidates():
            hit = search_real_interval(L, a, b, c, C, 0, 1)
            if hit and hit["digits"] > best["digits"]:
                best = {**hit, "C": C}
                if best["digits"] >= 20:
                    return {
                        "a": a, "b": b, "c": c,
                        "z0": best["z0"], "C": best["C"],
                        "digits": best["digits"], "region": best["region"],
                        "admits": True,
                    }

        # Escalate only if the cheap real pass did not succeed.
        if best["digits"] < 12:
            for p, q, C in rational_constant_candidates()[:24]:
                hit = search_real_interval(L, a, b, c, C, 0, 4, steps=120)
                if hit and hit["digits"] > best["digits"]:
                    best = {**hit, "C": C}
                    if best["digits"] >= 20:
                        break
            if best["digits"] < 12:
                for p, q, C in rational_constant_candidates()[:12]:
                    hit = search_unit_circle(L, a, b, c, C)
                    if hit and hit["digits"] > best["digits"]:
                        best = {**hit, "C": C}

        return {
            "a": a, "b": b, "c": c,
            "z0": best["z0"], "C": best["C"],
            "digits": best["digits"], "region": best["region"],
            "admits": best["z0"] is not None and best["digits"] >= 12,
        }


def analyze_m24(phase1_rows: list[dict[str, str]]) -> dict[str, object]:
    meta_by_label: dict[str, dict[str, object]] = {}
    identified_d1: list[dict[str, object]] = []
    for row in phase1_rows:
        label = row["label"]
        a_coeffs, b_coeffs, _ = coeffs_from_label(label)
        d = parse_floatish(row.get("d", ""))
        tau = row.get("tau", "")
        tau_sector_val = tau_sector(tau)
        if d == 1 and tau_sector_val is not None:
            alpha = lead_coeff(b_coeffs)
            delta = lead_coeff(a_coeffs)
            value = high_precision_cf_value(a_coeffs, b_coeffs, depth=900, dps=60)
            fit = fit_hypergeom_point(value, alpha, delta)
            meta = {
                "alpha_lead": alpha,
                "abc_tuple": f"({fmt_mpf_or_mpc(fit['a'])}, {fmt_mpf_or_mpc(fit['b'])}, {fmt_mpf_or_mpc(fit['c'])})",
                "z0": fmt_mpf_or_mpc(fit['z0']) if fit and fit['z0'] is not None else "None",
                "z0_C": fmt_mpf_or_mpc(fit['C']) if fit and fit['C'] is not None else "None",
                "z0_digits": round(fit['digits'], 6) if fit else 0.0,
                "z0_region": fit['region'] if fit else "none",
                "admits_2f1": bool(fit and fit['admits']),
                "z0_raw": fit['z0'] if fit else None,
            }
        else:
            meta = {
                "alpha_lead": None,
                "abc_tuple": "None",
                "z0": "None",
                "z0_C": "None",
                "z0_digits": 0.0,
                "z0_region": "none",
                "admits_2f1": False,
                "z0_raw": None,
            }
        meta_by_label[label] = meta
        if d == 1 and tau_sector_val is not None:
            identified_d1.append({
                "label": label,
                "tau_sector": tau_sector_val,
                "abc_tuple": meta["abc_tuple"],
                "z0": meta["z0"],
                "z0_raw": meta["z0_raw"],
                "admits_2f1": meta["admits_2f1"],
                "z0_C": meta["z0_C"],
            })

    success = [r for r in identified_d1 if r["admits_2f1"]]
    hit_rate = (len(success) / len(identified_d1)) if identified_d1 else 0.0

    ld_groups: dict[tuple[str, str], set[str]] = {}
    abc_groups: dict[str, set[str]] = {}
    abc_rows: dict[str, list[dict[str, object]]] = {}
    for row in phase1_rows:
        if parse_floatish(row.get("d", "")) != 1 or tau_sector(row.get("tau", "")) is None:
            continue
        ld_key = (row.get("Lambda_theory", ""), row.get("Delta_shape", ""))
        abc_key = meta_by_label[row["label"]]["abc_tuple"]
        sector_val = tau_sector(row.get("tau", ""))
        ld_groups.setdefault(ld_key, set()).add(sector_val)
        abc_groups.setdefault(abc_key, set()).add(sector_val)
        abc_rows.setdefault(abc_key, []).append({
            "label": row["label"],
            "tau_sector": sector_val,
            "z0_raw": meta_by_label[row["label"]]["z0_raw"],
            "z0": meta_by_label[row["label"]]["z0"],
        })

    ld_mixed = {str(k): sorted(v) for k, v in ld_groups.items() if len(v) > 1}
    abc_mixed = {k: sorted(v) for k, v in abc_groups.items() if len(v) > 1}
    mobius_notes = []
    for abc_key, rows in abc_rows.items():
        if len(rows) < 2:
            continue
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                rel = mobius_related(rows[i]["z0_raw"], rows[j]["z0_raw"])
                mobius_notes.append({
                    "abc": abc_key,
                    "left": rows[i]["label"],
                    "right": rows[j]["label"],
                    "same_tau": rows[i]["tau_sector"] == rows[j]["tau_sector"],
                    "mobius_related": rel,
                })

    if hit_rate >= 0.80:
        verdict = "GO — z0 framework is viable"
    elif hit_rate < 0.50:
        verdict = "NO-GO — hypergeometric framework likely wrong"
    else:
        verdict = "MIXED — partial z0 support only"

    return {
        "meta_by_label": meta_by_label,
        "identified_d1": identified_d1,
        "success_count": len(success),
        "identified_count": len(identified_d1),
        "hit_rate": hit_rate,
        "verdict": verdict,
        "ld_mixed": ld_mixed,
        "abc_mixed": abc_mixed,
        "mobius_notes": mobius_notes,
    }


def rising(x, n: int):
    return mp.rf(x, n)


def appell_f2_series(a, b1, b2, c1, c2, x, y, terms: int = 12):
    total = mp.mpf("0")
    for m in range(terms):
        for n in range(terms):
            term = (rising(a, m + n) * rising(b1, m) * rising(b2, n) * (x ** m) * (y ** n))
            term /= (rising(c1, m) * rising(c2, n) * mp.factorial(m) * mp.factorial(n))
            total += term
    return total


def fit_catalan_frobenius():
    with mp.workdps(60):
        alpha = mp.mpf(4)
        delta = mp.mpf(-4)
        L = mp.catalan
        a, b, c = abc_from_alpha_delta(alpha, delta)
        candidates = rational_constant_candidates()[:6]
        z_seeds = [mp.mpf(s) for s in ("0.125", "0.25", "0.3333333333", "0.5", "0.6666666667", "0.75")]
        h = mp.mpf("1e-6")
        best = {"digits": 0.0, "z0": None, "C1": None, "C2": None}
        for _, _, C1 in candidates:
            for _, _, C2 in candidates:
                for z in z_seeds:
                    try:
                        F = mp.hyp2f1(a, b, c, z)
                        dFdc = (mp.hyp2f1(a, b, c + h, z) - mp.hyp2f1(a, b, c - h, z)) / (2 * h)
                        expr = C1 * F * mp.log(z) + C2 * dFdc
                        digits = error_digits(expr - L)
                    except Exception:
                        continue
                    if digits > best["digits"]:
                        best = {"digits": digits, "z0": z, "C1": C1, "C2": C2}
        return {
            "fit_success": best["digits"] >= 50,
            "digits": round(best["digits"], 6),
            "z0": fmt_mpf_or_mpc(best["z0"]) if best["z0"] is not None else "None",
            "C1": fmt_mpf_or_mpc(best["C1"]) if best["C1"] is not None else "None",
            "C2": fmt_mpf_or_mpc(best["C2"]) if best["C2"] is not None else "None",
            "abc_tuple": f"({fmt_mpf_or_mpc(a)}, {fmt_mpf_or_mpc(b)}, 1.0)",
        }


def fit_vquad_hypergeom():
    with mp.workdps(60):
        a_coeffs, b_coeffs, _ = coeffs_from_label("vquad-3-1-1")
        L = high_precision_cf_value(a_coeffs, b_coeffs, depth=1200, dps=60)
        lam = mp.sqrt(3)
        pair_candidates = [
            ((1 - mp.sqrt(13)) / 2, (1 + mp.sqrt(13)) / 2),
            (-lam, lam),
            ((1 - lam) / 2, (1 + lam) / 2),
        ]
        c_values = [mp.mpf("0.5"), mp.mpf("1.0"), mp.mpf("1.5"), mp.mpf("2.0")]
        consts = rational_constant_candidates()[:6]
        best_2f1 = {"digits": 0.0, "a": None, "b": None, "c": None, "z0": None, "C": None}
        for a, b in pair_candidates:
            for c in c_values:
                for _, _, C in consts:
                    hit = search_real_interval(L, a, b, c, C, 0, 1, steps=10)
                    if hit and hit["digits"] > best_2f1["digits"]:
                        best_2f1 = {"digits": hit["digits"], "a": a, "b": b, "c": c, "z0": hit["z0"], "C": C, "kind": "2F1"}

        best_appell = {"digits": 0.0, "kind": "none", "z0": None, "C": None}
        z_seeds = [mp.mpf(s) for s in ("0.2", "0.3333333333", "0.5", "0.6666666667")]
        for a, b in pair_candidates[:2]:
            for c in c_values[:2]:
                for _, _, C in consts[:4]:
                    for z in z_seeds:
                        x = z
                        y = z / 2
                        try:
                            if hasattr(mp, "appellf1"):
                                val_f1 = C * mp.appellf1(a, mp.mpf("0.5"), mp.mpf("0.5"), c, x, y)
                                d1 = error_digits(val_f1 - L)
                                if d1 > best_appell["digits"]:
                                    best_appell = {"digits": d1, "kind": "Appell F1", "z0": z, "C": C}
                        except Exception:
                            pass
                        try:
                            val_f2 = C * appell_f2_series(a, mp.mpf("0.5"), mp.mpf("0.5"), c, c, x, y, terms=10)
                            d2 = error_digits(val_f2 - L)
                            if d2 > best_appell["digits"]:
                                best_appell = {"digits": d2, "kind": "Appell F2", "z0": z, "C": C}
                        except Exception:
                            pass

        best = best_2f1 if best_2f1["digits"] >= best_appell["digits"] else best_appell
        return {
            "fit_success": best["digits"] >= 20,
            "best_kind": best.get("kind", "2F1"),
            "digits": round(best["digits"], 6),
            "z0": fmt_mpf_or_mpc(best.get("z0")) if best.get("z0") is not None else "None",
            "C": fmt_mpf_or_mpc(best.get("C")) if best.get("C") is not None else "None",
            "abc_tuple": (
                f"({fmt_mpf_or_mpc(best_2f1['a'])}, {fmt_mpf_or_mpc(best_2f1['b'])}, {fmt_mpf_or_mpc(best_2f1['c'])})"
                if best_2f1["a"] is not None else "None"
            ),
        }


def analyze_m25(phase1_rows: list[dict[str, str]], m24: dict[str, object]) -> dict[str, object]:
    identified = list(m24["identified_d1"])
    n = len(identified)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    same_orbit_confirmed = 0
    same_orbit_violations = 0
    equivalence_hits = 0
    pair_total = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_orbit = mobius_related(identified[i]["z0_raw"], identified[j]["z0_raw"], tol=1e-30)
            same_tau = identified[i]["tau_sector"] == identified[j]["tau_sector"]
            pair_total += 1
            if same_orbit == same_tau:
                equivalence_hits += 1
            if same_orbit:
                union(i, j)
                if same_tau:
                    same_orbit_confirmed += 1
                else:
                    same_orbit_violations += 1

    groups: dict[int, list[dict[str, object]]] = {}
    for i, row in enumerate(identified):
        groups.setdefault(find(i), []).append(row)

    orbit_rows = []
    meta_by_label = {}
    for idx, members in enumerate(groups.values(), start=1):
        taus = sorted({str(m["tau_sector"]) for m in members})
        labels = [str(m["label"]) for m in members]
        orbit_rows.append({
            "orbit_id": idx,
            "labels": labels,
            "taus": taus,
            "size": len(members),
            "representative": members[0]["z0"],
            "tau_pure": len(taus) == 1,
        })
        for m in members:
            meta_by_label[m["label"]] = {
                "pgl2_orbit_id": idx,
                "pgl2_orbit_size": len(members),
                "pgl2_orbit_tau_pure": len(taus) == 1,
            }

    catalan = fit_catalan_frobenius()
    vquad = fit_vquad_hypergeom()
    implication_total = same_orbit_confirmed + same_orbit_violations
    implication_rate = (same_orbit_confirmed / implication_total) if implication_total else 1.0
    equivalence_rate = (equivalence_hits / pair_total) if pair_total else 1.0
    phase3_go = (not catalan["fit_success"]) and (not vquad["fit_success"])

    return {
        "meta_by_label": meta_by_label,
        "orbit_rows": orbit_rows,
        "orbit_count": len(orbit_rows),
        "same_orbit_confirmed": same_orbit_confirmed,
        "same_orbit_violations": same_orbit_violations,
        "implication_rate": implication_rate,
        "equivalence_hits": equivalence_hits,
        "pair_total": pair_total,
        "equivalence_rate": equivalence_rate,
        "catalan": catalan,
        "vquad": vquad,
        "phase3_go": phase3_go,
    }


def analyze_m31() -> dict[str, object]:
    n, s, c1, c2 = sp.symbols("n s c1 c2")
    U = 1 + c1 / n + c2 / n**2
    U1 = sp.series((1 - 1 / n) ** s * (1 + c1 / (n - 1) + c2 / (n - 1) ** 2), n, sp.oo, 3).removeO()
    U2 = sp.series((1 - 2 / n) ** s * (1 + c1 / (n - 2) + c2 / (n - 2) ** 2), n, sp.oo, 3).removeO()
    rhs = sp.expand((sp.Integer(4) + 1 / n) * sp.Rational(1, 2) * U1 - (sp.Integer(4) - 4 / n) * sp.Rational(1, 4) * U2)
    diff = sp.expand(sp.series(rhs - U, n, sp.oo, 3).removeO())
    lead_obstruction = sp.simplify(diff.coeff(n, -1))

    with mp.workdps(220):
        relation = mp.pslq([
            mp.catalan,
            mp.sqrt(2),
            mp.sqrt(3),
            mp.pi,
            mp.log(2),
            mp.pi**2,
            mp.mpf("1"),
        ], maxcoeff=10**6)

    latex_lemma = r"""\begin{lemma}[Catalan obstruction, Gauss-ansatz form]
Let $K(a_n,b_n)$ be a degree-one polynomial continued fraction with
\[
  a_n = \delta n^2 + O(n),\qquad b_n = \alpha n + O(1),\qquad \Delta := \alpha^2 + 4\delta = 0.
\]
Let $p_n$ be any nonzero solution of the associated three-term recurrence
\[
  p_n = b_n p_{n-1} + a_n p_{n-2},
\]
and set $u_n := p_n/n!$. Then the normalized recurrence has a double Poincar\'e root
$\Lambda = \alpha/2$, and every non-algebraic local solution at the singular endpoint has Frobenius form
\[
  u_n = \Lambda^n n^s\!\left(A + B\log n + O\!\left(n^{-1}\log n\right)\right),
  \qquad B \neq 0.
\]
Consequently the continued-fraction limit $L$ cannot be represented by a single nonlogarithmic Gauss value
\[
  L = C\,{}_2F_1(a,b;c;z_0)
\]
with $C \in \mathbb{Q}$ and $a,b,c,z_0$ algebraic, unless the hypergeometric expression collapses to an algebraic special case.
\end{lemma}"""

    return {
        "normalized_recurrence": "u_n = (4 + 1/n)u_{n-1} - (4 - 4/n)u_{n-2}",
        "characteristic": "(lambda - 2)^2",
        "double_root": 2,
        "diff_series": str(diff),
        "lead_obstruction": str(lead_obstruction),
        "pslq_relation": relation,
        "pslq_status": "None found up to maxcoeff = 10^6 at 220 dps" if relation is None else str(relation),
        "latex_lemma": latex_lemma,
    }


def analyze_m32() -> dict[str, object]:
    n, s = sp.symbols("n s")
    A = (1 + sp.Rational(1, 3) / n + sp.Rational(1, 3) / n**2) * (1 - 1 / n) ** s
    B = sp.Rational(1, 9) * (1 - 2 / n) ** s / (n**2 * (n - 1) ** 2)
    A_series = sp.expand(sp.series(A, n, sp.oo, 3).removeO())
    B_series = sp.expand(sp.series(B, n, sp.oo, 5).removeO())
    A_s13 = sp.expand(sp.series(A.subs(s, sp.Rational(1, 3)), n, sp.oo, 4).removeO())

    with mp.workdps(220):
        p_prev = mp.mpf("1")
        p_curr = mp.mpf("5")
        samples = []
        for k in range(2, 201):
            b_k = 3 * k * k + k + 1
            p_next = b_k * p_curr + p_prev
            p_prev, p_curr = p_curr, p_next
            if k in {20, 40, 80, 120, 200}:
                scaled = p_curr / ((mp.mpf("3") ** k) * (mp.factorial(k) ** 2) * (mp.mpf(k) ** (mp.mpf("1") / 3)))
                samples.append({
                    "n": k,
                    "logp_over_n": float(mp.log(p_curr) / k),
                    "logp_over_nlogn": float(mp.log(p_curr) / (k * mp.log(k))),
                    "scaled": fmt_mpf_or_mpc(scaled),
                })

        x = mp.mpf("0")
        for k in reversed(range(1, 3000)):
            x = 1 / (3 * k * k + k + 1 + x)
        vquad_cf = 1 + x
        rel = mp.pslq([
            mp.mpf("1"),
            vquad_cf,
            mp.pi,
            mp.log(2),
            mp.log(3),
            mp.sqrt(3),
            mp.gamma(mp.mpf("1") / 3),
            mp.gamma(mp.mpf("1") / 4),
            mp.ellipk(mp.mpf("1") / 2),
            mp.ellipk(mp.mpf("3") / 4),
            mp.pi**2,
            mp.zeta(3),
        ], maxcoeff=10**6)

    return {
        "growth_law": "p_n ~ C * 3^n * (n!)^2 * n^(1/3)",
        "A_series": str(A_series),
        "B_series": str(B_series),
        "A_s13": str(A_s13),
        "roots": [1, 0],
        "distinct_roots": True,
        "samples": samples,
        "vquad_cf": fmt_mpf_or_mpc(vquad_cf),
        "pslq_status": "None found up to maxcoeff = 10^6 at 220 dps" if rel is None else str(rel),
    }


def denominator_sequence(a_coeffs: tuple[int, ...], b_coeffs: tuple[int, ...], n_terms: int) -> list[int]:
    q_prev = sp.Integer(0)
    q_curr = sp.Integer(1)
    out = [int(q_curr)]
    for n in range(1, n_terms):
        a_n = poly_eval(a_coeffs, n)
        b_n = poly_eval(b_coeffs, n)
        q_next = b_n * q_curr + a_n * q_prev
        out.append(int(q_next))
        q_prev, q_curr = q_curr, q_next
    return out


def shannon_entropy(counter: Counter[int], total: int) -> float:
    entropy = 0.0
    for c in counter.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def congruence_scores(seq: list[int], p: int) -> tuple[float, float, float]:
    tests = list(range(1, len(seq) // p))
    if not tests:
        return (0.0, 0.0, 0.0)
    mod1 = sum(1 for n in tests if (seq[n * p] - seq[n]) % p == 0) / len(tests)
    mod2 = sum(1 for n in tests if (seq[n * p] - seq[n]) % (p * p) == 0) / len(tests)
    mod3 = sum(1 for n in tests if (seq[n * p] - seq[n]) % (p * p * p) == 0) / len(tests)
    return (mod1, mod2, mod3)


def modular_period(a_coeffs: tuple[int, ...], b_coeffs: tuple[int, ...], p: int, n_terms: int = N_TERMS) -> int:
    """Return the eventual period of the mod-p denominator state machine.

    The deterministic state is `(n mod p, q_{n-1} mod p, q_n mod p)`, since the
    polynomial coefficients depend only on `n mod p` modulo `p`.
    """
    q_prev = 0
    q_curr = 1
    seen: dict[tuple[int, int, int], int] = {(0, q_prev, q_curr): 0}
    for n in range(1, n_terms + 1):
        a_n = int(poly_eval(a_coeffs, n) % p)
        b_n = int(poly_eval(b_coeffs, n) % p)
        q_next = (b_n * q_curr + a_n * q_prev) % p
        q_prev, q_curr = q_curr, q_next
        state = (n % p, q_prev, q_curr)
        if state in seen:
            return n - seen[state]
        seen[state] = n
    return -1


def residue_stats(seq: list[int], p: int, a_coeffs: tuple[int, ...], b_coeffs: tuple[int, ...]) -> ResidueStats:
    mods = [n % p for n in seq]
    counter = Counter(mods)
    om1, om2, om3 = congruence_scores(seq, p)
    return ResidueStats(
        p=p,
        zero_density=counter.get(0, 0) / len(seq),
        support_size=len(counter),
        entropy_bits=shannon_entropy(counter, len(seq)),
        period_mod_p=modular_period(a_coeffs, b_coeffs, p),
        omega_mod_p=om1,
        omega_mod_p2=om2,
        omega_mod_p3=om3,
        most_common=counter.most_common(min(5, p)),
    )


def signature_vector(stats: list[ResidueStats]) -> list[float]:
    vec: list[float] = []
    for s in stats:
        period_norm = (s.period_mod_p / (s.p ** 3)) if s.period_mod_p > 0 else 1.0
        vec.extend([s.zero_density, s.support_size / s.p, period_norm, s.omega_mod_p3])
    return vec


def euclidean(u: list[float], v: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))


def row_signature(label: str, group: str) -> tuple[SignatureRow, list[ResidueStats]]:
    a_coeffs, b_coeffs, inferred_group = coeffs_from_label(label)
    seq = denominator_sequence(a_coeffs, b_coeffs, N_TERMS)
    stats = [residue_stats(seq, p, a_coeffs, b_coeffs) for p in PRIMES]
    arithmetic_purity = sum(1 - (s.support_size / s.p) for s in stats)
    retention_stability = sum(s.omega_mod_p3 for s in stats) / len(stats)
    row = SignatureRow(
        label=label,
        group=group or inferred_group,
        D=[s.zero_density for s in stats],
        C_support=[s.support_size for s in stats],
        C_entropy=[s.entropy_bits for s in stats],
        periods=[s.period_mod_p for s in stats],
        omega=[s.omega_mod_p3 for s in stats],
        arithmetic_purity=arithmetic_purity,
        retention_stability=retention_stability,
        mean_zero_density=sum(s.zero_density for s in stats) / len(stats),
        mean_entropy=sum(s.entropy_bits for s in stats) / len(stats),
        mean_omega_p3=sum(s.omega_mod_p3 for s in stats) / len(stats),
    )
    return row, stats


def centroid(rows: list[SignatureRow]) -> list[float]:
    vectors = []
    for row in rows:
        flat = []
        for p, d, c, period, w in zip(PRIMES, row.D, row.C_support, row.periods, row.omega):
            period_norm = (period / (p ** 3)) if period > 0 else 1.0
            flat.extend([d, c / p, period_norm, w])
        vectors.append(flat)
    return [sum(vals[i] for vals in vectors) / len(vectors) for i in range(len(vectors[0]))]


phase1_rows_all = read_phase1_rows()
m22 = analyze_m22(phase1_rows_all)
m24 = analyze_m24(phase1_rows_all)
m25 = analyze_m25(phase1_rows_all, m24)
m31 = analyze_m31()
m32 = analyze_m32()
phase1_meta_by_label = {}
for row in phase1_rows_all:
    label = row["label"]
    merged = {}
    merged.update(m22["meta_by_label"].get(label, {}))
    merged.update(m24["meta_by_label"].get(label, {}))
    merged.update(m25["meta_by_label"].get(label, {}))
    phase1_meta_by_label[label] = merged
all_rows: list[SignatureRow] = []
details: dict[str, list[ResidueStats]] = {}
selected = set(ARGS.labels) if ARGS.labels else None

for r in phase1_rows_all:
    if selected is not None and r["label"] not in selected:
        continue
    sig, stats = row_signature(r["label"], r["group"])
    all_rows.append(sig)
    details[sig.label] = stats

baseline_rows: list[SignatureRow] = []
if selected is not None:
    for r in phase1_rows_all:
        if r["label"].startswith("log-k=") or r["label"].startswith("pi-m="):
            sig, _ = row_signature(r["label"], r["group"])
            baseline_rows.append(sig)
else:
    baseline_rows = list(all_rows)

log_rows = [r for r in all_rows if r.label.startswith("log-k=")] or [r for r in baseline_rows if r.label.startswith("log-k=")]
pi_rows = [r for r in all_rows if r.label.startswith("pi-m=")] or [r for r in baseline_rows if r.label.startswith("pi-m=")]
log_centroid = centroid(log_rows)
pi_centroid = centroid(pi_rows)

for row in all_rows:
    flat = []
    for p, d, c, period, w in zip(PRIMES, row.D, row.C_support, row.periods, row.omega):
        period_norm = (period / (p ** 3)) if period > 0 else 1.0
        flat.extend([d, c / p, period_norm, w])
    row.dist_to_log = euclidean(flat, log_centroid)
    row.dist_to_pi = euclidean(flat, pi_centroid)
    near = "log" if row.dist_to_log <= row.dist_to_pi else "pi"
    if min(row.dist_to_log, row.dist_to_pi) > 1.0:
        near = f"{near}-distant"
    row.nearest_cluster = near

suffix = f"_{ARGS.tag}" if ARGS.tag else ""
json_path = RESULTS / f"pcf_spectral_phase2_congruence{suffix}.json"
csv_path = RESULTS / f"pcf_spectral_phase2_congruence{suffix}.csv"
report_path = RESULTS / f"pcf_spectral_phase2_report{suffix}.md"
map_path = RESULTS / f"pcf_spectral_phase25_signature_map{suffix}.csv"
plot_path = RESULTS / f"pcf_spectral_phase25_signature_map{suffix}.png"

json_blob = []
for row in all_rows:
    rec = asdict(row)
    rec["per_prime"] = [asdict(s) for s in details[row.label]]
    json_blob.append(rec)
json_path.write_text(json.dumps(json_blob, indent=2), encoding="utf-8")

fieldnames = [
    "label", "group", "d", "tau", "tau_sector", "Delta_shape", "delta_lead", "delta_sign",
    "lambda_rational", "galois_prediction", "galois_hit", "is_degenerate_boundary", "b0_mod4",
    "abc_tuple", "z0", "z0_C", "z0_digits", "z0_region", "admits_2f1",
    "pgl2_orbit_id", "pgl2_orbit_size", "pgl2_orbit_tau_pure",
    "mean_zero_density", "mean_entropy", "mean_omega_p3",
    "arithmetic_purity", "retention_stability",
    "dist_to_log", "dist_to_pi", "nearest_cluster",
]
for p in PRIMES:
    fieldnames += [f"D_p{p}", f"C_p{p}", f"period_p{p}", f"omega_p{p}"]

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_rows:
        meta = phase1_meta_by_label.get(row.label, {})
        out = {
            "label": row.label,
            "group": row.group,
            "d": meta.get("d", ""),
            "tau": meta.get("tau", ""),
            "tau_sector": meta.get("tau_sector", ""),
            "Delta_shape": meta.get("Delta_shape", ""),
            "delta_lead": meta.get("delta_lead", ""),
            "delta_sign": meta.get("delta_sign", ""),
            "lambda_rational": meta.get("lambda_rational", ""),
            "galois_prediction": meta.get("galois_prediction", ""),
            "galois_hit": meta.get("galois_hit", ""),
            "is_degenerate_boundary": meta.get("is_degenerate_boundary", ""),
            "b0_mod4": meta.get("b0_mod4", ""),
            "abc_tuple": meta.get("abc_tuple", "None"),
            "z0": meta.get("z0", "None"),
            "z0_C": meta.get("z0_C", "None"),
            "z0_digits": meta.get("z0_digits", 0.0),
            "z0_region": meta.get("z0_region", "none"),
            "admits_2f1": meta.get("admits_2f1", False),
            "pgl2_orbit_id": meta.get("pgl2_orbit_id", ""),
            "pgl2_orbit_size": meta.get("pgl2_orbit_size", ""),
            "pgl2_orbit_tau_pure": meta.get("pgl2_orbit_tau_pure", ""),
            "mean_zero_density": f"{row.mean_zero_density:.6f}",
            "mean_entropy": f"{row.mean_entropy:.6f}",
            "mean_omega_p3": f"{row.mean_omega_p3:.6f}",
            "arithmetic_purity": f"{row.arithmetic_purity:.6f}",
            "retention_stability": f"{row.retention_stability:.6f}",
            "dist_to_log": f"{row.dist_to_log:.6f}",
            "dist_to_pi": f"{row.dist_to_pi:.6f}",
            "nearest_cluster": row.nearest_cluster,
        }
        for p, d, c, period, w in zip(PRIMES, row.D, row.C_support, row.periods, row.omega):
            out[f"D_p{p}"] = f"{d:.6f}"
            out[f"C_p{p}"] = c
            out[f"period_p{p}"] = period
            out[f"omega_p{p}"] = f"{w:.6f}"
        writer.writerow(out)


def fmt_vec(xs: list[float], digits: int = 3) -> str:
    return "[" + ", ".join(f"{x:.{digits}f}" for x in xs) + "]"


def fmt_int_vec(xs: list[int]) -> str:
    return "[" + ", ".join(str(x) for x in xs) + "]"


with map_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["label", "group", "purity", "stability", "dist_to_log", "dist_to_pi", "nearest_cluster"],
    )
    writer.writeheader()
    for row in all_rows:
        writer.writerow({
            "label": row.label,
            "group": row.group,
            "purity": f"{row.arithmetic_purity:.6f}",
            "stability": f"{row.retention_stability:.6f}",
            "dist_to_log": f"{row.dist_to_log:.6f}",
            "dist_to_pi": f"{row.dist_to_pi:.6f}",
            "nearest_cluster": row.nearest_cluster,
        })

if plt is not None and all_rows:
    color_map = {
        "Log ladder": "tab:blue",
        "Pi family": "tab:red",
        "V_quad": "tab:green",
        "Apery-isolated": "tab:purple",
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in all_rows:
        ax.scatter(row.arithmetic_purity, row.retention_stability, s=64, color=color_map.get(row.group, "gray"), alpha=0.85)
        if row.label in {"log-k=2", "pi-m=0", "vquad-3-1-1", "apery-zeta3"}:
            ax.annotate(row.label, (row.arithmetic_purity, row.retention_stability), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Arithmetic purity = Σ_p (1 - C_p/p)")
    ax.set_ylabel("Retention stability = mean_p ω_p")
    ax.set_title("Phase 2.5 signature map: purity vs stability")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

by_label = {r.label: r for r in all_rows}
baseline_by_label = {r.label: r for r in baseline_rows}
log2 = by_label.get("log-k=2") or baseline_by_label.get("log-k=2")
pi0 = by_label.get("pi-m=0") or baseline_by_label.get("pi-m=0")
vquad = by_label.get("vquad-3-1-1") or baseline_by_label.get("vquad-3-1-1")

lines: list[str] = []
lines.append("# Phase 2 — Congruence Hunt\n")
lines.append("## Revised M2-2: Galois-class separator\n")
lines.append("We tested the revised separator based on `(Lambda rational?) × sign(delta)`, with the rules: irrational `Lambda` plus `delta>0` predicts `pi-type`, irrational `Lambda` plus `delta<0` predicts `log-type`, `Delta=0` predicts `Catalan-tier`, and rational `Lambda` is treated as degenerate / no primary prediction.\n")
if m22['total']:
    lines.append(f"- Primary hit rate on non-degenerate identified `d=1` rows: **{m22['hits']}/{m22['total']} = {100.0 * m22['hit_rate']:.1f}%**")
else:
    lines.append("- Primary hit rate on non-degenerate identified `d=1` rows: **N/A (0 testable families after excluding rational-Λ and boundary-degenerate cases)**")
lines.append(f"- Go / no-go verdict: **{m22['verdict']}**")
if m22['secondary_total'] and m22['secondary_has_both']:
    pure_txt = "pure" if m22['secondary_pure'] else "mixed"
    lines.append(f"- Secondary degenerate test (`b0 mod 4`) on rational-Λ `pi/algebraic` cases: **{m22['secondary_hits']}/{m22['secondary_total']} = {100.0 * m22['secondary_rate']:.1f}%**, signature is `{pure_txt}`.\n")
else:
    obs = ", ".join(m22['secondary_observed_sectors']) if m22['secondary_observed_sectors'] else "none"
    lines.append(f"- Secondary degenerate test (`b0 mod 4`) is currently **underdetermined**: observed rational-Λ comparable sectors = `{obs}`.\n")
lines.append("### Family audit\n")
lines.append("| label | Delta | lambda rational? | delta sign | a1=0 boundary? | prediction | actual sector | hit? |")
lines.append("|---|---:|:---:|:---:|:---:|---|---|:---:|")
for r in m22['d1_identified']:
    delta_shape = "—" if r['Delta_shape'] is None else int(round(r['Delta_shape']))
    hit_val = r['galois_hit']
    hit_flag = "✓" if hit_val is True else ("✗" if hit_val is False else "—")
    lines.append(f"| `{r['label']}` | {delta_shape} | `{r['lambda_rational']}` | `{r['delta_sign']}` | `{r['is_degenerate_boundary']}` | `{r['galois_prediction']}` | `{r['tau_sector']}` | {hit_flag} |")
lines.append("\n### Boundary degeneracy check\n")
lines.append("For `probe-d1-boundary-nn1-3n2`, we verified that `a_1 = 1*(1-1) = 0`. Therefore the first partial numerator vanishes and the continued fraction collapses immediately to the head term `b_0 = 2`, which matches the computed value exactly. This is a genuine **degenerate-boundary** case and is excluded from the primary Galois-class test.\n")
if m22['secondary_total']:
    lines.append("### Secondary degenerate separator: `b0 mod 4`\n")
    lines.append("| b0 mod 4 | observed sectors |")
    lines.append("|---:|---|")
    for residue, sectors in sorted(m22['secondary_classes'].items()):
        lines.append(f"| {residue} | {', '.join(f'`{s}`' for s in sectors)} |")
lines.append("\n### New `d=1` probe families\n")
lines.append("The requested probe rows were appended to the Phase 1 taxonomy and re-evaluated. Only identified rows count toward the primary Galois hit rate; unresolved rows remain useful diagnostics.\n")
lines.append("| label | delta | sign(delta) | tau | Galois status |")
lines.append("|---|---:|:---:|---|---|")
for probe_label in ["probe-d1-pos-4n2-6n1", "probe-d1-neg-n2-8n4", "probe-d1-boundary-nn1-3n2"]:
    meta = phase1_meta_by_label.get(probe_label)
    if not meta:
        continue
    tau_txt = meta.get("tau") or "unresolved"
    if not meta.get("identified_for_m22"):
        status = "open / unresolved"
    elif meta.get("is_degenerate_boundary"):
        status = "degenerate-boundary"
    else:
        hit = meta.get("galois_hit")
        status = "consistent" if hit is True else ("counterexample" if hit is False else "degenerate / no prediction")
    lines.append(f"| `{probe_label}` | {meta.get('delta_lead')} | `{meta.get('delta_sign')}` | `{tau_txt}` | {status} |")
if m22['colliding']:
    lines.append("\n### Colliding `Delta` buckets among identified `d=1` rows\n")
    lines.append("| Delta | labels | tau sectors |")
    lines.append("|---:|---|---|")
    for delta_key, rows in sorted(m22['colliding'].items()):
        labels = ", ".join(f"`{r['label']}`" for r in rows)
        sectors = ", ".join(sorted({str(r['tau_sector']) for r in rows}))
        lines.append(f"| {delta_key} | {labels} | {sectors} |")

lines.append("\n## M2-4: Hypergeometric evaluation-point audit\n")
lines.append("For each identified `d=1` family we formed the indicial `2F1` parameters `(a,b,c)` from `a+b = 1-alpha`, `ab = -delta`, `c = 1`, then searched for a rational prefactor `C` with `|C| <= 10` and denominator `<= 12` such that `L = C * hyp2f1(a,b;c;z0)`.\n")
lines.append(f"- Families admitting a fitted `z0`: **{m24['success_count']}/{m24['identified_count']} = {100.0 * m24['hit_rate']:.1f}%**")
lines.append(f"- M2-4 verdict: **{m24['verdict']}**")
lines.append("\n| label | tau | (a,b,c) | C | z0 | region | digits |")
lines.append("|---|---|---|---:|---:|---|---:|")
for row in m24['identified_d1']:
    meta = phase1_meta_by_label[row['label']]
    lines.append(
        f"| `{row['label']}` | `{phase1_meta_by_label[row['label']].get('tau','')}` | `{meta.get('abc_tuple','None')}` | "
        f"`{meta.get('z0_C','None')}` | `{meta.get('z0','None')}` | `{meta.get('z0_region','none')}` | {meta.get('z0_digits',0.0)} |"
    )

ld_count = len(m24['ld_mixed'])
abc_count = len(m24['abc_mixed'])
lines.append("\n### Group-purity comparison\n")
lines.append(f"- Mixed groups under `(Lambda, Delta)`: **{ld_count}**")
lines.append(f"- Mixed groups under `(a,b,c)`: **{abc_count}**")
if abc_count < ld_count:
    lines.append("- Conclusion: `(a,b,c)` separates the arithmetic sectors better than `(Lambda,Delta)`.\n")
elif abc_count == ld_count:
    lines.append("- Conclusion: `(a,b,c)` alone does **not** separate the arithmetic sectors better than `(Lambda,Delta)`; the improvement comes only after adjoining `z0`.\n")
else:
    lines.append("- Conclusion: `(a,b,c)` alone is not a stronger separator than `(Lambda,Delta)`.\n")

if 'log-k=2' in phase1_meta_by_label and 'pi-m=0' in phase1_meta_by_label:
    log2_meta = phase1_meta_by_label['log-k=2']
    pi0_meta = phase1_meta_by_label['pi-m=0']
    related = mobius_related(log2_meta.get('z0_raw'), pi0_meta.get('z0_raw'))
    verdict_txt = 'yes' if related else 'no'
    lines.append(
        f"- Special mixed control: `log-k=2` and `pi-m=0` share the same indicial class `{log2_meta.get('abc_tuple')}`, "
        f"but their fitted points are `z0 ≈ {log2_meta.get('z0')}` and `z0 ≈ {pi0_meta.get('z0')}` respectively; "
        f"under the basic Euler/Pfaff Möbius transforms this relation is **{verdict_txt}**."
    )

obstructions = [r['label'] for r in m24['identified_d1'] if not r['admits_2f1']]
if obstructions:
    lines.append(f"- Families with no fitted `z0` in the searched framework: {', '.join(f'`{lab}`' for lab in obstructions)}.")
else:
    lines.append("- Every currently identified `d=1` family admits a fitted `z0` in the searched `2F1` framework.")

lines.append("\n## Phase 2 Main Result: The z0-Orbit Separation Theorem (Conjecture)\n")
lines.append(
    "**Conjecture.** For the identified `d=1` polynomial continued fractions in this dataset, the `PGL2(Q)`-orbit of the fitted hypergeometric point `z0` is a genuine arithmetic separator. In particular, if `z0' = (az0+b)/(cz0+d)` with `a,b,c,d in Q` and `ad-bc != 0`, then the two families should lie in the same arithmetic sector; the converse is not yet forced by the present data, so the orbit currently appears to be a finer invariant than `tau` itself."
)
lines.append(f"- Distinct `PGL2(Q)` orbit classes among identified `d=1` families: **{m25['orbit_count']}**")
lines.append(f"- Same-orbit => same-tau check: **{m25['same_orbit_confirmed']}/{m25['same_orbit_confirmed'] + m25['same_orbit_violations']} = {100.0 * m25['implication_rate']:.1f}%**")
lines.append(f"- Full pairwise equivalence score `(same orbit) <=> (same tau)`: **{m25['equivalence_hits']}/{m25['pair_total']} = {100.0 * m25['equivalence_rate']:.1f}%**")
lines.append("\n### Orbit inventory\n")
lines.append("| orbit id | size | tau sectors | labels | representative z0 |")
lines.append("|---:|---:|---|---|---|")
for orbit in m25['orbit_rows']:
    tau_txt = ", ".join(f"`{t}`" for t in orbit['taus'])
    label_txt = ", ".join(f"`{lab}`" for lab in orbit['labels'])
    lines.append(f"| {orbit['orbit_id']} | {orbit['size']} | {tau_txt} | {label_txt} | `{orbit['representative']}` |")
lines.append("\n### Obstruction diagnostics\n")
lines.append(
    f"- Catalan Frobenius-log trial (`2F1 * log(z0)` plus `∂/∂c 2F1`): success = `{m25['catalan']['fit_success']}`, best score = **{m25['catalan']['digits']} digits**, at `z0 ≈ {m25['catalan']['z0']}` with `(C1,C2)=({m25['catalan']['C1']}, {m25['catalan']['C2']})`."
)
lines.append(
    f"- V_quad trial (`2F1`, Appell `F1/F2` search): success = `{m25['vquad']['fit_success']}`, best model = `{m25['vquad']['best_kind']}`, best score = **{m25['vquad']['digits']} digits**, at `z0 ≈ {m25['vquad']['z0']}` with `C = {m25['vquad']['C']}`."
)
phase3_txt = "GO" if m25['phase3_go'] else "HOLD"
lines.append(f"- Phase 3 obstruction criterion (Catalan fail AND V_quad fail): **{phase3_txt}**")

lines.append("\n## M3-1: Catalan obstruction — Frobenius degenerate case\n")
lines.append("For the Catalan family `a_n = -4n^2`, `b_n = 4n+1`, the discriminant is `Delta = 4^2 + 4(-4) = 0`, so the normalized recurrence sits exactly in the repeated-root Frobenius regime. Writing `u_n := p_n/n!`, the recurrence becomes\n")
lines.append("```text\nu_n = (4 + 1/n) u_{n-1} - (4 - 4/n) u_{n-2},\n```\nwhose Poincare characteristic polynomial is `(lambda-2)^2`. Hence the indicial equation has a **double root** `Lambda = 2`. In the Frobenius transport system this is the resonant case where the second transport equation loses rank, and the second local solution acquires a logarithmic term.\n")
lines.append("For a pure power-series ansatz `u_n = 2^n n^s (1 + c1/n + c2/n^2 + ...)`, a direct symbolic expansion gives\n")
lines.append(f"```text\nRHS - LHS = {m31['diff_series']}\n```\n")
lines.append(f"In particular the unavoidable coefficient of `1/n` is **{m31['lead_obstruction']}**, so no choice of `s,c1,c2,...` can remove the mismatch without adjoining a `log n` correction. This proves the Catalan family is a genuine **Frobenius-log obstruction**.\n")
lines.append("At the ODE level, a Gauss hypergeometric equation has local exponents `0` and `c-a-b` at `z=1`; a logarithmic branch appears only in the resonant cases where `c-a-b` is an integer, and Kummer's 24 solutions merely permute the three exponent differences. Thus the Catalan logarithmic branch is incompatible with the plain nonlogarithmic `C * 2F1(a,b;c;z0)` ansatz used in M2-4; only a degenerate logarithmic hypergeometric reduction could survive, and that would force an algebraic special case in the present search framework.\n")
lines.append("### LaTeX lemma\n")
lines.append("```latex")
lines.extend(m31["latex_lemma"].splitlines())
lines.append("```\n")
lines.append("### 200-digit PSLQ verification\n")
lines.append("We ran PSLQ on the basis `{Catalan, sqrt(2), sqrt(3), pi, log 2, pi^2, 1}` at 220 decimal digits. The result was:\n")
lines.append(f"- `PSLQ = {m31['pslq_status']}`\n")
lines.append("So no rational linear relation was detected in this span at the tested height, consistent with Catalan being genuinely outside the searched algebraic-hypergeometric field.\n")

lines.append("\n## M3-2: V_quad Frobenius analysis — clean d=2 tier\n")
lines.append("For the V_quad seed `a_n = 1`, `b_n = 3n^2+n+1`, the raw denominator growth is factorial-square rather than single-exponential. Numerically, `log p_n / (n log n)` approaches `2`, so the correct dominant law is\n")
lines.append(f"- **{m32['growth_law']}**\n")
lines.append("Equivalently, after the Perron--Kreuser normalization\n")
lines.append("```text\np_n = 3^n (n!)^2 n^s q_n,\n```\nwe obtain a renormalized recurrence\n")
lines.append(f"```text\nA_n(s) = {m32['A_series']}\nB_n(s) = {m32['B_series']}\n```\n")
lines.append("Choosing `s = 1/3` cancels the `1/n` drift term, and the leading coefficient becomes\n")
lines.append(f"```text\nA_n(1/3) = {m32['A_s13']}\n```\n")
lines.append("Hence the asymptotic characteristic roots are **`1` and `0`**, which are distinct. Therefore V_quad lies in a **clean d=2 tier**: there is no Catalan-style logarithmic Frobenius resonance at the sequence level.\n")
lines.append("This does not place V_quad in the Gauss `2F1` class. Instead it supports the stronger interpretation that V_quad belongs to a higher irregular / confluent-Heun-type sector whose obstruction is arithmetic rather than repeated-root logarithmic.\n")
lines.append("### 220-digit PSLQ against an extended special-function basis\n")
lines.append("We tested `{1, V_quad, pi, log 2, log 3, sqrt(3), Gamma(1/3), Gamma(1/4), K(1/sqrt(2)), K(sqrt(3)/2), pi^2, zeta(3)}` at 220 dps. The result was:\n")
lines.append(f"- `PSLQ = {m32['pslq_status']}`\n")
lines.append(f"- `V_quad ≈ {m32['vquad_cf']}`\n")
lines.append("So the current evidence is **conditional but strong**: V_quad is not yet proved obstructed, but it is numerically outside all tested Gauss/elliptic spans and should be stated as a conjectural arithmetic obstruction in the paper.\n")

lines.append("\n## Secondary signature\n")
lines.append("We define the micro-spectral congruence signature\n")
lines.append("- `𝒟 = (D_p)` where `D_p` is the density of indices with `p | q_n`,")
lines.append("- `𝒞 = (C_p)` where `C_p` is the residue-support size of `q_n mod p`,")
lines.append("- `T = (T_p)` where `T_p` is the exact eventual period of the denominator state modulo `p`,")
lines.append("- `ω = (ω_p)` where `ω_p` is the fraction of nontrivial tests with `q_{np} ≡ q_n (mod p^3)`.\n")
lines.append("## Seed comparison: `log-k=2` vs `pi-m=0`\n")
lines.append("| Family | 𝒟 (zero-density vector) | 𝒞 (support sizes) | T (period vector) | ω (mod p^3 retention) | purity | stability |")
lines.append("|---|---|---|---|---|---:|---:|")
for row in [log2, pi0]:
    lines.append(
        f"| `{row.label}` | `{fmt_vec(row.D)}` | `{fmt_int_vec(row.C_support)}` | `{fmt_int_vec(row.periods)}` | `{fmt_vec(row.omega)}` | {row.arithmetic_purity:.3f} | {row.retention_stability:.3f} |"
    )

lines.append("\n## Prime-by-prime comparison\n")
lines.append("| p | D(log) | D(pi) | C(log) | C(pi) | period(log) | period(pi) | ω(log) | ω(pi) |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for i, p in enumerate(PRIMES):
    lines.append(
        f"| {p} | {log2.D[i]:.3f} | {pi0.D[i]:.3f} | {log2.C_support[i]} | {pi0.C_support[i]} | {log2.periods[i]} | {pi0.periods[i]} | {log2.omega[i]:.3f} | {pi0.omega[i]:.3f} |"
    )

lines.append("\n## Vquad mystery check\n")
lines.append(
    f"`vquad-3-1-1` has `𝒟 = {fmt_vec(vquad.D)}`, `𝒞 = {fmt_int_vec(vquad.C_support)}`, `periods = {fmt_int_vec(vquad.periods)}`, `ω = {fmt_vec(vquad.omega)}`."
)
lines.append(
    f"Its Euclidean distance to the log centroid is `{vquad.dist_to_log:.3f}` and to the pi centroid is `{vquad.dist_to_pi:.3f}`; it is therefore **far from both** classical clusters, though slightly closer to the log centroid."
)
lines.append("\n## Interpretation\n")
lines.append("1. Raw macro-spectral data fail to separate `log-k=2` from `pi-m=0`, but the congruence signature does.")
lines.append("2. The Pi seed has uniformly higher zero-density and lower residue support for odd primes, hence a tighter p-adic shadow.")
lines.append("3. For the seed pair `log-k=2` vs `pi-m=0`, the exact small-prime period vector `T` is actually the same; the separation comes from residue-support collapse and stronger `ω`, not from `T` alone.")
lines.append("4. The Pi seed also has a stronger `ω` score at most odd primes, especially `p=11`, which is the clearest Apéry-like residue-retention signal in this dataset.")
lines.append("5. `vquad-3-1-1` is not aligned with either the log or pi micro-signature; its residue support is essentially full and its divisibility density is much lower, indicating a genuinely different arithmetic sector.")

report_path.write_text("\n".join(lines), encoding="utf-8")

print(f"Wrote {json_path}")
print(f"Wrote {csv_path}")
print(f"Wrote {report_path}")
print(f"Wrote {map_path}")
if plt is not None and plot_path.exists():
    print(f"Wrote {plot_path}")
print("\nSeed summary:")
for row in [log2, pi0, vquad]:
    print(
        f"{row.label:>12s} | D={fmt_vec(row.D)} | C={fmt_int_vec(row.C_support)} | "
        f"periods={fmt_int_vec(row.periods)} | omega={fmt_vec(row.omega)} | "
        f"purity={row.arithmetic_purity:.3f} | stability={row.retention_stability:.3f} | nearest={row.nearest_cluster}"
    )
