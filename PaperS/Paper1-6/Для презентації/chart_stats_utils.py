import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional dependency
    scipy_stats = None


def wilson_interval(successes, total, z=1.96):
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, (center - margin) * 100.0), min(100.0, (center + margin) * 100.0)


def add_stats_box(ax, lines, loc="lower right", fontsize=10):
    if not lines:
        return
    anchors = {
        "lower right": (0.98, 0.04, "right", "bottom"),
        "lower left": (0.02, 0.04, "left", "bottom"),
        "upper right": (0.98, 0.96, "right", "top"),
        "upper left": (0.02, 0.96, "left", "top"),
    }
    x, y, ha, va = anchors.get(loc, anchors["lower right"])
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#b0b0b0",
            "alpha": 0.92,
            "linewidth": 0.8,
        },
    )


def _normal_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _chi2_sf_approx(chi2, df):
    if df <= 0:
        return None
    if chi2 <= 0:
        return 1.0
    # Wilson-Hilferty approximation to the upper chi-square tail.
    z = ((chi2 / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return _normal_sf(z)


def format_p(p):
    if p is None or not math.isfinite(float(p)):
        return "n/a"
    p = float(p)
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _clean_float(value):
    try:
        value = float(value)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _rank_average(values):
    pairs = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][1]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x_values, y_values):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.size < 3 or y.size < 3:
        return None
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman_from_pairs(raw_pairs):
    cleaned = []
    for x, y in raw_pairs or []:
        x_clean = _clean_float(x)
        y_clean = _clean_float(y)
        if x_clean is not None and y_clean is not None:
            cleaned.append((x_clean, y_clean))
    n = len(cleaned)
    if n < 3:
        return {"n": n, "rho": None, "p": None}

    x_values = [p[0] for p in cleaned]
    y_values = [p[1] for p in cleaned]
    if scipy_stats is not None:
        result = scipy_stats.spearmanr(x_values, y_values)
        return {"n": n, "rho": float(result.statistic), "p": float(result.pvalue)}

    rho = _pearson(_rank_average(x_values), _rank_average(y_values))
    if rho is None or abs(rho) >= 1.0:
        p = 0.0 if rho is not None else None
    else:
        t_value = abs(rho) * math.sqrt((n - 2) / max(1e-12, 1.0 - rho * rho))
        # For this dataset n is large; normal tail is a conservative fallback.
        p = 2.0 * _normal_sf(t_value)
    return {"n": n, "rho": rho, "p": p}


def point_biserial_from_pairs(raw_pairs):
    cleaned = []
    for x, y in raw_pairs or []:
        x_clean = _clean_float(x)
        if x_clean is not None:
            cleaned.append((x_clean, 1.0 if bool(y) else 0.0))
    n = len(cleaned)
    if n < 3:
        return {"n": n, "r": None, "p": None}

    x_values = np.asarray([p[0] for p in cleaned], dtype=float)
    y_values = np.asarray([p[1] for p in cleaned], dtype=float)
    if scipy_stats is not None:
        result = scipy_stats.pointbiserialr(y_values, x_values)
        return {"n": n, "r": float(result.statistic), "p": float(result.pvalue)}

    r = _pearson(x_values, y_values)
    if r is None or abs(r) >= 1.0:
        p = 0.0 if r is not None else None
    else:
        t_value = abs(r) * math.sqrt((n - 2) / max(1e-12, 1.0 - r * r))
        p = 2.0 * _normal_sf(t_value)
    return {"n": n, "r": r, "p": p}


def two_proportion_z(success_1, total_1, success_2, total_2):
    if total_1 <= 0 or total_2 <= 0:
        return {"z": None, "p": None, "rate_diff_pp": None}
    p1 = success_1 / total_1
    p2 = success_2 / total_2
    pooled = (success_1 + success_2) / (total_1 + total_2)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / total_1 + 1.0 / total_2))
    if se == 0.0:
        return {"z": 0.0, "p": 1.0, "rate_diff_pp": (p1 - p2) * 100.0}
    z_value = (p1 - p2) / se
    p_value = 2.0 * (_normal_sf(abs(z_value)) if scipy_stats is None else scipy_stats.norm.sf(abs(z_value)))
    return {"z": float(z_value), "p": float(p_value), "rate_diff_pp": (p1 - p2) * 100.0}


def _weighted_polyfit(x_values, y_values, weights, degree):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.size <= degree or np.sum(w) <= 0:
        return None, None
    coeffs = np.polyfit(x, y, degree, w=np.sqrt(w))
    fitted = np.polyval(coeffs, x)
    y_mean = np.average(y, weights=w)
    ss_res = np.sum(w * (y - fitted) ** 2)
    ss_tot = np.sum(w * (y - y_mean) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return coeffs, None if r2 is None else float(r2)


def build_bin_rows(labels, counts, successes, dataset="binned"):
    rows = []
    for label, count, success_count in zip(labels, counts, successes):
        count = int(count)
        success_count = int(success_count)
        rate = (success_count / count * 100.0) if count else 0.0
        ci_low, ci_high = wilson_interval(success_count, count)
        rows.append(
            {
                "dataset": dataset,
                "range": label,
                "n": count,
                "successes": success_count,
                "failures": count - success_count,
                "success_rate_pct": round(rate, 4),
                "ci95_low_pct": round(ci_low, 4),
                "ci95_high_pct": round(ci_high, 4),
            }
        )
    return rows


def compute_binned_numeric_stats(data, raw_pairs, variable_name, trend_degree=1):
    labels = data.get("ranges", [])
    counts = data.get("orders_count", [])
    successes = data.get("successes", [])
    rows = build_bin_rows(labels, counts, successes)

    total_n = int(sum(counts))
    total_success = int(sum(successes))
    overall_rate = total_success / total_n * 100.0 if total_n else 0.0
    rates = [row["success_rate_pct"] for row in rows]
    rate_min = min(rates) if rates else 0.0
    rate_max = max(rates) if rates else 0.0

    spearman = spearman_from_pairs(raw_pairs)
    point_biserial = point_biserial_from_pairs(raw_pairs)
    ci_halfwidths = [
        (row["ci95_high_pct"] - row["ci95_low_pct"]) / 2.0
        for row in rows
        if row["n"] > 0
    ]
    metrics = {
        "variable": variable_name,
        "n": total_n,
        "successes": total_success,
        "overall_success_rate_pct": round(overall_rate, 4),
        "bins": len(rows),
        "rate_min_pct": round(rate_min, 4),
        "rate_max_pct": round(rate_max, 4),
        "rate_range_pp": round(rate_max - rate_min, 4),
        "spearman_rho": None if spearman["rho"] is None else round(spearman["rho"], 6),
        "spearman_p": spearman["p"],
        "point_biserial_r": None if point_biserial["r"] is None else round(point_biserial["r"], 6),
        "point_biserial_p": point_biserial["p"],
        "max_ci95_halfwidth_pp": round(max(ci_halfwidths), 4) if ci_halfwidths else None,
    }

    valid_idx = [i for i, count in enumerate(counts) if count > 0]
    if len(valid_idx) > 1:
        x_values = valid_idx
        y_values = [rates[i] for i in valid_idx]
        weights = [counts[i] for i in valid_idx]
        coeffs, r2 = _weighted_polyfit(x_values, y_values, weights, 1)
        if coeffs is not None:
            metrics["weighted_linear_slope_pp_per_bin"] = round(float(coeffs[0]), 6)
            metrics["weighted_linear_r2"] = None if r2 is None else round(r2, 6)
        if trend_degree == 2 and len(valid_idx) > 2:
            quad_coeffs, quad_r2 = _weighted_polyfit(x_values, y_values, weights, 2)
            if quad_coeffs is not None:
                metrics["weighted_quadratic_r2"] = None if quad_r2 is None else round(quad_r2, 6)

    if rates:
        max_idx = int(np.argmax(rates))
        min_idx = int(np.argmin(rates))
        metrics["max_rate_bin"] = labels[max_idx]
        metrics["min_rate_bin"] = labels[min_idx]

    box_lines = [
        f"N = {total_n:,}; bins = {len(rows)}",
        f"Spearman rho = {_fmt_optional(metrics.get('spearman_rho'), 3)} (p {format_p(metrics.get('spearman_p'))})",
    ]
    if trend_degree == 2 and metrics.get("weighted_quadratic_r2") is not None:
        box_lines.append(f"Quadratic fit R^2 = {metrics['weighted_quadratic_r2']:.3f}")
        box_lines.append(f"Peak bin: {metrics.get('max_rate_bin', 'n/a')}")
    elif metrics.get("weighted_linear_slope_pp_per_bin") is not None:
        box_lines.append(
            f"Weighted trend = {metrics['weighted_linear_slope_pp_per_bin']:+.2f} pp/bin; "
            f"R^2 = {_fmt_optional(metrics.get('weighted_linear_r2'), 3)}"
        )
    box_lines.append(f"Bin rate range = {rate_min:.1f}-{rate_max:.1f}%")
    if metrics.get("max_ci95_halfwidth_pp") is not None:
        box_lines.append(f"Max 95% CI half-width = {metrics['max_ci95_halfwidth_pp']:.1f} pp")
    return metrics, rows, box_lines


def compute_categorical_success_stats(labels, counts, successes, variable_name):
    rows = build_bin_rows(labels, counts, successes, dataset="category")
    total_n = int(sum(counts))
    total_success = int(sum(successes))
    failures = [int(c) - int(s) for c, s in zip(counts, successes)]

    chi2 = 0.0
    df = max(0, len(labels) - 1)
    p_value = None
    if total_n > 0 and len(labels) > 1:
        col_totals = [int(c) for c in counts]
        row_totals = [total_success, total_n - total_success]
        for observed_row, row_total in zip([successes, failures], row_totals):
            for observed, col_total in zip(observed_row, col_totals):
                expected = row_total * col_total / total_n if total_n else 0.0
                if expected > 0:
                    chi2 += (observed - expected) ** 2 / expected
        if scipy_stats is not None:
            p_value = float(scipy_stats.chi2.sf(chi2, df))
        else:
            p_value = _chi2_sf_approx(chi2, df)

    cramer_v = math.sqrt(chi2 / total_n) if total_n > 0 else 0.0
    rates = [row["success_rate_pct"] for row in rows]
    metrics = {
        "variable": variable_name,
        "n": total_n,
        "successes": total_success,
        "overall_success_rate_pct": round(total_success / total_n * 100.0, 4) if total_n else 0.0,
        "categories": len(labels),
        "chi_square": round(float(chi2), 6),
        "chi_square_df": df,
        "chi_square_p": p_value,
        "cramers_v": round(float(cramer_v), 6),
        "rate_min_pct": round(min(rates), 4) if rates else 0.0,
        "rate_max_pct": round(max(rates), 4) if rates else 0.0,
        "rate_range_pp": round((max(rates) - min(rates)), 4) if rates else 0.0,
    }

    box_lines = [
        f"N = {total_n:,}; categories = {len(labels)}",
        f"Chi-square({df}) = {chi2:.2f}; p {format_p(p_value)}",
        f"Cramer's V = {cramer_v:.3f}",
        f"Rate range = {metrics['rate_min_pct']:.1f}-{metrics['rate_max_pct']:.1f}%",
    ]
    return metrics, rows, box_lines


def _fmt_optional(value, digits=3):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def write_statistical_report(output_path, figure_title, metrics, rows, notes=None):
    base = Path(output_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    notes = notes or []

    json_path = base.with_name(base.name + "_stats.json")
    csv_path = base.with_name(base.name + "_stats.csv")
    md_path = base.with_name(base.name + "_stats.md")

    payload = {
        "figure": figure_title,
        "metrics": _json_safe(metrics),
        "rows": _json_safe(rows),
        "notes": notes,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    if fieldnames:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        csv_path.write_text("", encoding="utf-8")

    lines = [f"# {figure_title}", "", "## Summary metrics", ""]
    for key, value in metrics.items():
        if key.endswith("_p"):
            value = format_p(value)
        lines.append(f"- {key}: {value}")
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend([f"- {note}" for note in notes])
    if rows:
        lines.extend(["", "## Binned/category statistics", ""])
        lines.append("| " + " | ".join(fieldnames) + " |")
        lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(key, "")) for key in fieldnames) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "md": md_path}
