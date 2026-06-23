# -*- coding: utf-8 -*-
"""
regenerate_figs_R3.py
=====================================================================
Regenerate Figs 9-13 (response to reviewer R3: "Figs 9-13 look like
demonstration, not analytics ... augment with parameters, statistics,
metrics").

PRINCIPLE: each figure is an EXACT COPY of the original (same points,
marker sizes, colors, legend, trend line where it existed). NOTHING is
added on top of the data points. All analytics live in a single
STATISTICS BOX in a corner of the plot.

The statistics box reports (computed on raw order-level data):
  - N and binning parameters (number of bins, n per bin);
  - success-rate span across bins;
  - widest 95% Wilson confidence interval across bins;
  - Spearman rho + p  (monotonic association);
  - point-biserial r + p  (linear association);
  - amount: R^2 of a quadratic fit (inverted-U);
  - discount: two-proportion z-test (with vs without discount);
  - day of week: chi-square + Cramer's V (effect size).

Binning, palette, font and point geometry are identical to the original
success_charts_generator_bin.py.

RUN:  python regenerate_figs_R3.py [csv] [out_dir]
Deps: numpy, scipy, matplotlib
=====================================================================
"""
import csv
import math
import json
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

# ---------------- PATHS ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(HERE, 'data_collector_extended.csv'),
    os.path.join(HERE, '..', 'Paper1-v3', 'novi-grafiky-7', 'data_collector_extended.csv'),
]
# also try the original cyrillic-named folder
import glob as _glob
CANDIDATES += _glob.glob(os.path.join(HERE, '..', 'Paper1-v3', '*', 'data_collector_extended.csv'))


def resolve_csv(argv):
    if len(argv) > 1 and os.path.isfile(argv[1]):
        return argv[1]
    for c in CANDIDATES:
        if os.path.isfile(c):
            return os.path.abspath(c)
    sys.exit('CSV not found. Provide path: python regenerate_figs_R3.py <csv>')


CSV_PATH = resolve_csv(sys.argv)
OUT_DIR = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(HERE, 'new_figs_R3')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- FONT (Times New Roman if available, else metric clone) ----------------
_av = {f.name for f in font_manager.fontManager.ttflist}
SERIF = next((f for f in ('Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif') if f in _av), 'serif')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = [SERIF]
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 14
print('Font:', SERIF)

# ---------------- PALETTE (original) ----------------
COLOR_LOW, COLOR_MED, COLOR_HIGH, COLOR_TREND = '#87CEEB', '#4169E1', '#000080', '#4682B4'


# ---------------- DATA / STATISTICS ----------------
def is_success(row):
    return row.get('state') == 'sale'


def read_orders(path):
    with open(path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print('Rows read:', len(rows))
    return rows


def wilson_ci(s, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h) * 100, min(1.0, c + h) * 100)


def rate_color(r):
    return COLOR_LOW if r < 50 else (COLOR_MED if r <= 80 else COLOR_HIGH)


def split_groups(points, min_per, max_g, min_g):
    total = len(points)
    if total <= 0:
        return []
    num = min(max_g, max(min_g, total // min_per))
    g, rem, out, st = total // num, total % num, [], 0
    for i in range(num):
        cur = g + (1 if i < rem else 0)
        if cur <= 0:
            break
        out.append(points[st:st + cur])
        st += cur
    return out


def to_float(v):
    try:
        return float(v)
    except Exception:
        return None


def pstr(p):
    return 'p < 0.001' if p < 0.001 else ('p = %.3f' % p)


def spearman(points):
    x = np.array([v for v, _ in points], float)
    y = np.array([1.0 if ok else 0.0 for _, ok in points])
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def point_biserial(points):
    x = np.array([v for v, _ in points], float)
    y = np.array([1.0 if ok else 0.0 for _, ok in points])
    r, p = stats.pointbiserialr(y, x)
    return float(r), float(p)


def two_prop_z(s1, n1, s2, n2):
    p1, p2 = s1 / n1, s2 / n2
    p = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, 2 * stats.norm.sf(abs(z))


def bins_from_groups(groups, fmt):
    rg, rt, ct, ci = [], [], [], []
    for g in groups:
        a, b = g[0][0], g[-1][0]
        s, n = sum(1 for _, ok in g if ok), len(g)
        rg.append(fmt(a, b))
        rt.append(s / n * 100)
        ct.append(n)
        ci.append(wilson_ci(s, n))
    return rg, rt, ct, ci


def thousands(n):
    return format(int(n), ',').replace(',', ' ')


# ---------------- STATISTICS BOX ----------------
def add_box(ax, lines, loc='upper left'):
    pos = {'upper left': (0.015, 0.985, 'top', 'left'),
           'upper right': (0.985, 0.985, 'top', 'right'),
           'lower left': (0.015, 0.04, 'bottom', 'left')}[loc]
    ax.text(pos[0], pos[1], '\n'.join(lines), transform=ax.transAxes,
            fontsize=12.5, va=pos[2], ha=pos[3], linespacing=1.5, zorder=10,
            bbox=dict(boxstyle='round,pad=0.55', fc='#F4F8FC', ec='#1F3B73', lw=1.3, alpha=0.97))


# ---------------- ORIGINAL SCATTER (exact copy) + box ----------------
def scatter_chart(xlabel, ranges, rates, counts, out, box_lines,
                  show_trend=False, y_top=105, rot=0, ha='center',
                  legend_loc='upper right', box_loc='upper left'):
    plt.figure(figsize=(15, 8))
    xs = [i for i, c in enumerate(counts) if c > 0]
    ys = [rates[i] for i in xs]
    cs = [counts[i] for i in xs]
    colors = [rate_color(r) for r in ys]
    sizes = [max(80, min(150, c / 2)) for c in cs]          # original marker size
    plt.scatter(xs, ys, s=sizes, alpha=0.6, c=colors)        # original alpha, no edge
    if show_trend and len(xs) > 1:
        z = np.polyfit(xs, ys, 1)
        plt.plot(xs, np.poly1d(z)(xs), color=COLOR_TREND, ls='--', alpha=0.8, lw=2, label='Trend line')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.ylim(-5, y_top)
    if len(ranges) <= 10:
        plt.xticks(range(len(ranges)), ranges, rotation=rot, ha=ha, fontsize=14)
    else:
        plt.xticks(range(len(ranges))[::2], [ranges[i] for i in range(0, len(ranges), 2)],
                   rotation=rot, ha=ha, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, ls='--', alpha=0.7)
    for yv, st in [(0, '-'), (50, '--'), (80, '--'), (100, '-')]:
        plt.axhline(y=yv, color='gray', ls=st, alpha=0.3)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_LOW, markersize=10, label='Success Rate < 50%'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MED, markersize=10, label='Success Rate 50-80%'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_HIGH, markersize=10, label='Success Rate > 80%'),
    ]
    if show_trend:
        handles.append(plt.Line2D([0], [0], color=COLOR_TREND, ls='--', lw=2, label='Trend line'))
    plt.legend(handles=handles, loc=legend_loc, fontsize=14)
    add_box(plt.gca(), box_lines, loc=box_loc)
    plt.savefig(out + '.png', format='png', bbox_inches='tight', dpi=300)
    plt.savefig(out + '.svg', format='svg', bbox_inches='tight')
    plt.close()
    print('Saved:', os.path.basename(out))


RESULTS = {}


def main():
    orders = read_orders(CSV_PATH)
    n_all = len(orders)
    succ_all = sum(1 for r in orders if is_success(r))
    overall = succ_all / n_all * 100
    print('Overall success: %.2f%%' % overall)

    # ===== Fig 9: messages =====
    pts = [(int(r['messages_count']), is_success(r)) for r in orders
           if r.get('messages_count', '').strip().lstrip('-').isdigit() and int(r['messages_count']) >= 0]
    pts.sort(key=lambda t: t[0])
    rho, p = spearman(pts)
    rb, pb = point_biserial(pts)
    rg, rt, ct, ci = bins_from_groups(split_groups(pts, 15, 20, 5), lambda a, b: ('%d' % a) if a == b else ('%d-%d' % (a, b)))
    maxhw = max((c[1] - c[0]) / 2 for c in ci)
    box = ['Statistics',
           'N = %s   |   %d equal-freq. bins (n ~ %s)' % (thousands(n_all), len(rg), thousands(ct[0])),
           'Success rate: %.1f%% - %.1f%%' % (rt[0], rt[-1]),
           r'Per-bin 95%% Wilson CI $\leq \pm$%.1f pp' % maxhw,
           r'Spearman $\rho$ = %.3f (%s)' % (rho, pstr(p)),
           'Point-biserial r = %.3f (%s)' % (rb, pstr(pb))]
    scatter_chart('Messages Count Range', rg, rt, ct, os.path.join(OUT_DIR, 'messages_success_chart'),
                  box, legend_loc='lower right', box_loc='upper left')
    RESULTS['Fig9_messages'] = dict(rho=rho, p=p, r_pb=rb, p_pb=pb, N=n_all, bins=len(rg),
                                    first=[rg[0], round(rt[0], 1), ct[0]], last=[rg[-1], round(rt[-1], 1), ct[-1]],
                                    max_ci_halfwidth=round(maxhw, 2))

    # ===== Fig 10: changes (trend line, as in original) =====
    pts = [(int(r['changes_count']), is_success(r)) for r in orders
           if r.get('changes_count', '').strip().lstrip('-').isdigit() and int(r['changes_count']) >= 0]
    pts.sort(key=lambda t: t[0])
    rho, p = spearman(pts)
    rb, pb = point_biserial(pts)
    rg, rt, ct, ci = bins_from_groups(split_groups(pts, 1, 6, 6), lambda a, b: ('%d' % a) if a == b else ('%d-%d' % (a, b)))
    xs = list(range(len(rt)))
    z = np.polyfit(xs, rt, 1)
    fit = np.poly1d(z)(xs)
    r2 = 1 - np.sum((np.array(rt) - fit) ** 2) / np.sum((np.array(rt) - np.mean(rt)) ** 2)
    maxhw = max((c[1] - c[0]) / 2 for c in ci)
    box = ['Statistics',
           'N = %s   |   %d equal-freq. bins' % (thousands(n_all), len(rg)),
           'Success rate: %.1f%% - %.1f%%' % (rt[0], rt[-1]),
           r'Per-bin 95%% Wilson CI $\leq \pm$%.1f pp' % maxhw,
           r'Spearman $\rho$ = %.3f (%s)' % (rho, pstr(p)),
           r'Trend: +%.1f pp/bin ($R^2$ = %.2f)' % (z[0], r2)]
    scatter_chart('Changes Count Range', rg, rt, ct, os.path.join(OUT_DIR, 'changes_success_chart'),
                  box, show_trend=True, legend_loc='lower right', box_loc='upper left')
    RESULTS['Fig10_changes'] = dict(rho=rho, p=p, r_pb=rb, p_pb=pb, N=n_all, bins=len(rg),
                                    slope_pp=round(float(z[0]), 2), trend_R2=round(float(r2), 3),
                                    first=[rg[0], round(rt[0], 1), ct[0]], last=[rg[-1], round(rt[-1], 1), ct[-1]])

    # ===== Fig 11: amount (non-monotonic; quadratic R^2 in box) =====
    pts = [(to_float(r['total_amount']), is_success(r)) for r in orders
           if to_float(r.get('total_amount')) is not None and to_float(r['total_amount']) > 0]
    pts.sort(key=lambda t: t[0])
    rho, p = spearman(pts)

    def amt(a, b):
        f = lambda v: (('%.0fK' % (v / 1000)) if v >= 1000 else ('%.0f' % v))
        return f(a) if a == b else (f(a) + '-' + f(b))

    rg, rt, ct, ci = bins_from_groups(split_groups(pts, 20, 30, 5), amt)
    xs = list(range(len(rt)))
    z2 = np.polyfit(xs, rt, 2)
    fit = np.poly1d(z2)(xs)
    r2 = 1 - np.sum((np.array(rt) - fit) ** 2) / np.sum((np.array(rt) - np.mean(rt)) ** 2)
    peak = int(np.argmax(rt))
    box = ['Statistics',
           'N = %s   |   %d equal-freq. bins' % (thousands(len(pts)), len(rg)),
           'Peak: %.1f%% at %s USD' % (rt[peak], rg[peak]),
           'Tails: %.1f%% (low) / %.1f%% (high)' % (rt[0], rt[-1]),
           r'Spearman $\rho$ = %.3f (%s), non-monotonic' % (rho, pstr(p)),
           r'Quadratic fit (inverted-U): $R^2$ = %.2f' % r2]
    scatter_chart('Order Amount Range (USD)', rg, rt, ct, os.path.join(OUT_DIR, 'amount_success_chart'),
                  box, rot=45, ha='right', legend_loc='upper right', box_loc='upper left')
    RESULTS['Fig11_amount'] = dict(rho=rho, p=p, N=len(pts), bins=len(rg), quad_R2=round(float(r2), 3),
                                   peak=[rg[peak], round(rt[peak], 1), ct[peak]],
                                   first=[rg[0], round(rt[0], 1), ct[0]], last=[rg[-1], round(rt[-1], 1), ct[-1]])

    # ===== Fig 12: discount (exact copy of two-panel original) + box in (b) =====
    pos_points, zero_orders = [], []
    for r in orders:
        d = to_float(r.get('discount_total'))
        if d is None:
            continue
        if d > 0:
            pos_points.append((d, is_success(r)))
        elif d == 0:
            zero_orders.append((r.get('customer_id'), is_success(r)))
    pos_points.sort(key=lambda t: t[0])
    zt = len(zero_orders)
    zs = sum(1 for _, ok in zero_orders if ok)
    ptl = len(pos_points)
    pss = sum(1 for _, ok in pos_points if ok)

    zero_sub = []
    if zt > 0:
        cstat = {}
        for cid, ok in zero_orders:
            cstat.setdefault(cid, {'t': 0, 's': 0})
            cstat[cid]['t'] += 1
            cstat[cid]['s'] += 1 if ok else 0
        bucket = {k: 0 for k in range(0, 101, 10)}
        for stt in cstat.values():
            r = stt['s'] / stt['t'] * 100 if stt['t'] else 0
            b = max(0, min(100, int(round(r / 10.0) * 10)))
            bucket[b] += stt['t']
        for b in sorted(bucket):
            if bucket[b] > 0:
                zero_sub.append({'rate': float(b), 'count': bucket[b]})

    pr, prt, pct, pci = bins_from_groups(split_groups(pos_points, 20, 30, 5),
                                         lambda a, b: ('%.0f' % a) if a == b else ('%.0f-%.0f' % (a, b)))
    rho_d, p_d = spearman(pos_points) if ptl > 2 else (float('nan'), float('nan'))
    zsc, zp = two_prop_z(pss, ptl, zs, zt)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 8))
    fs = 16
    labels = ['No discount', 'With discount']
    X = np.arange(2)
    if zero_sub:
        n = len(zero_sub)
        offs = [0.0] if n == 1 else list(np.linspace(-0.8, 0.8, n))
        for off, sg in zip(offs, zero_sub):
            axL.scatter(X[0] + off, sg['rate'], s=max(120, min(260, sg['count'])), c=rate_color(sg['rate']), alpha=0.7)
            axL.text(X[0] + off, sg['rate'] + 3, '%.1f%%\n(n=%d)' % (sg['rate'], sg['count']), ha='center', va='bottom', fontsize=fs)
    axL.scatter(X[1], pss / ptl * 100, s=max(120, min(260, ptl)), c=rate_color(pss / ptl * 100), alpha=0.7)
    axL.text(X[1], pss / ptl * 100 + 3, '%.1f%%\n(n=%d)' % (pss / ptl * 100, ptl), ha='center', va='bottom', fontsize=fs)
    axL.set_xticks(X)
    axL.set_xticklabels(labels, fontsize=fs)
    axL.tick_params(axis='y', labelsize=fs)
    axL.set_ylim(-5, 112)
    axL.set_ylabel('Success Rate (%)', fontsize=fs)
    axL.grid(True, ls='--', alpha=0.3, axis='y')
    axL.axvline(x=0.9, color='gray', ls='--', alpha=0.5, lw=1)
    axL.set_xlim(-1.1, 1.4)

    xp = list(range(len(pr)))
    axR.scatter(xp, prt, s=[max(80, min(150, c / 2)) for c in pct], alpha=0.6, c=[rate_color(r) for r in prt])
    axR.set_xlabel('Discount Amount Range (USD)', fontsize=fs)
    axR.set_ylabel('Success Rate (%)', fontsize=fs)
    axR.set_ylim(-5, 112)
    if len(pr) <= 10:
        axR.set_xticks(range(len(pr)))
        axR.set_xticklabels(pr, fontsize=fs)
    else:
        axR.set_xticks(range(len(pr))[::2])
        axR.set_xticklabels([pr[i] for i in range(0, len(pr), 2)], fontsize=fs)
    axR.tick_params(axis='y', labelsize=fs)
    axR.grid(True, ls='--', alpha=0.7)
    for yv, st in [(0, '-'), (50, '--'), (80, '--'), (100, '-')]:
        axR.axhline(y=yv, color='gray', ls=st, alpha=0.3)
    hb = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_LOW, markersize=10, label='Success Rate < 50%'),
          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MED, markersize=10, label='Success Rate 50-80%'),
          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_HIGH, markersize=10, label='Success Rate > 80%')]
    axR.legend(handles=hb, loc='lower right', fontsize=fs)
    rho_txt = (r'Spearman $\rho$ = %.3f (%s)' % (rho_d, pstr(p_d))) if ptl > 2 else 'n/a'
    add_box(axR, ['Statistics',
                  'No discount: %.1f%% (n = %s)' % (zs / zt * 100, thousands(zt)),
                  'With discount: %.1f%% (n = %d)' % (pss / ptl * 100, ptl),
                  'Two-proportion z-test: %s' % pstr(zp),
                  'Among discounted: ' + rho_txt,
                  'Underpowered subsample (n = %d)' % ptl], loc='upper left')
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    lb, rbx = axL.get_position(), axR.get_position()
    fig.text((lb.x0 + lb.x1) / 2, lb.y0 - 0.085, '(a)', ha='center', va='top', fontsize=fs)
    fig.text((rbx.x0 + rbx.x1) / 2, rbx.y0 - 0.085, '(b)', ha='center', va='top', fontsize=fs)
    fig.savefig(os.path.join(OUT_DIR, 'discount_success_chart.png'), format='png', bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUT_DIR, 'discount_success_chart.svg'), format='svg', bbox_inches='tight')
    plt.close(fig)
    print('Saved: discount_success_chart')
    RESULTS['Fig12_discount'] = dict(no_discount=[round(zs / zt * 100, 1), zt],
                                     with_discount=[round(pss / ptl * 100, 1), ptl],
                                     two_prop_p=zp, rho=rho_d, p=p_d)

    # ===== Fig 13: day of week (exact copy; chi2 + Cramer's V in box) =====
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    tot = {d: 0 for d in days}
    suc = {d: 0 for d in days}
    for r in orders:
        d = r.get('day_of_week')
        if d in tot:
            tot[d] += 1
            suc[d] += 1 if is_success(r) else 0
    rt = [suc[d] / tot[d] * 100 for d in days]
    ct = [tot[d] for d in days]
    chi2, pchi, dof, _ = stats.chi2_contingency(np.array([[suc[d], tot[d] - suc[d]] for d in days]))
    cramer = math.sqrt(chi2 / (n_all * 1))
    lo = min(range(7), key=lambda i: rt[i])
    hi = max(range(7), key=lambda i: rt[i])
    plt.figure(figsize=(15, 8))
    xs = list(range(7))
    plt.scatter(xs, rt, s=[max(100, min(260, c)) for c in ct], alpha=0.6, c=[rate_color(r) for r in rt])
    plt.xlabel('Day of Week', fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.ylim(-5, 105)
    plt.xticks(xs, days, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, ls='--', alpha=0.7)
    for yv, st in [(0, '-'), (50, '--'), (80, '--'), (100, '-')]:
        plt.axhline(y=yv, color='gray', ls=st, alpha=0.3)
    hb = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_LOW, markersize=10, label='Success Rate < 50%'),
          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MED, markersize=10, label='Success Rate 50-80%'),
          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_HIGH, markersize=10, label='Success Rate > 80%')]
    plt.legend(handles=hb, loc='lower right', fontsize=14)
    add_box(plt.gca(), ['Statistics',
                        'N = %s   |   overall mean = %.1f%%' % (thousands(n_all), overall),
                        'Range: %.1f%% (%s) - %.1f%% (%s)' % (rt[lo], days[lo][:3], rt[hi], days[hi][:3]),
                        r'$\chi^2$(6) = %.1f (%s)' % (chi2, pstr(pchi)),
                        r"Cramer's V = %.3f (negligible)" % cramer,
                        'Weekend n small: Sat %d, Sun %d' % (tot['Saturday'], tot['Sunday'])], loc='upper left')
    plt.savefig(os.path.join(OUT_DIR, 'dayofweek_success_chart.png'), format='png', bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUT_DIR, 'dayofweek_success_chart.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print('Saved: dayofweek_success_chart')
    RESULTS['Fig13_dayofweek'] = dict(rates=[round(x, 1) for x in rt], counts=ct, overall=round(overall, 1),
                                      chi2=round(float(chi2), 1), p_chi=pchi, cramers_v=round(cramer, 3))

    with open(os.path.join(OUT_DIR, 'figure_statistics_R3.json'), 'w', encoding='utf-8') as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=str)
    print('\n=== STATISTICS (figure_statistics_R3.json) ===')
    print(json.dumps(RESULTS, indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
