# -*- coding: utf-8 -*-
"""
regenerate_figs_R3.py
=====================================================================
Перегенерація Рис. 9-13 (відповідь на зауваження рецензента R3:
"Figs 9-13 ... look like demonstration, not like analytics ...
augment the figures with ... parameters, statistics, metrics").

ПРИНЦИП: рисунок є ТОЧНОЮ КОПІЄЮ оригіналу (ті самі точки, розміри
маркерів, кольори, легенда, лінія тренду там, де вона була). До самих
точок НІЧОГО не додається. Уся аналітика виводиться в ОКРЕМІЙ РАМЦІ
(statistics box) у кутку графіка.

Аналітична рамка містить (обчислюється на сирих даних рівня замовлення):
  - N та параметри бінінгу (к-сть бінів, n у біні);
  - діапазон частки успіху по бінах;
  - максимальну ширину 95% довірчого інтервалу Вілсона по бінах;
  - коеф. кореляції Спірмена (rho) + p  -- монотонний зв'язок;
  - точково-бісеріальний r + p  -- лінійний зв'язок;
  - для суми: R2 квадратичної апроксимації (перевернута U);
  - для знижки: z-тест двох часток (зі знижкою / без);
  - для днів тижня: chi-square + Cramer's V (розмір ефекту).

Бінінг, палітра, шрифт і вся геометрія точок -- ідентичні оригінальному
success_charts_generator_bin.py.

ЗАПУСК:  python regenerate_figs_R3.py [csv] [out_dir]
Залежності: numpy, scipy, matplotlib
=====================================================================
"""
import csv, math, json, os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

# ---------------- ШЛЯХИ ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(HERE, 'data_collector_extended.csv'),
    os.path.join(HERE, '..', 'Paper1-v3', 'нові-графіки-7', 'data_collector_extended.csv'),
    os.path.join(HERE, '..', 'Paper1-v3', 'нові-графіки-8', 'data_collector_extended.csv'),
]
def resolve_csv(argv):
    if len(argv) > 1 and os.path.isfile(argv[1]):
        return argv[1]
    for c in CANDIDATES:
        if os.path.isfile(c):
            return os.path.abspath(c)
    sys.exit('CSV не знайдено. Вкажіть шлях: python regenerate_figs_R3.py <csv>')
CSV_PATH = resolve_csv(sys.argv)
OUT_DIR = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(HERE, 'new_figs_R3')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- ШРИФТ (як в оригіналі: Times New Roman / сумісний) ----------------
_av = {f.name for f in font_manager.fontManager.ttflist}
SERIF = next((f for f in ('Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif') if f in _av), 'serif')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = [SERIF]
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 14
print('Шрифт:', SERIF)

# ---------------- ПАЛІТРА (оригінал) ----------------
COLOR_LOW, COLOR_MED, COLOR_HIGH, COLOR_TREND = '#87CEEB', '#4169E1', '#000080', '#4682B4'

# ---------------- ДАНІ / СТАТИСТИКА ----------------
def is_success(row): return row.get('state') == 'sale'

def read_orders(path):
    with open(path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print('Зчитано рядків:', len(rows))
    return rows

def wilson_ci(s, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = s / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = (z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
    return (max(0.0, c-h)*100, min(1.0, c+h)*100)

def rate_color(r): return COLOR_LOW if r < 50 else (COLOR_MED if r <= 80 else COLOR_HIGH)

def split_groups(points, min_per, max_g, min_g):
    total = len(points)
    if total <= 0: return []
    num = min(max_g, max(min_g, total // min_per))
    g, rem, out, st = total // num, total % num, [], 0
    for i in range(num):
        cur = g + (1 if i < rem else 0)
        if cur <= 0: break
        out.append(points[st:st+cur]); st += cur
    return out

def to_float(v):
    try: return float(v)
    except Exception: return None

def pstr(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'

def spearman(points):
    x = np.array([v for v, _ in points], float)
    y = np.array([1.0 if ok else 0.0 for _, ok in points])
    rho, p = stats.spearmanr(x, y); return float(rho), float(p)

def point_biserial(points):
    x = np.array([v for v, _ in points], float)
    y = np.array([1.0 if ok else 0.0 for _, ok in points])
    r, p = stats.pointbiserialr(y, x); return float(r), float(p)

def two_prop_z(s1, n1, s2, n2):
    p1, p2 = s1/n1, s2/n2
    p = (s1+s2)/(n1+n2)
    se = math.sqrt(p*(1-p)*(1/n1+1/n2))
    if se == 0: return 0.0, 1.0
    z = (p1-p2)/se
    return z, 2*stats.norm.sf(abs(z))

def bins_from_groups(groups, fmt):
    rg, rt, ct, ci = [], [], [], []
    for g in groups:
        a, b = g[0][0], g[-1][0]
        s, n = sum(1 for _, ok in g if ok), len(g)
        rg.append(fmt(a, b)); rt.append(s/n*100); ct.append(n); ci.append(wilson_ci(s, n))
    return rg, rt, ct, ci

# ---------------- АНАЛІТИЧНА РАМКА ----------------
def add_box(ax, lines, loc='upper left'):
    pos = {'upper left': (0.015, 0.985, 'top', 'left'),
           'upper right': (0.985, 0.985, 'top', 'right'),
           'lower left': (0.015, 0.04, 'bottom', 'left')}[loc]
    txt = ax.text(pos[0], pos[1], '\n'.join(lines), transform=ax.transAxes,
                  fontsize=12, va=pos[2], ha=pos[3], linespacing=1.45, zorder=10,
                  bbox=dict(boxstyle='round,pad=0.55', fc='#F4F8FC', ec='#1F3B73', lw=1.3, alpha=0.97))
    return txt

# ---------------- ОРИГІНАЛЬНИЙ МАЛЮВАЛЬНИК (точна копія) + рамка ----------------
def scatter_chart(xlabel, ranges, rates, counts, out, box_lines,
                  show_trend=False, y_top=105, rot=0, ha='center',
                  legend_loc='upper right', box_loc='upper left'):
    plt.figure(figsize=(15, 8))
    xs = [i for i, c in enumerate(counts) if c > 0]
    ys = [rates[i] for i in xs]
    cs = [counts[i] for i in xs]
    colors = [rate_color(r) for r in ys]
    sizes = [max(80, min(150, c/2)) for c in cs]                 # <- РОЗМІР ТОЧОК ЯК В ОРИГІНАЛІ
    plt.scatter(xs, ys, s=sizes, alpha=0.6, c=colors)            # <- alpha 0.6, без обведення (оригінал)
    if show_trend and len(xs) > 1:
        z = np.polyfit(xs, ys, 1)
        plt.plot(xs, np.poly1d(z)(xs), color=COLOR_TREND, ls='--', alpha=0.8, lw=2, label='Trend line')
    plt.xlabel(xlabel, fontsize=14); plt.ylabel('Success Rate (%)', fontsize=14)
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
    plt.savefig(f'{out}.png', format='png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{out}.svg', format='svg', bbox_inches='tight')
    plt.close()
    print('Збережено:', os.path.basename(out) + '.png/.svg')

RESULTS = {}

def main():
    orders = read_orders(CSV_PATH)
    n_all = len(orders); succ_all = sum(1 for r in orders if is_success(r))
    overall = succ_all / n_all * 100
    print(f'Загальна успішність: {overall:.2f}%')

    # ===== Рис. 9: messages =====
    pts = [(int(r['messages_count']), is_success(r)) for r in orders
           if r.get('messages_count', '').strip().lstrip('-').isdigit() and int(r['messages_count']) >= 0]
    pts.sort(key=lambda t: t[0])
    rho, p = spearman(pts); rb, pb = point_biserial(pts)
    rg, rt, ct, ci = bins_from_groups(split_groups(pts, 15, 20, 5), lambda a, b: f'{a}' if a == b else f'{a}-{b}')
    maxhw = max((c[1]-c[0])/2 for c in ci)
    box = ['Statistics',
           f'N = {n_all:,}'.replace(',', ' ') + f' · {len(rg)} equal-freq. bins (n ≈ {ct[0]:,})'.replace(',', ' '),
           f'Success rate: {rt[0]:.1f}% – {rt[-1]:.1f}%',
           f'Per-bin 95% CI (Wilson) ≤ ±{maxhw:.1f} pp',
           f'Spearman ρ = {rho:.3f} ({pstr(p)})',
           f'Point-biserial r = {rb:.3f} ({pstr(pb)})']
    scatter_chart('Messages Count Range', rg, rt, ct, os.path.join(OUT_DIR, 'messages_success_chart'),
                  box, legend_loc='lower right', box_loc='upper left')
    RESULTS['Fig9_messages'] = dict(rho=rho, p=p, r_pb=rb, p_pb=pb, N=n_all, bins=len(rg),
                                    first=[rg[0], round(rt[0], 1), ct[0]], last=[rg[-1], round(rt[-1], 1), ct[-1]],
                                    max_ci_halfwidth=round(maxhw, 2))

    # ===== Рис. 10: changes (з лінією тренду — як в оригіналі) =====
    pts = [(int(r['changes_count']), is_success(r)) for r in orders
           if r.get('changes_count', '').strip().lstrip('-').isdigit() and int(r['changes_count']) >= 0]
    pts.sort(key=lambda t: t[0])
    rho, p = spearman(pts); rb, pb = point_biserial(pts)
    rg, rt, ct, ci = bins_from_groups(split_groups(pts, 1, 6, 6), lambda a, b: f'{a}' if a == b else f'{a}-{b}')
    xs = list(range(len(rt))); z = np.polyfit(xs, rt, 1)
    fit = np.poly1d(z)(xs); ss_res = np.sum((np.array(rt)-fit)**2); ss_tot = np.sum((np.array(rt)-np.mean(rt))**2)
    r2 = 1 - ss_res/ss_tot
    maxhw = max((c[1]-c[0])/2 for c in ci)
    box = ['Statistics',
           f'N = {n_all:,}'.replace(',', ' ') + f' · {len(rg)} equal-freq. bins',
           f'Success rate: {rt[0]:.1f}% – {rt[-1]:.1f}%',
           f'Per-bin 95% CI (Wilson) ≤ ±{maxhw:.1f} pp',
           f'Spearman ρ = {rho:.3f} ({pstr(p)})',
           f'Trend line: +{z[0]:.1f} pp/bin (R² = {r2:.2f})']
    scatter_chart('Changes Count Range', rg, rt, ct, os.path.join(OUT_DIR, 'changes_success_chart'),
                  box, show_trend=True, legend_loc='lower right', box_loc='upper left')
    RESULTS['Fig10_changes'] = dict(rho=rho, p=p, r_pb=rb, p_pb=pb, N=n_all, bins=len(rg),
                                    slope_pp=round(float(z[0]), 2), trend_R2=round(float(r2), 3),
                                    first=[rg[0], round(rt[0], 1), ct[0]], last=[rg[-1], round(rt[-1], 1), ct[-1]])

    # ===== Рис. 11: amount (немонотонна, квадратична апроксимація в рамці) =====
    pts = [(to_float(r['total_amount']), is_success(r)) for r in orders
           if to_float(r.get('total_amount')) is not None and to_float(r['total_amount']) > 0]
    pts.sort(key=lambda t: t[0])
    rho, p = spearman(pts)
    def amt(a, b):
        f = lambda v: (f'{v/1000:.0f}K' if v >= 1000 else f'{v:.0f}')
        return f(a) if a == b else f'{f(a)}-{f(b)}'
    rg, rt, ct, ci = bins_from_groups(split_groups(pts, 20, 30, 5), amt)
    xs = list(range(len(rt))); z2 = np.polyfit(xs, rt, 2)
    fit = np.poly1d(z2)(xs); ss_res = np.sum((np.array(rt)-fit)**2); ss_tot = np.sum((np.array(rt)-np.mean(rt))**2)
    r2 = 1 - ss_res/ss_tot
    peak = int(np.argmax(rt))
    box = ['Statistics',
           f'N = {len(pts):,}'.replace(',', ' ') + f' · {len(rg)} equal-freq. bins',
           f'Peak: {rt[peak]:.1f}% at {rg[peak]} USD',
           f'Tails: {rt[0]:.1f}% (low) / {rt[-1]:.1f}% (high)',
           f'Spearman ρ = {rho:.3f} ({pstr(p)}) — non-monotonic',
           f'Quadratic fit (inverted-U): R² = {r2:.2f}']
    scatter_chart('Order Amount Range (USD)', rg, rt, ct, os.path.join(OUT_DIR, 'amount_success_chart'),
                  box, rot=45, ha='right', legend_loc='upper right', box_loc='upper left')
    RESULTS['Fig11_amount'] = dict(rho=rho, p=p, N=len(pts), bins=len(rg), quad_R2=round(float(r2), 3),
                                   peak=[rg[peak], round(rt[peak], 1), ct[peak]],
                                   first=[rg[0], round(rt[0], 1), ct[0]], last=[rg[-1], round(rt[-1], 1), ct[-1]])

    # ===== Рис. 12: discount (ТОЧНА КОПІЯ двопанельного оригіналу) + рамка у (b) =====
    pos_points, zero_orders = [], []
    for r in orders:
        d = to_float(r.get('discount_total'))
        if d is None: continue
        if d > 0: pos_points.append((d, is_success(r)))
        elif d == 0: zero_orders.append((r.get('customer_id'), is_success(r)))
    pos_points.sort(key=lambda t: t[0])
    zt = len(zero_orders); zs = sum(1 for _, ok in zero_orders if ok)
    ptl = len(pos_points); pss = sum(1 for _, ok in pos_points if ok)

    # subgroups no-discount (оригінальна логіка)
    zero_sub = []
    if zt > 0:
        cstat = {}
        for cid, ok in zero_orders:
            cstat.setdefault(cid, {'t': 0, 's': 0}); cstat[cid]['t'] += 1; cstat[cid]['s'] += 1 if ok else 0
        bucket = {k: 0 for k in range(0, 101, 10)}
        for st_ in cstat.values():
            r = st_['s']/st_['t']*100 if st_['t'] else 0
            b = max(0, min(100, int(round(r/10.0)*10))); bucket[b] += st_['t']
        for b in sorted(bucket):
            if bucket[b] > 0: zero_sub.append({'rate': float(b), 'count': bucket[b]})

    pr, prt, pct, pci = bins_from_groups(split_groups(pos_points, 20, 30, 5),
                                         lambda a, b: f'{a:.0f}' if a == b else f'{a:.0f}-{b:.0f}')
    rho_d, p_d = spearman(pos_points) if ptl > 2 else (float('nan'), float('nan'))
    zsc, zp = two_prop_z(pss, ptl, zs, zt)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 8)); fs = 16
    labels = ['No discount', 'With discount']; X = np.arange(2)
    if zero_sub:
        n = len(zero_sub); offs = [0.0] if n == 1 else list(np.linspace(-0.8, 0.8, n))
        for off, sg in zip(offs, zero_sub):
            axL.scatter(X[0]+off, sg['rate'], s=max(120, min(260, sg['count'])), c=rate_color(sg['rate']), alpha=0.7)
            axL.text(X[0]+off, sg['rate']+3, f"{sg['rate']:.1f}%\n(n={sg['count']})", ha='center', va='bottom', fontsize=fs)
    axL.scatter(X[1], pss/ptl*100, s=max(120, min(260, ptl)), c=rate_color(pss/ptl*100), alpha=0.7)
    axL.text(X[1], pss/ptl*100+3, f'{pss/ptl*100:.1f}%\n(n={ptl})', ha='center', va='bottom', fontsize=fs)
    axL.set_xticks(X); axL.set_xticklabels(labels, fontsize=fs); axL.tick_params(axis='y', labelsize=fs)
    axL.set_ylim(-5, 112); axL.set_ylabel('Success Rate (%)', fontsize=fs)
    axL.grid(True, ls='--', alpha=0.3, axis='y'); axL.axvline(x=0.9, color='gray', ls='--', alpha=0.5, lw=1); axL.set_xlim(-1.1, 1.4)

    xp = list(range(len(pr)))
    axR.scatter(xp, prt, s=[max(80, min(150, c/2)) for c in pct], alpha=0.6, c=[rate_color(r) for r in prt])
    axR.set_xlabel('Discount Amount Range (USD)', fontsize=fs); axR.set_ylabel('Success Rate (%)', fontsize=fs)
    axR.set_ylim(-5, 112)
    if len(pr) <= 10:
        axR.set_xticks(range(len(pr))); axR.set_xticklabels(pr, fontsize=fs)
    else:
        axR.set_xticks(range(len(pr))[::2]); axR.set_xticklabels([pr[i] for i in range(0, len(pr), 2)], fontsize=fs)
    axR.tick_params(axis='y', labelsize=fs); axR.grid(True, ls='--', alpha=0.7)
    for yv, st in [(0, '-'), (50, '--'), (80, '--'), (100, '-')]:
        axR.axhline(y=yv, color='gray', ls=st, alpha=0.3)
    hb = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_LOW, markersize=10, label='Success Rate < 50%'),
          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MED, markersize=10, label='Success Rate 50-80%'),
          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_HIGH, markersize=10, label='Success Rate > 80%')]
    axR.legend(handles=hb, loc='lower right', fontsize=fs)
    rho_txt = f'Spearman ρ = {rho_d:.3f} ({pstr(p_d)})' if ptl > 2 else 'n/a'
    add_box(axR, ['Statistics',
                  f'No discount: {zs/zt*100:.1f}% (n = {zt:,})'.replace(',', ' '),
                  f'With discount: {pss/ptl*100:.1f}% (n = {ptl})',
                  f'Two-proportion z-test: {pstr(zp)}',
                  f'Among discounted: {rho_txt}',
                  f'Underpowered subsample (n = {ptl})'], loc='upper left')
    # підписи панелей (a)/(b) — як в оригіналі
    plt.tight_layout(); fig.subplots_adjust(bottom=0.2)
    lb, rbx = axL.get_position(), axR.get_position()
    fig.text((lb.x0+lb.x1)/2, lb.y0-0.085, '(a)', ha='center', va='top', fontsize=fs)
    fig.text((rbx.x0+rbx.x1)/2, rbx.y0-0.085, '(b)', ha='center', va='top', fontsize=fs)
    fig.savefig(os.path.join(OUT_DIR, 'discount_success_chart.png'), format='png', bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUT_DIR, 'discount_success_chart.svg'), format='svg', bbox_inches='tight')
    plt.close(fig); print('Збережено: discount_success_chart.png/.svg')
    RESULTS['Fig12_discount'] = dict(no_discount=[round(zs/zt*100, 1), zt], with_discount=[round(pss/ptl*100, 1), ptl],
                                     two_prop_p=zp, rho=rho_d, p=p_d)

    # ===== Рис. 13: day of week (ТОЧНА КОПІЯ; chi2 + Cramer's V у рамці) =====
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    tot = {d: 0 for d in days}; suc = {d: 0 for d in days}
    for r in orders:
        d = r.get('day_of_week')
        if d in tot:
            tot[d] += 1; suc[d] += 1 if is_success(r) else 0
    rt = [suc[d]/tot[d]*100 for d in days]; ct = [tot[d] for d in days]
    chi2, pchi, dof, _ = stats.chi2_contingency(np.array([[suc[d], tot[d]-suc[d]] for d in days]))
    cramer = math.sqrt(chi2 / (n_all * 1))  # min(r,c)-1 = 1
    lo = min(range(7), key=lambda i: rt[i]); hi = max(range(7), key=lambda i: rt[i])
    plt.figure(figsize=(15, 8))
    xs = list(range(7)); colors = [rate_color(r) for r in rt]; sizes = [max(100, min(260, c)) for c in ct]
    plt.scatter(xs, rt, s=sizes, alpha=0.6, c=colors)
    plt.xlabel('Day of Week', fontsize=14); plt.ylabel('Success Rate (%)', fontsize=14)
    plt.ylim(-5, 105); plt.xticks(xs, days, fontsize=14); plt.yticks(fontsize=14)
    plt.grid(True, ls='--', alpha=0.7)
    for yv, st in [(0, '-'), (50, '--'), (80, '--'), (100, '-')]:
        plt.axhline(y=yv, color='gray', ls=st, alpha=0.3)
    hb = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_LOW, markersize=10, label='Success Rate < 50%'),
          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MED, markersize=10, label='Success Rate 50-80%'),
          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_HIGH, markersize=10, label='Success Rate > 80%')]
    plt.legend(handles=hb, loc='lower right', fontsize=14)
    add_box(plt.gca(), ['Statistics',
                        f'N = {n_all:,}'.replace(',', ' ') + f' · overall mean = {overall:.1f}%',
                        f'Range: {rt[lo]:.1f}% ({days[lo][:3]}) – {rt[hi]:.1f}% ({days[hi][:3]})',
                        f'χ²(6) = {chi2:.1f} ({pstr(pchi)})',
                        f"Cramér's V = {cramer:.3f} (negligible effect)",
                        f'Weekend n small: Sat {tot["Saturday"]}, Sun {tot["Sunday"]}'], loc='upper left')
    plt.savefig(os.path.join(OUT_DIR, 'dayofweek_success_chart.png'), format='png', bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUT_DIR, 'dayofweek_success_chart.svg'), format='svg', bbox_inches='tight')
    plt.close(); print('Збережено: dayofweek_success_chart.png/.svg')
    RESULTS['Fig13_dayofweek'] = dict(rates=[round(x, 1) for x in rt], counts=ct, overall=round(overall, 1),
                                      chi2=round(float(chi2), 1), p_chi=pchi, cramers_v=round(cramer, 3))

    with open(os.path.join(OUT_DIR, 'figure_statistics_R3.json'), 'w', encoding='utf-8') as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=str)
    print('\n=== СТАТИСТИКА (figure_statistics_R3.json) ===')
    print(json.dumps(RESULTS, indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
