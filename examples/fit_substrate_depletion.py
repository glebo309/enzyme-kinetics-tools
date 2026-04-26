"""
Substrate depletion fitting — inactivation model comparison
with properly constrained Km bounds

Example: comparing time-dependent vs turnover-dependent
enzyme inactivation models.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ET_mM = 0.035  # 35 µM

# Data
t_g1 = np.array([0, 2, 5, 30, 60], dtype=float)
S_05  = np.array([0.53, 0.35, 0.27, 0.06, 0.00])
S_10  = np.array([1.02, 0.56, 0.51, 0.46, 0.25])
S_15  = np.array([1.50, 1.02, 0.96, 0.59, 0.37])
S_25  = np.array([2.50, 2.09, 1.98, 1.34, 0.97])
t_g2 = np.array([0, 10, 30, 60], dtype=float)
S_50  = np.array([5.00, 4.65, 4.38, 3.70])
S_100 = np.array([10.00, 9.86, 9.75, 8.57])

times_list = [t_g1, t_g1, t_g1, t_g1, t_g2, t_g2]
S_obs_list = [S_05, S_10, S_15, S_25, S_50, S_100]
S0_list    = [s[0] for s in S_obs_list]
labels     = ["0.5 mM", "1 mM", "1.5 mM", "2.5 mM", "5 mM", "10 mM"]
FLOOR = 0.005


# ── Models ────────────────────────────────────────────────────
# All 2-state: y = [S, e] where e = E/E0 (normalized), params use Vmax not kcat

def ode_time_inact(y, t, vmax, km, k_inact):
    """Time-dependent inactivation (spontaneous Fe loss / auto-oxidation)"""
    S, e = max(y[0], 0), max(y[1], 0)
    rate = vmax * e * S / (km + S + 1e-12)
    return [-rate, -k_inact * e]

def ode_S_inact(y, t, vmax, km, k_inact_S):
    """[S]-dependent inactivation (Fe consumed faster when more turnovers)"""
    S, e = max(y[0], 0), max(y[1], 0)
    rate = vmax * e * S / (km + S + 1e-12)
    return [-rate, -k_inact_S * S * e]

def ode_turnover_inact(y, t, vmax, km, r_inact):
    """Mechanism-based: inactivation proportional to turnover rate.
    r_inact = fraction of turnovers that inactivate the enzyme.
    de/dt = -r_inact * (rate/ET)"""
    S, e = max(y[0], 0), max(y[1], 0)
    rate = vmax * e * S / (km + S + 1e-12)
    return [-rate, -r_inact * rate / ET_mM]

def ode_time_plus_S_inact(y, t, vmax, km, k0, k_S):
    """Combined: de/dt = -(k0 + k_S*S)*e — both spontaneous and [S]-driven"""
    S, e = max(y[0], 0), max(y[1], 0)
    rate = vmax * e * S / (km + S + 1e-12)
    return [-rate, -(k0 + k_S * S) * e]

def ode_si_time_inact(y, t, vmax, km, ki, k_inact):
    """Substrate inhibition + time-dependent inactivation"""
    S, e = max(y[0], 0), max(y[1], 0)
    rate = vmax * e * S / (km + S + S**2/ki + 1e-12)
    return [-rate, -k_inact * e]

def ode_cosubstrate_limited(y, t, vmax, km, k_akg):
    """Co-substrate (αKG) depletion model.
    y = [S, A] where A = [αKG] remaining.
    Each turnover consumes one αKG.
    rate = Vmax * S * A / ((Km_S + S) * (Km_αKG + A))
    Simplified: rate limited by min(S-kinetics, A-kinetics)"""
    S, A = max(y[0], 0), max(y[1], 0)
    rate = vmax * S * A / ((km + S) * (k_akg + A) + 1e-12)
    return [-rate, -rate]  # both S and αKG consumed 1:1


# ── Fitting machinery ─────────────────────────────────────────

def solve_2state(params, ode_func, times_list, S0_list, y0_extra=1.0):
    preds = []
    for times, S0 in zip(times_list, S0_list):
        try:
            sol = odeint(ode_func, [S0, y0_extra], times,
                         args=tuple(params), mxstep=5000)
            preds.append(np.maximum(sol[:, 0], 0))
        except Exception:
            preds.append(np.full_like(times, 1e10))
    return preds

def residuals(log_p, ode_func, times_list, S_obs_list, S0_list, y0_extra=1.0):
    params = np.exp(log_p)
    preds = solve_2state(params, ode_func, times_list, S0_list, y0_extra)
    r = []
    for obs_arr, pred_arr in zip(S_obs_list, preds):
        for obs, pred in zip(obs_arr, pred_arr):
            if obs <= FLOOR:
                r.append(max(0, pred - FLOOR) * 10)
            else:
                r.append(np.log(obs) - np.log(max(pred, FLOOR)))
    return np.array(r)

def fit(name, ode_func, param_names, bounds_log, y0_extra=1.0, n_starts=40):
    n_data = sum(len(s) for s in S_obs_list)
    n_p = len(param_names)
    lb, ub = np.array(bounds_log[0]), np.array(bounds_log[1])
    best_cost, best_res = np.inf, None
    for trial in range(n_starts):
        p0 = (lb+ub)/2 if trial == 0 else lb + np.random.rand(n_p)*(ub-lb)
        try:
            res = least_squares(residuals, p0,
                                args=(ode_func, times_list, S_obs_list,
                                      S0_list, y0_extra),
                                bounds=(lb, ub), method='trf', max_nfev=10000)
            if res.cost < best_cost:
                best_cost, best_res = res.cost, res
        except Exception:
            continue
    if best_res is None:
        return None
    params = np.exp(best_res.x)
    ss = np.sum(best_res.fun**2)
    s2 = ss / n_data
    loglik = -0.5*n_data*(1+np.log(2*np.pi*s2))
    for so in S_obs_list:
        for o in so:
            if o > FLOOR: loglik -= np.log(o)
    aic = -2*loglik + 2*n_p
    bic = -2*loglik + n_p*np.log(n_data)
    try:
        J = best_res.jac
        se = np.sqrt(np.diag(np.linalg.inv(J.T@J)*s2)) * params
    except Exception:
        se = np.full(n_p, np.nan)
    preds = solve_2state(params, ode_func, times_list, S0_list, y0_extra)
    ss_abs = sum(np.sum((o-p)**2) for o,p in zip(S_obs_list, preds))
    return {'name': name, 'params': params, 'param_names': param_names,
            'se': se, 'aic': aic, 'bic': bic, 'sigma': np.sqrt(s2),
            'ss_abs': ss_abs, 'n_p': n_p, 'ode_func': ode_func,
            'preds': preds, 'y0_extra': y0_extra}


# ── Run fits ──────────────────────────────────────────────────
np.random.seed(42)

results = {}

print("=" * 72)
print("Inactivation model comparison (Km constrained ≤ 5 mM)")
print("=" * 72)

# 1. Time-dependent inactivation (de/dt = -k*e)
print("1. Time-dependent inactivation")
r = fit("Time inact", ode_time_inact, ["Vmax", "Km", "k_inact"],
        (np.log([0.01, 0.05, 0.005]), np.log([2, 5, 2])))
if r: results["time"] = r

# 2. [S]-dependent inactivation (de/dt = -k_S*S*e)
print("2. [S]-dependent inactivation")
r = fit("[S]-dep inact", ode_S_inact, ["Vmax", "Km", "k_inact_S"],
        (np.log([0.01, 0.05, 0.001]), np.log([2, 5, 5])))
if r: results["S_dep"] = r

# 3. Mechanism-based (de/dt ∝ turnover rate)
print("3. Mechanism-based (turnover-dep) inactivation")
r = fit("Turnover inact", ode_turnover_inact, ["Vmax", "Km", "r_inact"],
        (np.log([0.01, 0.05, 0.001]), np.log([2, 5, 2])))
if r: results["turnover"] = r

# 4. Combined: time + [S]-dependent
print("4. Combined (time + [S]-dep) inactivation")
r = fit("Time+[S] inact", ode_time_plus_S_inact, ["Vmax", "Km", "k0", "k_S"],
        (np.log([0.01, 0.05, 0.001, 0.0001]),
         np.log([2, 5, 2, 5])))
if r: results["combined"] = r

# 5. SI + time inactivation
print("5. Substrate inhibition + time inactivation")
r = fit("SI + time inact", ode_si_time_inact, ["Vmax", "Km", "Ki_s", "k_inact"],
        (np.log([0.01, 0.05, 0.5, 0.005]),
         np.log([2, 5, 500, 2])))
if r: results["SI_time"] = r

# 6. αKG depletion model (y0_extra = initial [αKG] = 5 mM)
print("6. αKG co-substrate depletion")
r = fit("αKG depletion", ode_cosubstrate_limited, ["Vmax", "Km_S", "Km_αKG"],
        (np.log([0.01, 0.05, 0.01]), np.log([2, 5, 20])),
        y0_extra=5.0)  # 5 mM αKG initially
if r: results["akg"] = r


# ── Results ───────────────────────────────────────────────────
print("\n" + "=" * 72)
print("MODEL COMPARISON")
print("=" * 72)
print(f"{'Model':<24s} {'AIC':>8s} {'BIC':>8s} {'ΔAIC':>7s} {'σ%':>5s} {'SS_abs':>8s}")
print("-" * 60)

skeys = sorted(results.keys(), key=lambda k: results[k]['aic'])
best_aic = results[skeys[0]]['aic']

for k in skeys:
    r = results[k]
    print(f"{r['name']:<24s} {r['aic']:>8.1f} {r['bic']:>8.1f} "
          f"{r['aic']-best_aic:>7.1f} {r['sigma']*100:>4.0f}% {r['ss_abs']:>8.3f}")

print("\n" + "=" * 72)
print("PARAMETER ESTIMATES")
print("=" * 72)

for k in skeys:
    r = results[k]
    pdict = dict(zip(r['param_names'], r['params']))
    print(f"\n  {r['name']}  (ΔAIC={r['aic']-best_aic:.1f})")
    for pn, pv, ps in zip(r['param_names'], r['params'], r['se']):
        se_str = f"± {ps:.4f}" if not np.isnan(ps) else ""
        print(f"    {pn:<12s} = {pv:.4f} {se_str}")
    vmax = pdict.get('Vmax', pdict.get('Vmax', 0))
    km = pdict.get('Km', pdict.get('Km_S', 0))
    kcat = vmax / ET_mM
    print(f"    kcat         = {kcat:.1f} min⁻¹ ({kcat/60:.2f} s⁻¹)")
    if 'k_inact' in pdict:
        hl = np.log(2) / pdict['k_inact']
        total_turnovers = kcat * hl / np.log(2)
        print(f"    t½           = {hl:.1f} min")
        print(f"    total t.o.   ≈ {total_turnovers:.0f} turnovers before 50% inact")
    if 'k_inact_S' in pdict:
        k = pdict['k_inact_S']
        for sref in [0.5, 1, 2.5, 5, 10]:
            print(f"    t½ @{sref}mM     = {np.log(2)/(k*sref):.1f} min")
    if 'r_inact' in pdict:
        print(f"    partition    = {1/pdict['r_inact']:.0f} (turnovers/inactivation)")
    if 'k0' in pdict and 'k_S' in pdict:
        k0, kS = pdict['k0'], pdict['k_S']
        print(f"    t½ @S→0      = {np.log(2)/k0:.1f} min (spontaneous)")
        for sref in [1, 5, 10]:
            print(f"    t½ @{sref}mM     = {np.log(2)/(k0+kS*sref):.1f} min")
    if 'Ki_s' in pdict:
        ki = pdict['Ki_s']
        s_opt = np.sqrt(km * ki)
        print(f"    Ki_s         = {ki:.2f} mM")
        print(f"    S_opt        = {s_opt:.2f} mM")
    if 'Km_αKG' in pdict:
        print(f"    Km_αKG       = {pdict['Km_αKG']:.2f} mM")


# ── Plot ──────────────────────────────────────────────────────
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Show top 3 models, each with all 6 curves
top3 = skeys[:3]
fig, axes = plt.subplots(3, 6, figsize=(20, 12), sharex='col')

for row, key in enumerate(top3):
    r = results[key]
    for i in range(6):
        ax = axes[row, i]
        t_fine = np.linspace(0, times_list[i][-1], 400)
        sol = odeint(r['ode_func'], [S0_list[i], r['y0_extra']], t_fine,
                     args=tuple(r['params']), mxstep=5000)
        ax.plot(t_fine, np.maximum(sol[:, 0], 0), '-', color=colors[i], lw=2)
        ax.plot(times_list[i], S_obs_list[i], 'o', color=colors[i], ms=8, zorder=5)

        # Enzyme activity (right axis)
        if sol.shape[1] > 1:
            ax2 = ax.twinx()
            e_pct = np.maximum(sol[:, 1], 0)
            if r['y0_extra'] != 1.0:
                e_pct = e_pct / r['y0_extra']  # normalize for αKG model
            ax2.fill_between(t_fine, e_pct*100, alpha=0.08, color='red')
            ax2.set_ylim(-5, 105)
            if i == 5:
                ax2.set_ylabel("Active %", fontsize=7, color='red')
            ax2.tick_params(labelsize=6, colors='red')

        # Residual annotations
        for j, (tj, oj) in enumerate(zip(times_list[i], S_obs_list[i])):
            pj = r['preds'][i][j]
            if oj > FLOOR:
                pct = (oj - pj)/oj*100
                c = 'red' if abs(pct) > 15 else 'gray'
                ax.annotate(f"{pct:+.0f}%", (tj, oj), fontsize=6,
                            xytext=(3, 6), textcoords='offset points', color=c)

        ax.set_ylim(bottom=-0.1)
        if row == 0:
            ax.set_title(labels[i], fontsize=10, fontweight='bold')
        if row == 2:
            ax.set_xlabel("Time (min)", fontsize=8)
        if i == 0:
            ax.set_ylabel(f"{r['name']}\n[S] (mM)", fontsize=8)

plt.suptitle("Top 3 inactivation models — per-curve fits with enzyme activity (red shading)",
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = "fit_results.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out}")
plt.close()

# Summary figure
best = results[skeys[0]]
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

# v0 vs [S]
ax = axes2[0]
v0 = [-(S[1]-S[0])/(t[1]-t[0]) for t, S in zip(times_list, S_obs_list)]
ax.plot(S0_list, v0, 'ro', ms=10, zorder=5, label='Measured v₀')
S_r = np.linspace(0.01, 12, 300)
for k in skeys[:3]:
    r = results[k]
    p = dict(zip(r['param_names'], r['params']))
    vm = p.get('Vmax', 0)
    km_v = p.get('Km', p.get('Km_S', 0.5))
    ki_v = p.get('Ki_s', 1e10)
    v = vm * S_r / (km_v + S_r + S_r**2/ki_v)
    ax.plot(S_r, v, lw=2, label=f"{r['name']} (t=0)")
ax.set_xlabel("[S] (mM)")
ax.set_ylabel("v₀ (mM/min)")
ax.set_title("Initial rate (note: measured v₀ at 5-10mM\nare 10min averages, not true v₀)")
ax.legend(fontsize=7)

# Model comparison
ax = axes2[1]
names = [results[k]['name'] for k in skeys]
daic = [results[k]['aic']-best_aic for k in skeys]
cols = ['green' if d==0 else 'steelblue' for d in daic]
ax.barh(range(len(names)), daic, color=cols)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("ΔAIC")
ax.set_title("Model Comparison")
ax.invert_yaxis()

# Enzyme lifetime vs [S] for top models
ax = axes2[2]
S_hl = np.linspace(0.1, 12, 200)
for k in skeys[:3]:
    r = results[k]
    p = dict(zip(r['param_names'], r['params']))
    if 'k_inact' in p:
        hl = np.full_like(S_hl, np.log(2)/p['k_inact'])
    elif 'k_inact_S' in p:
        hl = np.log(2) / (p['k_inact_S'] * S_hl)
    elif 'r_inact' in p:
        vm = p['Vmax']
        km_v = p.get('Km', 0.5)
        rate_per_E = vm * S_hl / ((km_v + S_hl) * ET_mM)
        hl = np.log(2) / (p['r_inact'] * rate_per_E)
    elif 'k0' in p:
        hl = np.log(2) / (p['k0'] + p['k_S'] * S_hl)
    elif 'Km_αKG' in p:
        continue
    else:
        continue
    ax.plot(S_hl, hl, lw=2, label=r['name'])
ax.set_xlabel("[S] (mM)")
ax.set_ylabel("Enzyme half-life (min)")
ax.set_title("Enzyme stability vs [Substrate]")
ax.legend(fontsize=8)
ax.set_ylim(0, 30)

plt.tight_layout()
out2 = "fit_summary.png"
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f"Saved: {out2}")
plt.close()
