"""
plot_resultater.py
==================
Plotter transmissionsresultater fra pickle-filer.
Laver figurer svarende til artiklen.

Brug:
    python plot_resultater.py H2_V_sweep.pkl
    python plot_resultater.py H2_V_sweep.pkl --zoom 5.175 5.25
    python plot_resultater.py H2_V_sweep.pkl --log
    python plot_resultater.py H2_V_sweep.pkl --gem figur.png
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os


# =======================================================================
# Indlæs pickle
# =======================================================================

def load_pickle(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"'{filename}' findes ikke.")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    data = [r for r in data if "fejl" not in r]
    print(f"Indlæst {len(data)} punkter fra '{filename}'")
    return data


# =======================================================================
# Plot svarende til artiklen
# =======================================================================

def plot_artikel_stil(
    data,
    zoom_min   = None,
    zoom_max   = None,
    log_skala  = False,
    gem_som    = None,
    farver     = None,
):
    """
    To-panel figur som i artiklen:
    - Øverste: zoom på resonans
    - Nedre: fuldt spektrum

    Parametre
    ----------
    data      : liste fra load_pickle()
    zoom_min  : venstre grænse for zoom (eV)
    zoom_max  : højre grænse for zoom (eV)
    log_skala : brug logaritmisk y-akse
    """

    if farver is None:
        farver = ['#3274A1', '#E1812C', '#3A923A', '#C03D3E', '#9372B2']

    mol      = data[0].get("mol_navn", "?")
    E_res_eV = data[0].get("E_res_eV", None)

    fig, (ax_top, ax_bund) = plt.subplots(
        2, 1, figsize=(10, 8)
    )
    fig.patch.set_facecolor('white')

    for i, r in enumerate(data):
        E_vals = np.array(r["E_vals"])
        T_vals = np.array(r["T_vals"])
        V      = r["V"]
        farve  = farver[i % len(farver)]
        lbl    = f"$V_n = {V:.1f}$"

        ax_top.plot(E_vals, T_vals, color=farve, lw=1.5, label=lbl)
        ax_bund.plot(E_vals, T_vals, color=farve, lw=1.5, label=lbl)

    # Resonanslinje
    if E_res_eV is not None:
        for ax in [ax_top, ax_bund]:
            ax.axvline(x=E_res_eV, color='k', ls='--',
                       lw=1, alpha=0.6,
                       label=f'$E_0^{{(2)}}+E_{{\\rm kin}}=E_0^{{(3)}}$')

    # Øverste panel: zoom
    if zoom_min is None or zoom_max is None:
        if E_res_eV is not None:
            zoom_min = E_res_eV - 0.05
            zoom_max = E_res_eV + 0.05
        else:
            all_E    = np.concatenate([r["E_vals"] for r in data])
            zoom_min = float(np.nanmin(all_E))
            zoom_max = float(np.nanmin(all_E)) + 0.1

    ax_top.set_xlim(zoom_min, zoom_max)
    ax_top.set_ylim(0, 0.55)
    ax_top.set_ylabel('Transmission Magnitude', fontsize=12)
    ax_top.tick_params(labelsize=11)
    ax_top.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['top'].set_visible(False)
    ax_top.grid(False)

    # Nedre panel: fuldt spektrum
    all_E = np.concatenate([r["E_vals"] for r in data])
    ax_bund.set_xlim(float(np.nanmin(all_E)),
                     float(np.nanmax(all_E)))
    all_E_min = float(np.nanmin(all_E))
    xlabel = r'$E_\mathrm{in} = p^2/2m$ [eV]' if all_E_min >= -0.1 else 'Incoming Energy [eV]'
    ax_bund.set_xlabel(xlabel, fontsize=12)
    ax_bund.set_ylabel('Transmission Magnitude', fontsize=12)
    ax_bund.tick_params(labelsize=11)
    ax_bund.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_bund.spines['right'].set_visible(False)
    ax_bund.spines['top'].set_visible(False)
    ax_bund.grid(False)

    if log_skala:
        for ax in [ax_top, ax_bund]:
            ax.set_yscale('log')
            ax.set_ylim(1e-11, 1)

    fig.suptitle(f'{mol}: Enkelt-elektron transmission', fontsize=13)
    plt.tight_layout()

    if gem_som:
        plt.savefig(gem_som, dpi=150, bbox_inches='tight')
        print(f"Figur gemt: {gem_som}")

    plt.show()


# =======================================================================
# Plot 4-panel svarende til første artikel-figur
# =======================================================================

def plot_fire_panel(filer_og_V, gem_som=None, log_skala=True):
    """
    Laver 2x2 figur med fire forskellige V-værdier.
    Svarende til den første artikel-figur.

    filer_og_V : liste af (filnavn, V_filter) tuples
                 eller én fil med fire V-værdier
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    farve = '#E07B54'   # orange som i artiklen

    for ax_idx, (filnavn, V_val) in enumerate(filer_og_V):
        if ax_idx >= 4:
            break

        data = load_pickle(filnavn)
        # Find datapunkt med denne V-værdi
        pts  = [r for r in data if abs(r["V"] - V_val) < 0.01]
        if not pts:
            print(f"V={V_val} ikke fundet i '{filnavn}'")
            continue

        r      = pts[0]
        E_vals = np.array(r["E_vals"])
        T_vals = np.array(r["T_vals"])
        E_res  = r.get("E_res_eV", None)
        ax     = axes[ax_idx]

        ax.plot(E_vals, T_vals, color=farve, lw=1.5)

        if E_res is not None:
            # To stiplede linjer som i artiklen (grundtilstand + exciteret)
            ENp1 = r.get("eigval_Np1", [])
            E0   = r.get("E_0_N", 0)
            for e in ENp1[:2]:
                E_res_i = (e - E0 - r.get("site_energy", 1.0)) * 27.2114
                if 0 < E_res_i < max(E_vals):
                    ax.axvline(x=E_res_i, color='k',
                               ls='-.', lw=1, alpha=0.7)

        if log_skala:
            ax.set_yscale('log')
            ax.set_ylim(1e-11, 1)

        ax.set_xlim(0, max(E_vals) if len(E_vals) > 0 else 90)
        ax.text(0.75, 0.92, f'$V_n = {V_val:.0f}$',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white',
                          alpha=0.8))
        ax.set_xlabel('Incoming Energy [eV]', fontsize=10)
        ax.set_ylabel('Transmission Magnitude', fontsize=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.tight_layout()

    if gem_som:
        plt.savefig(gem_som, dpi=150, bbox_inches='tight')
        print(f"Figur gemt: {gem_som}")

    plt.show()


# =======================================================================
# Vis indhold
# =======================================================================

def vis_indhold(filnavn):
    data = load_pickle(filnavn)
    print(f"\nIndhold af '{filnavn}':")
    print(f"  Antal punkter : {len(data)}")
    if data:
        r = data[0]
        print(f"  mol_navn  : {r.get('mol_navn','?')}")
        print(f"  V-værdier : {[r['V'] for r in data]}")
        print(f"  E_res_eV  : {r.get('E_res_eV','?'):.3f} eV")
        print(f"  E_0_N     : {r.get('E_0_N','?'):.6f} Ha")
        T = r.get('T_vals', [])
        E = r.get('E_vals', [])
        print(f"  E_vals    : {min(E):.2f} – {max(E):.2f} eV "
              f"({len(E)} punkter)")
        print(f"  T_vals    : max={max(T):.4f}, min={min(T):.2e}")


# =======================================================================
# Kommandolinje
# =======================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plot transmissionsresultater"
    )
    parser.add_argument("filer", nargs="+",
                        help="Pickle-fil(er)")
    parser.add_argument("--zoom", nargs=2, type=float, default=None,
                        metavar=("E_MIN", "E_MAX"),
                        help="Zoom-grænser i eV, fx --zoom 5.175 5.25")
    parser.add_argument("--log",  action="store_true",
                        help="Logaritmisk y-akse")
    parser.add_argument("--gem",  default=None,
                        help="Gem figur som PNG")
    parser.add_argument("--info", action="store_true",
                        help="Vis kun indhold")
    parser.add_argument("--fire_panel", action="store_true",
                        help="Lav 2x2 figur (kræver --V_vals)")
    parser.add_argument("--V_vals", nargs="+", type=float,
                        default=None,
                        help="V-værdier til fire-panel figur")
    args = parser.parse_args()

    if args.info:
        for fil in args.filer:
            vis_indhold(fil)

    elif args.fire_panel and args.V_vals:
        # Byg liste af (fil, V) par
        fil = args.filer[0]
        par = [(fil, V) for V in args.V_vals[:4]]
        plot_fire_panel(par, gem_som=args.gem, log_skala=args.log)

    else:
        for fil in args.filer:
            try:
                data = load_pickle(fil)
                plot_artikel_stil(
                    data,
                    zoom_min  = args.zoom[0] if args.zoom else None,
                    zoom_max  = args.zoom[1] if args.zoom else None,
                    log_skala = args.log,
                    gem_som   = args.gem,
                )
            except FileNotFoundError as e:
                print(e)
