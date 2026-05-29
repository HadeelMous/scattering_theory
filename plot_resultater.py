"""
plot_resultater.py
==================
Plotter transmissionsresultater fra pickle-filer
gemt af beregn.py.

Brug:
    python plot_resultater.py H2_resultater.pkl
    python plot_resultater.py H2_resultater.pkl --gem H2_figur.png
    python plot_resultater.py H2_resultater.pkl LiH_resultater.pkl
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
        raise FileNotFoundError(f"Filen '{filename}' findes ikke.")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    # Filtrer fejlede punkter fra
    data = [r for r in data if "fejl" not in r]
    print(f"Indlæst {len(data)} gyldige punkter fra '{filename}'")
    return data


# =======================================================================
# Plot: transmission for én fil
# =======================================================================

def plot_en_fil(data, ax_top, ax_bund, farve, label_prefix=""):
    """
    Plotter T(E) for alle B-feltvaerdier i én fil.
    Øverste panel: zoom på resonans.
    Nedre panel: fuldt spektrum.
    """
    if not data:
        return

    # Find resonansenergi fra første punkt
    E0   = data[0]["E_0_N"]
    ENp1 = data[0]["eigval_Np1"][0]
    E_res = ENp1 - E0

    cmap   = plt.cm.viridis
    n      = len(data)

    for i, r in enumerate(data):
        E_vals = np.array(r["E_vals"])
        T_vals = np.array(r["T_vals"])
        B      = r["B"]
        farve_i = cmap(i / max(1, n - 1))

        lbl = f"{label_prefix}B=({B[0]:.1f},{B[1]:.1f},{B[2]:.1f})"

        ax_top.plot(E_vals, T_vals, color=farve_i,
                    lw=1.5, label=lbl)
        ax_bund.plot(E_vals, T_vals, color=farve_i,
                     lw=1.5, label=lbl)

    # Resonanslinje
    for ax in [ax_top, ax_bund]:
        ax.axvline(x=E_res, color='k', ls='--',
                   lw=1, alpha=0.5, label=f'Resonans {E_res:.3f} Ha')


# =======================================================================
# Hoved plotfunktion
# =======================================================================

def plot_resultater(filnavne, gem_som=None):
    """
    Plotter resultater fra en eller flere pickle-filer.

    Parametre
    ----------
    filnavne : list   Liste af pickle-filnavne
    gem_som  : str    Gem figur som PNG (None = vis kun)
    """

    fig, axes = plt.subplots(
        len(filnavne), 2,
        figsize=(14, 5 * len(filnavne))
    )
    fig.patch.set_facecolor('#fafafa')

    # Gør axes til 2D liste selv for én fil
    if len(filnavne) == 1:
        axes = [axes]

    farver = ['#185FA5', '#0F6E56', '#D85A30', '#534AB7']

    for fil_idx, filnavn in enumerate(filnavne):
        try:
            data = load_pickle(filnavn)
        except FileNotFoundError as e:
            print(e)
            continue

        ax_top  = axes[fil_idx][0]
        ax_bund = axes[fil_idx][1]
        mol     = data[0].get("mol_navn", filnavn)
        farve   = farver[fil_idx % len(farver)]

        plot_en_fil(data, ax_top, ax_bund, farve, label_prefix=f"{mol} ")

        # Resonansenergi
        E0   = data[0]["E_0_N"]
        ENp1 = data[0]["eigval_Np1"][0]
        E_res = ENp1 - E0

        # Øverste panel: zoom på resonans
        ax_top.set_xlim(E_res - 0.1, E_res + 0.1)
        ax_top.set_ylim(0, 1.05)
        ax_top.set_title(f'{mol}: Zoom på resonans '
                         f'({E_res:.3f} Ha)', fontsize=11)
        ax_top.set_xlabel('Indkommende energi (Ha)', fontsize=10)
        ax_top.set_ylabel('Transmission $T$', fontsize=10)
        ax_top.grid(True, alpha=0.3, ls='--')
        ax_top.set_facecolor('#f8f8f8')
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['top'].set_visible(False)
        if len(data) <= 8:
            ax_top.legend(fontsize=7, loc='upper right',
                          ncol=1, framealpha=0.8)

        # Nedre panel: fuldt spektrum
        all_E = np.concatenate([r["E_vals"] for r in data])
        E_min = np.nanmin(all_E)
        E_max = np.nanmax(all_E)
        ax_bund.set_xlim(E_min, E_max)
        ax_bund.set_ylim(0, 1.05)
        ax_bund.set_title(f'{mol}: Fuldt spektrum', fontsize=11)
        ax_bund.set_xlabel('Indkommende energi (Ha)', fontsize=10)
        ax_bund.set_ylabel('Transmission $T$', fontsize=10)
        ax_bund.grid(True, alpha=0.3, ls='--')
        ax_bund.set_facecolor('#f8f8f8')
        ax_bund.spines['right'].set_visible(False)
        ax_bund.spines['top'].set_visible(False)

        # Print info
        print(f"\n{mol}:")
        print(f"  E_0^(eta)   = {E0:.6f} Ha")
        print(f"  E_0^(eta+1) = {ENp1:.6f} Ha")
        print(f"  1e resonans = {E_res:.6f} Ha")
        print(f"  V           = {data[0].get('V', '?')}")
        print(f"  beta        = {data[0].get('beta', '?')}")

    fig.suptitle('Enkelt-elektron transmission', fontsize=13, y=1.01)
    plt.tight_layout()

    if gem_som:
        plt.savefig(gem_som, dpi=150,
                    bbox_inches='tight', facecolor='#fafafa')
        print(f"\nFigur gemt: {gem_som}")

    plt.show()


# =======================================================================
# Vis indhold af pickle uden at plotte
# =======================================================================

def vis_indhold(filnavn):
    data = load_pickle(filnavn)
    print(f"\nIndhold af '{filnavn}':")
    print(f"  Antal punkter : {len(data)}")
    if data:
        print(f"  Nøgler        : {list(data[0].keys())}")
        r = data[0]
        print(f"\nFørste punkt:")
        print(f"  mol_navn  : {r.get('mol_navn','?')}")
        print(f"  B         : {r.get('B','?')}")
        print(f"  E_0_N     : {r.get('E_0_N','?'):.6f} Ha")
        print(f"  eigval_Np1: {r.get('eigval_Np1',['?'])[:3]} ...")
        T = r.get('T_vals', [])
        print(f"  T_vals    : {len(T)} punkter, "
              f"max={max(T):.3f}, min={min(T):.3f}")


# =======================================================================
# Kommandolinje
# =======================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plot transmissionsresultater"
    )
    parser.add_argument(
        "filer", nargs="+",
        help="Pickle-fil(er) med resultater"
    )
    parser.add_argument(
        "--gem", default=None,
        help="Gem figur som PNG"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Vis kun indhold, ingen plot"
    )
    args = parser.parse_args()

    if args.info:
        for fil in args.filer:
            vis_indhold(fil)
    else:
        plot_resultater(args.filer, gem_som=args.gem)
