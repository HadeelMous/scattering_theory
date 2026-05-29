"""
beregn.py
=========
Generel beregning af enkelt-partikel transmission
for vilkaarlige molekyler med B-felt sweep.

Brug:
    python beregn.py                     <- H2, ingen B-felt
    python beregn.py --mol LiH           <- skift molekyle
    python beregn.py --mol H2 --Bmax 2.0 <- med B-felt sweep
    python beregn.py --mol H2O --V 0.8 --beta -1.5
"""

import numpy as np
import pickle
import argparse
import sys

sys.path.append(".")

from s_matrix_1e import S_matrix_1e, integral_provider
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform
)


# =======================================================================
# Molekyledatabase
# =======================================================================

MOLECULES = {

    "H2": {
        "geometry" : "H 0.0 0.0 0.0; H 0.0 0.0 0.7414",
        "basis_set": "sto-3g",
        "cas"      : (2, 2),
        "beskr"    : "Molekylaer brint, STO-3G",
    },

    "H2_ccpvdz": {
        "geometry" : "H 0.0 0.0 0.0; H 0.0 0.0 0.7414",
        "basis_set": "cc-pvdz",
        "cas"      : (2, 10),
        "beskr"    : "Molekylaer brint, cc-pVDZ",
    },

    "LiH": {
        "geometry" : "Li 0.0 0.0 0.0; H 0.0 0.0 1.595",
        "basis_set": "sto-3g",
        "cas"      : (2, 3),
        "beskr"    : "Lithiumhydrid, STO-3G",
    },

    "H2O": {
        "geometry" : (
            "O  0.000  0.000  0.117; "
            "H  0.000  0.757 -0.469; "
            "H  0.000 -0.757 -0.469"
        ),
        "basis_set": "sto-3g",
        "cas"      : (2, 4),
        "beskr"    : "Vand, STO-3G",
    },

    "NH3": {
        "geometry" : (
            "N  0.000  0.000  0.116; "
            "H  0.000  0.939 -0.271; "
            "H  0.813 -0.470 -0.271; "
            "H -0.813 -0.470 -0.271"
        ),
        "basis_set": "sto-3g",
        "cas"      : (2, 4),
        "beskr"    : "Ammoniak, STO-3G",
    },

}


# =======================================================================
# Beregn transmission for ét B-punkt
# =======================================================================

def beregn_et_punkt(
    geometry, basis_set, cas,
    B_vec, V, site_energy, beta, p_vals
):
    """
    Beregner T(p) for ét B-felt punkt.
    Returnerer dict med resultater.
    """
    mo, h_kin, h_nuc_ao, g_eri, \
        t_LB, t_BB, h_nuc, pyscf_mol = \
        integral_provider(geometry, basis_set, B_vec)

    WF = WaveFunctionUCC(
        cas=cas,
        mo_coeffs=mo,
        integral_generator=pyscf_mol,
        excitations="sd"
    )
    WF._h_mo = one_electron_integral_transform(
        mo, h_kin + h_nuc_ao + t_BB + t_LB
    )

    SMobj   = S_matrix_1e(wf=WF, h_nuc=h_nuc)
    SMobj.V = V

    E0   = SMobj.eigval_N[0]
    ENp1 = SMobj.eigval_Np1

    print(f"  E_0^(eta)   = {E0:.6f} Ha")
    print(f"  E_0^(eta+1) = {ENp1[0]:.6f} Ha")
    print(f"  1e resonans = {ENp1[0]-E0:.6f} Ha")

    T_vals = []
    E_vals = []

    for p in p_vals:
        try:
            S, E_tot = SMobj.S_matrix(
                num_leads=2,
                site_energy=site_energy,
                beta=beta,
                p_in=p
            )
            T_vals.append(float(abs(S[0, 1])**2))
            E_vals.append(float(E_tot - E0))
        except Exception:
            T_vals.append(float("nan"))
            E_vals.append(float("nan"))

    return {
        "B"         : B_vec.tolist(),
        "E_0_N"     : float(E0),
        "eigval_Np1": [float(e) for e in ENp1],
        "p_vals"    : p_vals.tolist(),
        "E_vals"    : E_vals,
        "T_vals"    : T_vals,
    }


# =======================================================================
# Hoved: sweep over B-felt
# =======================================================================

def beregn_molekyle(
    mol_navn,
    B_x_vals, B_y_vals, B_z_vals,
    V=1.0, site_energy=1.0, beta=-1.0,
    N_p=200, gem_fil=None
):
    if mol_navn not in MOLECULES:
        raise ValueError(
            f"'{mol_navn}' ikke fundet. "
            f"Valgmuligheder: {list(MOLECULES.keys())}"
        )

    mol       = MOLECULES[mol_navn]
    geometry  = mol["geometry"]
    basis_set = mol["basis_set"]
    cas       = mol["cas"]

    if gem_fil is None:
        gem_fil = f"{mol_navn}_resultater.pkl"

    p_vals = np.linspace(0.05, np.pi - 0.05, N_p)

    print(f"\n{'='*60}")
    print(f"Molekyle : {mol_navn}  ({mol['beskr']})")
    print(f"Basis    : {basis_set}, CAS: {cas}")
    print(f"V={V}, beta={beta}, site_energy={site_energy}")
    print(f"Gem til  : {gem_fil}")
    print(f"{'='*60}\n")

    resultater = []
    total = len(B_x_vals) * len(B_y_vals) * len(B_z_vals)
    tael  = 0

    for By in B_y_vals:
        for Bx in B_x_vals:
            for Bz in B_z_vals:
                tael += 1
                B_vec = np.array([float(Bx), float(By), float(Bz)])
                print(f"[{tael}/{total}] B = {B_vec}")

                try:
                    res = beregn_et_punkt(
                        geometry, basis_set, cas,
                        B_vec, V, site_energy, beta, p_vals
                    )
                    res["mol_navn"]    = mol_navn
                    res["basis_set"]   = basis_set
                    res["cas"]         = cas
                    res["V"]           = V
                    res["beta"]        = beta
                    res["site_energy"] = site_energy
                    resultater.append(res)

                except Exception as e:
                    print(f"  FEJL: {e}")
                    resultater.append({
                        "mol_navn": mol_navn,
                        "B"       : B_vec.tolist(),
                        "fejl"    : str(e),
                    })

                # Gem løbende efter hvert punkt
                with open(gem_fil, "wb") as f:
                    pickle.dump(resultater, f)
                print(f"  Gemt ({len(resultater)} punkter)")

    print(f"\nFaerdig. Resultat: '{gem_fil}'")
    return resultater


# =======================================================================
# Kommandolinje
# =======================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Beregn enkelt-partikel transmission"
    )
    parser.add_argument(
        "--mol", default="H2",
        help=f"Molekyle ({list(MOLECULES.keys())})"
    )
    parser.add_argument("--V",           type=float, default=1.0)
    parser.add_argument("--beta",        type=float, default=-1.0)
    parser.add_argument("--site_energy", type=float, default=1.0)
    parser.add_argument("--N_p",         type=int,   default=200)
    parser.add_argument("--Bmax",        type=float, default=0.0)
    parser.add_argument("--Bstep",       type=float, default=0.5)
    parser.add_argument("--gem",         default=None)
    args = parser.parse_args()

    if args.Bmax > 0:
        B_x = np.arange(0, args.Bmax + args.Bstep/2, args.Bstep)
        B_y = np.arange(0, args.Bmax + args.Bstep/2, args.Bstep)
    else:
        B_x = [0.0]
        B_y = [0.0]
    B_z = [0.0]

    beregn_molekyle(
        mol_navn    = args.mol,
        B_x_vals    = B_x,
        B_y_vals    = B_y,
        B_z_vals    = B_z,
        V           = args.V,
        beta        = args.beta,
        site_energy = args.site_energy,
        N_p         = args.N_p,
        gem_fil     = args.gem,
    )
