"""
beregn.py
=========
Generel beregning af enkelt-partikel transmission
for vilkaarlige molekyler.

Kan sweepe over:
  - B-felt (Bx, By, Bz)
  - Koblingsstyrke V

Brug:
    python beregn.py --mol H2
    python beregn.py --mol H2 --V_vals 2.0 2.1 2.2 2.3
    python beregn.py --mol H2 --Bmax 2.0
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

Ha_to_eV = 27.2114


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

    "H2_631g": {
        "geometry" : "H 0.0 0.0 0.0; H 0.0 0.0 0.7414",
        "basis_set": "6-31g",
        "cas"      : (2, 4),
        "beskr"    : "Molekylaer brint, 6-31G",
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

}


# =======================================================================
# Beregn transmission for ét sæt parametre
# =======================================================================

def beregn_et_punkt(
    SMobj,
    V, site_energy, beta,
    p_vals, num_leads=2, asymmetric=False
):
    """
    Beregner T(E_kin) for ét (SMobj, V) punkt.
    Energi returneres i eV.

    asymmetric : hvis True bruges V_L = sqrt(2)*V_R (SI C.2) for at
                 opnaa nul-reflektion med 3 leads. V er da V_R.
    """
    SMobj.V = V

    if asymmetric and num_leads >= 2:
        V_leads = [np.sqrt(2) * V] + [V] * (num_leads - 1)
    else:
        V_leads = None   # bruger self.V for alle leads

    T_vals = []
    E_vals = []   # kinetisk energi i eV

    for p in p_vals:
        try:
            S, _ = SMobj.S_matrix(
                num_leads=num_leads,
                site_energy=site_energy,
                beta=beta,
                p_in=p,
                V_leads=V_leads,
            )
            # Kinetisk energi i lead: E_kin = alpha + 2*beta*cos(p)
            E_kin = (site_energy + 2*beta*np.cos(p)) * Ha_to_eV
            T_vals.append(float(abs(S[0, 1])**2))
            E_vals.append(float(E_kin))
        except Exception:
            T_vals.append(float("nan"))
            E_vals.append(float("nan"))

    return np.array(E_vals), np.array(T_vals)


# =======================================================================
# Byg SMobj én gang og sweep over V
# =======================================================================

def beregn_V_sweep(
    mol_navn,
    V_vals,
    B_vec        = np.array([0., 0., 0.]),
    site_energy  = 1.0,
    beta         = -1.0,
    N_p          = 500,
    num_leads    = 2,
    asymmetric   = False,
    a_dot        = None,
    gem_fil      = None,
):
    """
    Beregner transmission for en liste af V-værdier
    ved fast B-felt. Bygger kun WF én gang.

    a_dot : kvanteprikkafstand i Angstrom (SI afsnit A).
            Hvis angivet, beregnes beta = -hbar^2/(2m*a^2) og
            site_energy = -2*beta, saa E_kin = E_free = p^2/2m
            (fri partikel kinetisk energi). p sweeper fra 0 til
            p_max svarende til E_free = (E_res + 5 eV).
    """
    if mol_navn not in MOLECULES:
        raise ValueError(f"'{mol_navn}' ikke fundet. "
                         f"Valg: {list(MOLECULES.keys())}")

    mol       = MOLECULES[mol_navn]
    geometry  = mol["geometry"]
    basis_set = mol["basis_set"]
    cas       = mol["cas"]

    if a_dot is not None:
        a_bohr   = a_dot / 0.529177      # Angstrom -> Bohr
        beta     = -1.0 / (2.0 * a_bohr**2)   # hbar=1, m_e=1 (a.u.)
        site_energy = -2.0 * beta        # band bund = alpha + 2*beta = 0

    if gem_fil is None:
        gem_fil = f"results/{mol_navn}_V_sweep.pkl"

    print(f"\n{'='*60}")
    print(f"Molekyle : {mol_navn}  ({mol['beskr']})")
    print(f"Basis    : {basis_set}, CAS: {cas}")
    print(f"B        : {B_vec}")
    print(f"V-sweep  : {V_vals}")
    print(f"beta={beta:.4f} Ha, site_energy={site_energy:.4f} Ha")
    if a_dot is not None:
        print(f"a_dot={a_dot} AA  (fri-partikel tilstand)")
    print(f"Gem til  : {gem_fil}")
    print(f"{'='*60}\n")

    # Byg WF og SMobj én gang
    print("Bygger bølgefunktion...")
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

    SMobj = S_matrix_1e(wf=WF, h_nuc=h_nuc, pyscf_mol=pyscf_mol, mo_coeffs=mo)

    E0   = SMobj.eigval_N[0]
    ENp1 = SMobj.eigval_Np1[0]
    # Resonansenergi = E_0(N+1) - E_0(N) i fri-partikel energi (eV)
    E_res_free_eV = (ENp1 - E0) * Ha_to_eV

    if a_dot is not None:
        # Fri-partikel: sweep p fra 0 til p_max saa E_free daekker
        # op til E_res + 5 eV med lidt margin
        E_max_Ha = (ENp1 - E0) + 5.0 / Ha_to_eV
        p_max    = np.sqrt(2.0 * E_max_Ha / abs(beta))   # E_free = |beta|*p^2
        p_vals   = np.linspace(0.001, min(p_max, np.pi - 0.001), N_p)
        print(f"Resonans (fri)  = {E_res_free_eV:.3f} eV")
        print(f"p-sweep: [0, {p_max:.4f}] rad  "
              f"(E_free_max = {E_max_Ha*Ha_to_eV:.1f} eV)\n")
    else:
        E_res_eV = (ENp1 - E0 - site_energy) * Ha_to_eV
        p_vals   = np.linspace(0.01, np.pi - 0.01, N_p)
        print(f"E_0^(eta)   = {E0:.6f} Ha")
        print(f"E_0^(eta+1) = {ENp1:.6f} Ha")
        print(f"1e resonans = {E_res_eV:.3f} eV\n")

    resultater = []

    for V in V_vals:
        print(f"V = {V:.2f}...")
        E_vals, T_vals = beregn_et_punkt(
            SMobj, V, site_energy, beta, p_vals, num_leads, asymmetric
        )
        resultater.append({
            "mol_navn"   : mol_navn,
            "basis_set"  : basis_set,
            "cas"        : cas,
            "B"          : B_vec.tolist(),
            "V"          : float(V),
            "beta"       : beta,
            "site_energy": site_energy,
            "E_0_N"      : float(E0),
            "eigval_Np1" : [float(e) for e in SMobj.eigval_Np1],
            "E_res_eV"   : float(E_res_free_eV),
            "E_vals"     : E_vals.tolist(),   # eV
            "T_vals"     : T_vals.tolist(),
        })

        with open(gem_fil, "wb") as f:
            pickle.dump(resultater, f)

    print(f"\nFaerdig. Gemt: '{gem_fil}'")
    return resultater


# =======================================================================
# Byg SMobj én gang og sweep over B-felt
# =======================================================================

def beregn_B_sweep(
    mol_navn,
    B_x_vals, B_y_vals, B_z_vals,
    V            = 1.0,
    site_energy  = 1.0,
    beta         = -1.0,
    N_p          = 300,
    num_leads    = 2,
    gem_fil      = None,
):
    """
    Sweeper over B-felt. Bygger WF for hvert B-punkt.
    """
    if mol_navn not in MOLECULES:
        raise ValueError(f"'{mol_navn}' ikke fundet.")

    mol       = MOLECULES[mol_navn]
    geometry  = mol["geometry"]
    basis_set = mol["basis_set"]
    cas       = mol["cas"]

    if gem_fil is None:
        gem_fil = f"results/{mol_navn}_B_sweep.pkl"

    p_vals = np.linspace(0.01, np.pi - 0.01, N_p)

    print(f"\n{'='*60}")
    print(f"Molekyle : {mol_navn}  ({mol['beskr']})")
    print(f"V={V}, beta={beta}, site_energy={site_energy}")
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

                    SMobj = S_matrix_1e(wf=WF, h_nuc=h_nuc)

                    E0    = SMobj.eigval_N[0]
                    ENp1  = SMobj.eigval_Np1[0]
                    E_res = (ENp1 - E0 - site_energy) * Ha_to_eV

                    print(f"  1e resonans = {E_res:.3f} eV")

                    E_vals, T_vals = beregn_et_punkt(
                        SMobj, V, site_energy, beta, p_vals, num_leads
                    )

                    resultater.append({
                        "mol_navn"   : mol_navn,
                        "B"          : B_vec.tolist(),
                        "V"          : float(V),
                        "beta"       : beta,
                        "site_energy": site_energy,
                        "E_0_N"      : float(E0),
                        "eigval_Np1" : [float(e) for e in SMobj.eigval_Np1],
                        "E_res_eV"   : float(E_res),
                        "E_vals"     : E_vals.tolist(),
                        "T_vals"     : T_vals.tolist(),
                    })

                except Exception as e:
                    print(f"  FEJL: {e}")
                    resultater.append({
                        "mol_navn": mol_navn,
                        "B"       : B_vec.tolist(),
                        "fejl"    : str(e),
                    })

                with open(gem_fil, "wb") as f:
                    pickle.dump(resultater, f)
                print(f"  Gemt ({len(resultater)} punkter)")

    print(f"\nFaerdig. Gemt: '{gem_fil}'")
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
        help=f"Molekyle: {list(MOLECULES.keys())}"
    )
    parser.add_argument(
        "--V_vals", nargs="+", type=float, default=[1.0],
        help="V-vaerdier fx --V_vals 2.0 2.1 2.2 2.3"
    )
    parser.add_argument("--beta",        type=float, default=-1.0)
    parser.add_argument("--site_energy", type=float, default=1.0)
    parser.add_argument("--N_p",         type=int,   default=500)
    parser.add_argument("--num_leads",   type=int,   default=2)
    parser.add_argument("--asymmetric",  action="store_true",
                        help="V_L = sqrt(2)*V_R (SI C.2, nul-reflektion med 3 leads)")
    parser.add_argument("--a_dot",       type=float, default=None,
                        help="Kvanteprikkafstand i Angstrom (SI afsnit A). "
                             "Saetter beta = -hbar^2/(2m*a^2) og site_energy = -2*beta, "
                             "saa x-aksen bliver fri-partikel kinetisk energi p^2/2m.")
    parser.add_argument("--Bmax",        type=float, default=0.0)
    parser.add_argument("--Bstep",       type=float, default=0.5)
    parser.add_argument("--gem",         default=None)
    args = parser.parse_args()

    if args.Bmax > 0:
        # B-felt sweep
        B_x = np.arange(0, args.Bmax + args.Bstep/2, args.Bstep)
        B_y = np.arange(0, args.Bmax + args.Bstep/2, args.Bstep)
        B_z = [0.0]
        beregn_B_sweep(
            mol_navn    = args.mol,
            B_x_vals    = B_x,
            B_y_vals    = B_y,
            B_z_vals    = B_z,
            V           = args.V_vals[0],
            beta        = args.beta,
            site_energy = args.site_energy,
            N_p         = args.N_p,
            gem_fil     = args.gem,
        )
    else:
        # V-sweep ved B=0
        beregn_V_sweep(
            mol_navn    = args.mol,
            V_vals      = args.V_vals,
            B_vec       = np.array([0., 0., 0.]),
            beta        = args.beta,
            site_energy = args.site_energy,
            N_p         = args.N_p,
            num_leads   = args.num_leads,
            asymmetric  = args.asymmetric,
            a_dot       = args.a_dot,
            gem_fil     = args.gem,
        )
