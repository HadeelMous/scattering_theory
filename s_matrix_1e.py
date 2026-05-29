"""
s_matrix_1e.py
==============
Enkelt-partikel S-matrix klasse og integral_provider.
Denne fil importeres af beregn.py og plot_resultater.py.
"""

import numpy as np
import pyscf
from pyscf import gto, scf

from slowquant.unitary_coupled_cluster.ci_spaces import get_indexing
from slowquant.unitary_coupled_cluster.operators import hamiltonian_0i_0a, a_op
from slowquant.unitary_coupled_cluster.fermionic_operator import FermionicOperator
from slowquant.molecularintegrals.integralfunctions import (
    one_electron_integral_transform
)


# =======================================================================
# Hjælpefunktion: lav fermionisk operator
# =======================================================================

def annihilation(p: int, spin: str, dagger: bool) -> FermionicOperator:
    return a_op(p, spin, dagger=dagger)


# =======================================================================
# Integraler fra PySCF
# =======================================================================

def integral_provider(geometry: str, basis_set: str, B: np.ndarray):
    """
    Beregner alle molekylære integraler inkl. B-felt bidrag.

    Parametre
    ----------
    geometry  : str        Molekylegeometri i PySCF format
    basis_set : str        Basisset fx 'sto-3g', 'cc-pvdz'
    B         : np.ndarray B-felt vektor [Bx, By, Bz] i Tesla

    Returnerer
    ----------
    mo          : MO-koefficienter
    h_int1e_kin : kinetisk energiintegral (AO basis)
    h_int1e_nuc : kernetiltrækning (AO basis)
    g_eri       : to-elektron integraler (AO basis)
    t_LB        : lineært B-felt bidrag
    t_BB        : kvadratisk B-felt bidrag
    h_nuc       : kernerepulsionsenergi
    mol         : PySCF mol objekt
    """

    mol = pyscf.gto.Mole()
    mol.atom = geometry
    mol.basis = basis_set
    mol.build()

    rhf = scf.RHF(mol)
    rhf.verbose = 0
    rhf.kernel()

    mo    = rhf.mo_coeff
    h_nuc = rhf.energy_nuc()

    h_int1e_kin = mol.intor("int1e_kin")
    h_int1e_nuc = mol.intor("int1e_nuc")
    g_eri       = mol.intor("int2e")

    t_LB = np.zeros((len(mo), len(mo)), dtype=np.complex128)
    t_BB = np.zeros((len(mo), len(mo)), dtype=np.complex128)

    B_x, B_y, B_z = B[0], B[1], B[2]

    with mol.with_common_origin((0, 0, 0)):
        L_x, L_y, L_z = mol.intor("int1e_cg_irxp", comp=3)
        r_xx, r_xy, r_xz, _, r_yy, r_yz, _, _, r_zz = mol.intor("int1e_rr")
        r_dot_r = mol.intor("int1e_r2")

    for i in range(len(t_LB)):
        for j in range(len(t_LB)):
            t_LB[i, j] = 1j * 0.5 * (
                B_x * L_x[i, j]
              + B_y * L_y[i, j]
              + B_z * L_z[i, j]
            )
            t_BB[i, j] = 0.125 * (
                B @ B * r_dot_r[i, j]
              - B_x**2 * r_xx[i, j]
              - B_y**2 * r_yy[i, j]
              - B_z**2 * r_zz[i, j]
              - 2 * B_x * B_y * r_xy[i, j]
              - 2 * B_x * B_z * r_xz[i, j]
              - 2 * B_y * B_z * r_yz[i, j]
            )

    return mo, h_int1e_kin, h_int1e_nuc, g_eri, t_LB, t_BB, h_nuc, mol


# =======================================================================
# S_matrix_1e klasse
# =======================================================================

class S_matrix_1e:
    """
    Beregner enkelt-partikel S-matrix for et molekyle
    koblet til tight-binding leads.

    Workflow:
        SMobj = S_matrix_1e(wf=WF, h_nuc=h_nuc)
        SMobj.V = 1.0
        S, E_tot = SMobj.S_matrix(num_leads=2, site_energy=1.0,
                                   beta=-1.0, p_in=1.0)
    """

    def __init__(self, wf, h_nuc):
        self.wf = wf

        # --- Byg determinantrum for eta og eta+1 ---
        ci_N = get_indexing(
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
            self.wf.num_active_elec_alpha,
            self.wf.num_active_elec_beta
        )
        self.idx2det_N = ci_N.idx2det
        self.det2idx_N = dict(ci_N.det2idx)

        ci_Np1 = get_indexing(
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs,
            self.wf.num_active_elec_alpha + 1,
            self.wf.num_active_elec_beta
        )
        self.idx2det_Np1 = ci_Np1.idx2det
        self.det2idx_Np1 = dict(ci_Np1.det2idx)

        # Unified indexing: eta + eta+1
        self.idx2det_unified = np.concatenate(
            [self.idx2det_N, self.idx2det_Np1]
        )
        self.det2idx_unified = {}
        for det, idx in self.det2idx_N.items():
            self.det2idx_unified[det] = idx
        for det, idx in self.det2idx_Np1.items():
            self.det2idx_unified[det] = idx + len(self.det2idx_N)

        # --- Diagonalisér Hamiltonian ---
        H = hamiltonian_0i_0a(
            self.wf.h_mo, self.wf.g_mo,
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs
        ).get_folded_operator(
            self.wf.num_inactive_orbs,
            self.wf.num_active_orbs,
            self.wf.num_virtual_orbs
        )

        H_mat_N   = self.build_operator_matrix(
            op=H, idx2det=self.idx2det_N, det2idx=self.det2idx_N
        )
        H_mat_Np1 = self.build_operator_matrix(
            op=H, idx2det=self.idx2det_Np1, det2idx=self.det2idx_Np1
        )

        # Eta sektor
        eigval_N, eigvec_N_ = np.linalg.eig(H_mat_N)
        sort_idx = np.argsort(eigval_N)
        eigval_N   = eigval_N[sort_idx]
        eigvec_N_  = eigvec_N_[:, sort_idx]

        N_unified = len(self.idx2det_unified)
        self.eigvec_N = np.zeros(
            (N_unified, N_unified), dtype=np.complex128
        )
        for i in range(len(eigvec_N_)):
            self.eigvec_N[:, i] = \
                self.expand_statevector_to_unified_indexing(
                    eigvec_N_[:, i], self.idx2det_N,
                    self.det2idx_unified
                )

        # Eta+1 sektor
        eigval_Np1, eigvec_Np1_ = np.linalg.eig(H_mat_Np1)
        sort_idx    = np.argsort(eigval_Np1)
        eigval_Np1  = eigval_Np1[sort_idx]
        eigvec_Np1_ = eigvec_Np1_[:, sort_idx]

        self.eigvec_Np1 = np.zeros(
            (N_unified, N_unified), dtype=np.complex128
        )
        for i in range(len(eigvec_Np1_)):
            self.eigvec_Np1[:, i] = \
                self.expand_statevector_to_unified_indexing(
                    eigvec_Np1_[:, i], self.idx2det_Np1,
                    self.det2idx_unified
                )

        # Tilfoej kernerepulsion og tjek imaginaerdel
        self.eigval_N   = [np.real(e + h_nuc) for e in eigval_N]
        self.eigval_Np1 = [np.real(e + h_nuc) for e in eigval_Np1]

        for en in self.eigval_N:
            if abs(np.imag(en)) > 1e-6:
                raise ValueError(f"Imaginaer energi i eta: {en}")
        for en in self.eigval_Np1:
            if abs(np.imag(en)) > 1e-6:
                raise ValueError(f"Imaginaer energi i eta+1: {en}")

    # -------------------------------------------------------------------
    # Property: V (koblingsstyrke)
    # -------------------------------------------------------------------

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, value):
        self._V = value

    # -------------------------------------------------------------------
    # Byg matrixrepræsentation af fermionisk operator
    # -------------------------------------------------------------------

    def build_operator_matrix(
        self,
        op: FermionicOperator,
        idx2det,
        det2idx: dict,
        unsafe_mode: bool = False,
    ) -> np.ndarray:

        num_dets = len(idx2det)
        op_mat   = np.zeros((num_dets, num_dets), dtype=np.complex128)

        parity_check = {0: 0}
        num = 0
        for i in range(2 * self.wf.num_active_orbs - 1, -1, -1):
            num += 2**i
            parity_check[2 * self.wf.num_active_orbs - i] = num

        for i in range(num_dets):
            det_ = idx2det[i]

            for op_key, coefficient in op.operators.items():
                det           = det_
                phase_changes = 0

                for orb_idx, is_dagger in reversed(op_key):
                    nth_bit = (
                        det >> (2 * self.wf.num_active_orbs - 1 - orb_idx)
                    ) & 1

                    if nth_bit == 0 and is_dagger:
                        det ^= 2**(2*self.wf.num_active_orbs - 1 - orb_idx)
                        phase_changes += (
                            det & parity_check[orb_idx]
                        ).bit_count()

                    elif nth_bit == 1 and is_dagger:
                        break

                    elif nth_bit == 0 and not is_dagger:
                        break

                    elif nth_bit == 1 and not is_dagger:
                        det ^= 2**(2*self.wf.num_active_orbs - 1 - orb_idx)
                        phase_changes += (
                            det & parity_check[orb_idx]
                        ).bit_count()

                else:
                    val = coefficient * (-1)**phase_changes
                    if det not in det2idx and unsafe_mode:
                        continue
                    if abs(val) > 1e-14:
                        op_mat[det2idx[det], i] += val

        return op_mat

    # -------------------------------------------------------------------
    # Udvid tilstandsvektor til unified indexing
    # -------------------------------------------------------------------

    def expand_statevector_to_unified_indexing(
        self, vec, idx2det, det2idx_unified
    ):
        new_vec = np.zeros(len(det2idx_unified), dtype=np.complex128)
        for i, coeff in enumerate(vec):
            det          = idx2det[i]
            new_idx      = det2idx_unified[det]
            new_vec[new_idx] = coeff
        return new_vec

    # -------------------------------------------------------------------
    # Vis determinantudvikling
    # -------------------------------------------------------------------

    def get_determinant_expansion(self, statevector, idx2det):
        coeffs = []
        dets   = []
        for i, coeff in enumerate(statevector):
            if abs(coeff) > 1e-10:
                dets.append(
                    format(idx2det[i],
                           f"0{2*self.wf.num_active_orbs}b")
                )
                coeffs.append(str(coeff))
        return coeffs, dets

    # -------------------------------------------------------------------
    # Beregn S-matrix for givet impuls
    # -------------------------------------------------------------------

    def S_matrix(
        self,
        num_leads:   int,
        site_energy: float,
        beta:        float,
        p_in:        float,
    ):
        """
        Beregner enkelt-partikel S-matrix.

        Parametre
        ----------
        num_leads   : antal leads
        site_energy : on-site energi i lead (alpha)
        beta        : hop-amplitude i lead
        p_in        : indkommende impuls

        Returnerer
        ----------
        S     : S-matrix (num_leads x num_leads)
        E_tot : total energi = site_energy + 2*beta*cos(p) + E_0^(eta)
        """

        num_orbs = (
            self.wf.num_inactive_orbs
          + self.wf.num_active_orbs
          + self.wf.num_virtual_orbs
        )
        spins = ["alpha", "beta"]
        nNp1  = len(self.eigval_Np1)

        E_in = site_energy + 2 * beta * np.cos(p_in)

        # --- Byg B-matricen ---
        B_mat = np.zeros((nNp1, num_leads), dtype=np.complex128)

        for g in range(nNp1):
            B_element = 0.0 + 0j
            for spin in spins:
                for p in range(num_orbs):
                    apd = annihilation(p, spin, True).get_folded_operator(
                        self.wf.num_inactive_orbs,
                        self.wf.num_active_orbs,
                        self.wf.num_virtual_orbs
                    )
                    apd_mat = self.build_operator_matrix(
                        op=apd,
                        idx2det=self.idx2det_unified,
                        det2idx=self.det2idx_unified,
                        unsafe_mode=True
                    )
                    B_element += (
                        self.eigvec_Np1[:, g]
                        @ apd_mat
                        @ self.eigvec_N[:, 0]
                    )
            for n in range(num_leads):
                B_mat[g, n] = B_element

        # --- D^{-1} ---
        D_inv = np.diag([
            1.0 / (E_in + self.eigval_N[0] - self.eigval_Np1[i])
            for i in range(nNp1)
        ])

        # --- Q matricer ---
        ph_p = np.cos(p_in) + 1j * np.sin(p_in)
        ph_m = np.cos(p_in) - 1j * np.sin(p_in)

        Q1 = (beta * np.eye(num_leads)
              - self.V**2 * ph_p * B_mat.conj().T @ D_inv @ B_mat)
        Q2 = (beta * np.eye(num_leads)
              - self.V**2 * ph_m * B_mat.conj().T @ D_inv @ B_mat)

        S = -np.linalg.inv(Q1) @ Q2

        # --- Unitaritets tjek ---
        if not np.allclose(S @ S.conj().T, np.eye(num_leads)):
            raise ValueError("S-matrix er ikke unitær!")

        return S, E_in + self.eigval_N[0]
