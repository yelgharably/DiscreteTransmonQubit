# %% [markdown]
# # **Modeling Transmon Qubits as Tight-Binding Hamiltonians**
# ## QSEG610 - Engineering The Quantum Revolution Final Project
# ## Youssef El Gharably -- Fall 2025
# ### Instructors: Dr Garcia-Frias, Dr Vishal Saxena

# %% [markdown]
# Superconducting Qubits -especially Transmon qubits- are the workhorses of modern quantum processors. They are built from the simplest possible superconducting circuit, a Josephson Junction (JJ) shunted by a large capacitance. The configuration is able to make a nonlinear quantum oscillator whose lowest two energy levels form an effective qubit. 
# 
# The Transmon Qubit sits at the intersection of:
# - Quantum Mechanics; it provides the Hamiltonian and its Quantization rules:
# \begin{equation*}
#     \hat{H} = 4E_C (\hat{n}-n_g)^2 - E_J \cos(\hat{\phi}),
# \end{equation*}
# - Tight-Binding Model; it provides a mean of reducing the continous Schrodinger equation** to a discrete tight-binding eigenproblem solvable by matrix methods,
# - Electronics; it connects the circuit parameters (capacitance, inductance, critical current) to the Quantum Energies $E_C$ and $E_J$ by desigining control/readout hardware that couples via voltage and flux lines.
# 
# ### The Physical Picture of the Transmon Qubit as a Quantum Circuit:
# A JJ behaves like a nonlinear, nondissipative inductor whose current-phase relation is $I(\phi) = I_C \sin(\phi)$. In circuit quantization, the Lagrangian of a node flux $\Phi=\frac{\Phi_0}{2\pi}\phi$ is:
# \begin{equation*}
#     L=\frac{1}{2} C_\Sigma \dot{\Phi}^2 + E_J \cos(\phi).
# \end{equation*}
# The conjugate charge $q=C_\Sigma \dot{\Phi}=2en$ leads to the Hamiltonian:
# \begin{equation*}
#     \hat{H}= 4E_C (\hat{n}-n_g)^2 - E_J \cos(\hat{\phi}), \quad [\hat{\phi},\hat{n}]=i\hbar,
# \end{equation*}
# where $E_C=e^2/2C_\Sigma$ represents the charging energy or the electrostatic term, $E_J=\Phi_0 I_C/2\pi$ represents the Josephson Energy for the nonlinear inductive term, and $n_g$ represents an offset charge controlled by a gate electrode. Physically, this Hamiltonian describes a quantum harmonic osciallator of a particle of mass $1/8E_C$ moving in a periodic potential $U(\phi)=-E_J\cos(\phi)$.
# 
# ### The Current-Phase Relationship (CPR):
# At the junction level, the current-phase relationship (CPR) reflects the microscopic materials' properties:
# \begin{equation*}
#     I(\phi) = \sum_{m\geq 1} I_m \sin(m\phi).
# \end{equation*}
# - For a tunnel junction (Superconductor-Insulator-Superconductor/SIS): $I_1$ dominates so $I(\phi)$ is nearly sinusodial.
# - For a metallic constriction (Superconductor-Constriction-Superconductor/SCS): higher harmonics tend to appear in the series. 
# Integrating $I(\phi)$ over all $\phi$ gives the Josephson Potential:
# \begin{equation*}
#     U(\phi) = -\sum_m E_m \cos(m\phi), \quad E_m = \frac{\Phi_0 I_m}{2\pi m}.
# \end{equation*}
# So, the potential is able to influence the transmon's anharmonicity and noise sensitivity.
# 
# ### Discretizing the Continous Hamiltonian into a Tight-Binding Hamiltonian:
# We pick the charge basis $\ket{n}$ (Cooper-pair number). Using the identity $e^{\pm i\hat{\phi}}\ket{n}=\ket{n\mp 1}$, we have:
# \begin{equation*}
#     \cos(m\hat{\phi}) = \frac{1}{2} \sum_n \left(\ket{n+m}\bra{n}+\ket{n}\bra{n+m} \right),
# \end{equation*}
# and the Hamiltonian becomes a 1-D tight-binding lattice in $n$:
# \begin{equation*}
#     H_{nm'} = 4E_C (n-n_g)^2 \delta_{n,n'} - \frac{1}{2} E_m (\delta_{n',n+m} + \delta_{n',n-m}).
# \end{equation*}
# Each harmonic $m$ in the CPR adds a hopping range $m$ to the lattice. Diagnolizing this Hermitian matrix gives the eigenvalues $E_i$ and eigenvectors $U_{ni}$. From $E_i$, we can compute key metrics:
# - Transition Frequency: $\omega_{01}=E_1-E_0$ for qubit operating frequency;
# - Anharmonicity: $\alpha(E_2 - E_1)-(E_1-E_0)$ for gate selectivity and leakage;
# - Charge Dispersion: variation between $\omega_{01}$ and $n_g$ for charge-noise sensitivity.
# 
# ### Open System Dynamics and Decoherence:
# In realistic environments, the transmon interacts with external electromagnetic fields. Mathematically, we can describe this as an open quantum system using the Lindblad master equation:
# \begin{equation*}
#     \frac{\partial \rho}{\partial t} = -i \hbar[\hat{H},\hat{\rho}] + \sum_k \left(L_k \hat{\rho} L_l^\dagger - \frac{1}{2} \{L_k^\dagger L_k, \hat{\rho} \} \right),
# \end{equation*}
# with operators $L_k$ encoding:
# - Energy Relaxation $(T_1)$: $L_1 = \sqrt{1/T_1} \ket{0}\bra{1}$;
# - Pure dephasing $(T_\varphi)$: $L_\varphi = \sqrt{1/T_\varphi} \hat{\sigma}_z$.
# 
# For slow, correlated noise, we could also replace the Lindblad master equation with a stochastic Hamiltonian evolution:
# \begin{equation*}
#     i\hbar \frac{\partial}{\partial t} \ket{\psi} = \left(\hat{H}_0 + \zeta(t) \hat{V} \right)\ket{\psi},
# \end{equation*}
# for the noise potential contribution $\hat{V}$ which averages over many noise realizations. This should produce a Gaussian like coherence decay.
# 
# ## **Building the Transmon Qubit computationally:**

# %%
# Imports and Dependencies
import numpy as np
import kwant as kw
from matplotlib import pyplot as plt
from typing import Dict, Tuple, List
from types import SimpleNamespace
hbar = 1.0545718e-34  # J.s
kB = 1.380649e-23    # J/K

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral' # my typical choice of font

from dataclasses import dataclass
import scipy

@dataclass
class EigenSystem:
    evals: np.ndarray          # ascending eigenvalues (in units of E_C)
    evecs: np.ndarray          # columns are eigenvectors in charge basis (size: Nbasis x Nstates)
    n_vals: np.ndarray         # integer charge grid aligned with rows of evecs

def _get_xp():
    try:
        import cupy as xp; _ = xp.linalg.eigh; return xp, True
    except Exception:
        import numpy as xp; return xp, False

def comm(A,B): return A @ B - B @ A
def anticomm(A,B): return A @ B + B @ A

xp, GPU = _get_xp()

def _has(*names):
    return all(n in globals() for n in names)

def _maybe(name):
    return globals().get(name, None)

def to_xp(a, xp_mod):
    """Convert NumPy/CuPy array or Python list to xp_mod array (complex128)."""
    # If it's a CuPy array and xp_mod is cupy: leave as-is
    try:
        import cupy as _cp
        if xp_mod is _cp and isinstance(a, _cp.ndarray):
            return a.astype(_cp.complex128, copy=False)
    except Exception:
        pass
    # If it's NumPy or anything else: just xp.asarray
    return xp_mod.asarray(a, dtype=xp_mod.complex128)

# %%
"""
This cell contains functions mainly responsible for constructing and diagonalizing the Transmon Hamiltonian,
as well as computing relevant observables such as transition frequencies and anharmonicity.
"""
def transmon_H(
        EC: float,
        Em: Dict[int,float],
        n_cutoff: int=30,
        n_g: float=0.0,
        dtype=None
):
    if dtype is None:
        dtype = xp.float64
    dim = 2 * n_cutoff + 1
    ns = xp.arange(-n_cutoff, n_cutoff + 1, dtype=xp.int64)
    H = xp.zeros((dim, dim), dtype=dtype)
    H += xp.diag(4.0 * EC * (ns - n_g) ** 2)

    for m, Em_val in Em.items():
        if m <= 0:
            continue
        hop = -0.5 * float(Em_val)
        i = xp.arange(0, dim - m, dtype=xp.int64)
        H[i, i + m] += hop
        H[i + m, i] += hop
    return H,ns

def diagnolize_H(
        H
) -> Tuple:
    w,v = xp.linalg.eigh(H)
    return w,v

def matrix_elements(
        ns,
        U
):
    n_op = xp.diag(ns.astype(U.dtype)); n_mat = U.conj().T @ n_op @ U
    return n_mat

def observables(
        evals
):
    if evals.shape[0] < 2:
        raise ValueError("Need at least two energy levels to compute observables")
    w01 = float(evals[1] - evals[0])
    w12 = float(evals[2] - evals[1])
    alpha = w12 - w01
    return w01, alpha

# %%
"""
This cell contains functions for estimating asymptotic values of Transmon observables,
as well as functions for convergence testing and parameter sweeps.
"""

def transmon_asymptotics(EJ,EC):
    w01_lit = xp.sqrt(8.0 * EJ * EC) - EC
    alpha_lit = -EC
    n_zpt_lit = (EJ/(32.0 * EC))**0.25
    return w01_lit, alpha_lit, n_zpt_lit

def converge_cutoff(
        EC: float,
        Em,
        n_g: float=0.0,
        cuts=(20,25,30,35,40,45,50)):
    rows = []
    for n_cutoff in cuts:
        H,ns = transmon_H(EC, Em, n_cutoff=n_cutoff, n_g=n_g)
        evals,U = diagnolize_H(H)
        n_mat = matrix_elements(ns, U)
        w01, alpha = observables(evals)
        rows.append((n_cutoff, w01, alpha, abs(n_mat[0,1]), abs(n_mat[1,2])))
    return rows

def sweep_ng(
        EC,
        Em,
        n_cutoff: int=50,
        num = 80
):
    ngs = xp.linspace(0.0, 1.0, num=num, dtype=xp.float64)
    w01s = xp.zeros(num, dtype=xp.float64)
    for k, ng in enumerate(ngs):
        H,ns = transmon_H(EC, Em, n_cutoff=n_cutoff, n_g=ng)
        evals,U = diagnolize_H(H)
        w01, alpha = observables(evals)
        w01s[k] = w01
    w01s = w01s.get() if GPU else w01s
    eps01 = float(xp.max(w01s) - xp.min(w01s))
    return ngs, w01s, eps01

def EJ_flux(EJ_sum,Phi_norm):
    return EJ_sum * xp.cos(xp.pi * Phi_norm)

def sweep_anharmonicity_vs_EJ_EC(
        EC,
        EJ_sum,
        n_cutoff: int=30,
        n_g = 0.0,
        phi_grid=xp.linspace(0,1,40),
        Em_harmonics = None,
        abs=False
):
    alphas = xp.zeros_like(phi_grid); w01s = xp.zeros_like(phi_grid)
    for k, phi in enumerate(phi_grid):
        EJ_phi = EJ_flux(EJ_sum, phi)
        if abs:
            EJ_phi = xp.abs(EJ_phi)
        if Em_harmonics is None:
            Em = {1: EJ_phi}
        else:
            base = EJ_phi * Em_harmonics.get(1,1.0)
            Em = {1: base}
            for m, rel in Em_harmonics.items():
                if m == 1:
                    continue
                Em[m] = base * rel
        H, ns = transmon_H(EC, Em, n_cutoff=n_cutoff, n_g=n_g)
        evals, _ = diagnolize_H(H)
        w01, alpha = observables(evals)
        alphas[k] = alpha; w01s[k] = w01
    alphas = alphas.get() if GPU else alphas
    w01s = w01s.get() if GPU else w01s
    return phi_grid, w01s, alphas

def main():
    EC = 0.5; EJ =20.0; Em = {1: EJ}
    EJ_sum = EJ * 2.0; n_cut =50
    H,ns = transmon_H(EC, Em, n_cutoff=n_cut, n_g=0.0)
    evals,U = diagnolize_H(H)
    n_mat = matrix_elements(ns, U)
    w01, alpha = observables(evals)

    print(f"w01 = {w01:.4f}, alpha = {alpha:.4f}")
    print(f"<0|n|1> = {n_mat[0,1]:.4f}, <1|n|2> = {n_mat[1,2]:.4f}")
    print(f"GPU used: {GPU}")

    w01_est, alpha_est, n_est = transmon_asymptotics(EJ, EC)
    print(f"estimates: w01≈{w01_est:.3f}, alpha≈{alpha_est:.3f}, |n01|≈{n_est:.3f}")
    print(f"from TB : w01={w01:.3f}, alpha={alpha:.3f}, |n01|={abs(n_mat[0,1]):.3f}")

    Em_scs = {1: EJ, 2: 0.15*EJ, 3: 0.02*EJ}
    H, ns = transmon_H(EC, Em_scs, n_cutoff=n_cut, n_g=0.0)
    evals, U = diagnolize_H(H)
    w01_scs, alpha_scs = observables(evals)
    n_mat = matrix_elements(ns, U)
    print(f"With higher harmonics: w01={w01_scs:.4f}, alpha={alpha_scs:.4f}")
    print(f"<0|n|1>={n_mat[0,1]:.4f}, <1|n|2>={n_mat[1,2]:.4f}")

    phis = xp.linspace(0,1,500)
    w01s_flux = []
    for ph in phis:
        Em_flux = {1: EJ_flux(EJ, ph)}
        H, ns = transmon_H(EC, Em_flux, n_cutoff=30, n_g=0.0)
        E, U = diagnolize_H(H)
        w01_flux, _ = observables(E)
        w01s_flux.append(w01_flux)
    phis = phis.get() if GPU else phis
    plt.figure(figsize=(12,6));plt.plot(phis, w01s_flux)
    plt.xlabel(r'Normalized Flux $\Phi/\Phi_0$');plt.ylabel(r'$\omega_{01}$',fontsize=16);plt.title('Flux Dependence of $\omega_{01}$',fontsize=16)
    plt.grid();plt.savefig('flux_dependence_omega01.png', dpi=300);plt.show();plt.close()

    phi, w01_phi, alpha_phi = sweep_anharmonicity_vs_EJ_EC(EC, EJ_sum)

    phi2, w01_phi2, alpha_phi2 = sweep_anharmonicity_vs_EJ_EC(
        EC, EJ_sum,
        Em_harmonics={1:1.0, 2:0.15, 3:0.02},
    )
    phi3, w01_phi3, alpha_phi3 = sweep_anharmonicity_vs_EJ_EC(
        EC, EJ_sum,
        Em_harmonics={1:1.0, 2:0.3, 3:0.05},
    )
    phi4, w01_phi4, alpha_phi4 = sweep_anharmonicity_vs_EJ_EC(
        EC, EJ_sum,
        Em_harmonics={1:1.0,2:0.15, 3:0.05, 4:0.01},
    )
    phi = phi.get() if GPU else phi2
    phi2 = phi2.get() if GPU else phi2
    phi3 = phi3.get() if GPU else phi3
    phi4 = phi4.get() if GPU else phi4

    lineswidth = 2
    plt.figure(figsize=(12,6));plt.plot(phi, alpha_phi,linewidth=lineswidth, label="SIS ($m=1$ only)");plt.plot(phi2, alpha_phi2,linewidth=lineswidth, label="With 2nd harmonic $(E_2/E_1=0.15)$");plt.plot(phi3, alpha_phi3,linewidth=lineswidth, label="With 3rd harmonic $(E_3/E_1=0.05)$")
    plt.plot(phi4, alpha_phi4,linewidth=lineswidth, label="With 4th harmonic $(E_4/E_1=0.01)$")
    plt.plot(phi, -EC * np.ones_like(phi), linewidth=lineswidth, linestyle="--", label=r"Transmon est. $\alpha\approx -E_C$", color="gray")
    plt.xlabel(r"Normalized Flux $\Phi/\Phi_0$");plt.ylabel(r"Anharmonicity $\alpha$");plt.title("Flux Dependence of Transmon Anharmonicity")
    plt.legend();plt.grid();plt.savefig('flux_dependence_anharmonicity.png', dpi=300);plt.show()

    return None

if __name__ == "__main__":
    main()

# %% [markdown]
# The output of the previous cell shows proper behavior of a Transmon qubit where $\omega_{01}\approx \sqrt{8E_J E_C}$ and both $\braket{0|n|1}$ and $\braket{1|n|2}$ are non-zero. We also have the off-diagonal components of the TB Hamiltonian as $-10$, which is the hopping term. 
# Furthermore, the plot showing the flux dependence on $\omega_{01}$ shows a maximum at $\Phi=0$ when $E_J=E_{J,\Sigma}$, so the qubit freuqency is highest. It also shows a minimum at $\Phi=\Phi_0/2$ when $E_J=0$. This is due to the symmetric SQUID limit when the Josephson potential nearly vanishes causing $\omega_{01}$ to collapse to a small value. This shape matches the experimental data acquired by Koch et al (2007): Transmon Frequency vs Flux Bias.
# 
# Looking at the Anharmonity plot, we see some interesting behavior due to the current-phase relationship (CPR):
# \begin{equation*}
#     I(\phi) = I_1 \sin(\phi) + I_2 \sin(2\phi) + I_3 \sin(3\phi) + I_4 \sin(4\phi)
# \end{equation*}
# which would correspond to the potential:
# \begin{equation*}
#     U(\phi) = -E_1 \cos(\phi) - E_2 \cos(2\phi) - E_3 \cos(3\phi) - E_4 \cos(4\phi).
# \end{equation*}
# 
# By expanding into more harmonics, the JJ behaves in a non-sinusodial manner. This is typical for metallic constriction junctions (ScS), graphene-based junctions, and high-transperancy 2D weak links where more harmonics are measurable and significant. The plots validate this physical behavior through a TB hamiltonian representation.
# 
# ## **Markovian and Non-Markovian System Evolution**:
# In order to properly set this up, we need to pick a bias, pre-build the Hamiltonian and eigenbasis, then truncate to the lowest $d$ levels.

# %%
EC, EJ = 0.5, 20.0
Em = {1: EJ}
H, ns = transmon_H(EC, Em, n_cutoff=50, n_g=0.0)
evals, U = diagnolize_H(H)
d = 3
Es = evals[:d]
Ured = U[:,:d]
n_mat = matrix_elements(ns, Ured)

w01, alpha = observables(evals)
print(f"w01 = {w01:.4f}, alpha = {alpha:.4f}")
print(f"<0|n|1> = {n_mat[0,1]:.4f}, <1|n|2> = {n_mat[1,2]:.4f}")

# %% [markdown]
# In the eigenbasis, a charge drive couples via $\hat{n}$:
# \begin{equation*}
#     \hat{H}(t) = \mathrm{diag}(E_0,\dots,E_{d-1})+ A\cos(\omega t)\hat{n}.
# \end{equation*}
# For a Rabi measurement, we set $\omega=\omega_{01}$ and A such that $\Omega_{01}=A|n_{01}|$ gives us the target Rabi rate.

# %%
def H_time(
        t: float,
        A: float,
        omega: float,
        n_mat: np.ndarray,
        Es: np.ndarray
) -> np.ndarray:
    d = Es.shape[0]
    H0 = np.diag(Es)
    H_drive = A * np.cos(omega * t) * n_mat
    H_total = H0 + H_drive
    return H_total

# %% [markdown]
# We then evolve the density operator such that: 
# \begin{equation*}
#     \frac{\partial}{\partial t}\rho = -i\hbar [\hat{H},\hat{\rho}] + \sum_k \mathcal{D}[L_k]\hat{\rho}, \quad \mathcal{D}[L]\hat{\rho} = L\rho L^\dagger - \frac{1}{2} \{L^\dagger L,\hat{\rho} \}.
# \end{equation*}
# We collapse the operators such that:
# - Relaxation $T_1$ between adjacent levels is $L_{10} = \sqrt{\Gamma_{10}} \ket{0}\bra{1}$, and $L_{21}=\sqrt{\Gamma_{21}}\ket{1}\bra{2}$, where $\Gamma$ is the relaxation rate or energy decay rate. We can model the relaxation $T_1$ in a more physical sense using the spectral density of fluctuations at that frequency $S_n(\omega_{ij})$ such that for relaxation:
# \begin{equation*}
#     \Gamma_{ij} = |\braket{i|\hat{n}|j}|^2 S_n(\omega_{ij}), \quad S_n(\omega) = \int_{-\infty}^\infty \braket{\delta n(t)\delta n(0)}e^{i\omega t} \, dt.
# \end{equation*}
# - Dephasing $T_\varphi$ between adjacent levels is similarly represented where $L_\varphi = -\sqrt{\gamma_\varphi}\hat{\sigma}_z = \sqrt{\gamma_\varphi} (\ket{1}\bra{1}-\ket{0}\bra{0})$ for $\gamma_\varphi=1/T_\varphi$. A more accurate physical description of this can also be formed using $S_n(\omega)$ such that:
# \begin{equation*}
#     \gamma_\varphi = |\braket{1|\hat{n}|1} - \braket{0|\hat{n}|0}|^2 S_n(0),
# \end{equation*}
# since zero-frequency noise causes random shifts of the energy splitting.
# 
# We will represent noise in three forms:
# - Ohmic: $S(\omega) = \eta \omega e^{-\omega/\omega_c} \coth\left(\frac{\hbar\omega}{2k_B T} \right)$
# - White Noise: $S(\omega) = S_0$
# - 1/$f$ regularized noise: $S(\omega) = \frac{A}{|\omega|+\omega_{min}}$
# 
# We then compute the Markovian rates from Fermi's golden rule, downward emission and upward absorption:
# \begin{equation*}
#     \Gamma_{i\rightarrow j} = |A_{ij}|^2 S(+\omega_{ij}), \quad \Gamma_{j\rightarrow i} = |A_{ij}|^2 S(-\omega_{ij}), \quad A_{ij} = \braket{i|\hat{A}|j}, \quad \omega_{ij} = E_i - E_j,
# \end{equation*}
# where $\hat{A}=\hat{n}$ represents the charge noise or the flux operator.

# %%
def S_noise(w,wmin=2*xp.pi, eta=1e-6, wc=2*xp.pi*10e9,T=10e-3,S0=1e-12):
    """Spectral density of noise with Ohmic, White, and 1/f components."""
    beta = hbar / (kB*T) if T > 0 else xp.inf
    nB = 1.0/(xp.exp(beta*xp.abs(w))+1e-100)
    S_ohm = eta*abs(w)*np.exp(-abs(w)/wc)*(1+2*nB)
    S_white = S0
    S_1f = xp.where(xp.abs(w)>=wmin, S0*(wmin/abs(w)), S0)
    S_total = S_ohm + S_white + S_1f
    return S_total

def rates_from_S(Es,A_mat,S_func):
    d = len(Es)
    G = xp.zeros((d,d), dtype=A_mat.dtype)
    for i in range(d):
        for j in range(i):
            wij = float(Es[i] - Es[j])
            Sij = S_func(wij)
            gdown = (xp.abs(A_mat[i,j])**2) * Sij
            gup = (xp.abs(A_mat[j,i])**2) * S_func(-wij)
            G[i,j] = gdown
            G[j,i] = gup
    dA = (A_mat[1,1] - A_mat[0,0]).real
    gphi = dA**2 * S_func(0)
    return G, gphi

def collapse_ops(T1=None, Tphi=None, n_mat=None):
    Ls = []
    d = n_mat.shape[0]
    if T1 is not None and n_mat is not None:
        g10 = 1.0/T1
        L10 = xp.zeros((d,d), dtype=n_mat.dtype); L10[0,1] = xp.sqrt(g10); Ls.append(L10)
        if d >= 3:
            g21 = 2.0/T1
            L21 = xp.zeros((d,d), dtype=n_mat.dtype); L21[1,2] = xp.sqrt(g21); Ls.append(L21)
    
    if Tphi is not None and n_mat is not None:
        gphi = 1.0/Tphi
        Lphi = xp.zeros((d,d), dtype=n_mat.dtype)
        for k in range(d):
            Lphi[k,k] = xp.sqrt(gphi)
        Ls.append(Lphi)
    return Ls

# %% [markdown]
# ### And now to implement the Lindblad Master Equation:

# %%
def lindblad_rhs(
        rho,
        H,
        Ls
):
    dr = -1j * comm(H, rho)
    for L in Ls:
        LdL = L.conj().T @ L
        dr += L @ rho @ L.conj().T - 0.5 * anticomm(LdL, rho)
    return dr

def rk4_step(
        rho,
        t,
        dt,
        H_func,
        args,
        Ls):
    H1 = H_func(t, *args); k1 = lindblad_rhs(rho, H1, Ls)
    H2 = H_func(t + 0.5 * dt, *args); k2 = lindblad_rhs(rho + 0.5 * dt * k1, H2, Ls)
    H3 = H_func(t + 0.5 * dt, *args); k3 = lindblad_rhs(rho + 0.5 * dt * k2, H3, Ls)
    H4 = H_func(t + dt, *args); k4 = lindblad_rhs(rho + dt * k3, H4, Ls)
    return rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# %% [markdown]
# ### Ramsey and Rabi:
# We prepare $\ket{+}=(\ket{0}+\ket{1})/\sqrt{2}$ to evolve free by setting $A=0$. Then, we read $\braket{\hat{\sigma}_x} = 2\Re[\rho_{01}]$. For Rabi, we choose $A\neq 0$ and $\omega=\omega_{01}$, and read $p_1=\rho_{11}$.

# %%
def rho_pure(vec):
    return xp.outer(vec, vec.conj())

def run_ramsey(
        Es,
        n_mat,
        T1,
        Tphi,
        tmax,
        dt,
        detuning=0.0
):
    d = len(Es)
    H_func = lambda t, A, w, Es, n:xp.diag(Es) + detuning * xp.array([[1,0,0],[0,-1,0],[0,0,0]])[:d,:d]
    Ls = collapse_ops(T1=T1, Tphi=Tphi, n_mat=n_mat)
    psi_plus = xp.zeros(d,dtype=xp.complex128); psi_plus[0] = 1.0/xp.sqrt(2); psi_plus[1]=1/xp.sqrt(2)
    rho = rho_pure(psi_plus)
    ts, sx = [],[]; t=0.0
    while t <= tmax:
        ts.append(t); sx.append(2.0*xp.real(rho[0,1]))
        rho = rk4_step(rho, t, dt, H_func, (0.0, 0.0, Es, n_mat), Ls); t += dt
    ts = xp.array(ts).get() if GPU else xp.array(ts)
    sx = xp.array(sx).get() if GPU else xp.array(sx)
    return ts, sx

# %%
# Plotting Lindblad Rabi Decay
EC, EJ = 0.5, 20.0
Em = {1: EJ}
n_cut = 10
H, ns = transmon_H(EC, Em, n_cutoff=n_cut, n_g=0.0)
evals, U = diagnolize_H(H)
Es = evals[:3]
Ured = U[:,:3]
n_mat = matrix_elements(ns, Ured)
T1 = 20e-6
Tphi = 30e-6
tmax = 10e-5
dt = 1e-8
ts, sx = run_ramsey(Es, n_mat, T1, Tphi, tmax, dt)
plt.figure(figsize=(12,6)); plt.plot(ts*1e6, sx, label=f"$T_1$={T1*1e6}$\\mu$s, $T_\\phi$={Tphi*1e6}$\\mu$s")
plt.xlabel("Time ($\\mu s$)",fontsize=16); plt.ylabel(r"$\langle \sigma_x \rangle$", fontsize=16); #plt.title("Lindblad Ramsey Decay")
plt.legend(); plt.grid(); plt.savefig('lindblad_ramsey_decay.png', dpi=300); plt.show()

# %% [markdown]
# ### Modeling the Stochastic Hamiltonian (Non-Markovian)
# To capture slow drift, 1/$f$ charge noise, or flux noise, we need to make the parameters a random process and evolve pure states then average at the end. 
# The Hamiltonian on-site term contains $4E_C (n-n_g)^2$, we need to rebuild H each step with an updated $n_g(t)$ value, or we can add the linearized perturbation $\delta H \approx -8E_C(\hat{n}-n_g)\delta n_g$. We do so using the Ornstein-Uhlenbeck with mean 0, correlation time $\tau_c$ and asymptotic standard deviation $\sigma$. The definition of a Ornstein-Uhlenbeck process is as follow:
# 
# >"The Ornstein-Uhlenbeck process is a stochastic process that models the tendency of a variable to revert to a long-term mean, while also experiencing random fluctuations. It is defined by a stochastic differential equation where a drift term pulls the process toward its mean, and a diffusion term introduces random noise." ~ Wikipedia

# %%
def ou_step(x,dt,tau_c,sigma,rng):
    a = xp.exp(-dt/tau_c)
    var = sigma**2 * (1 - a**2)
    return a*x + xp.sqrt(var) * rng.standard_normal()

def ramsey_stochastic_ng(
        EC,
        Em,
        n_cutoff,
        Es,
        U,
        d,
        tmax,
        dt,
        tau_c,
        sigma_ng,
        n_g0,
        n01=None
):
    psi0 = xp.zeros(d,dtype=xp.complex128); psi0[0] = 1.0/xp.sqrt(2); psi0[1]=1/xp.sqrt(2)
    t = 0.0; n_g = n_g0
    ts= []; c01 = []
    psi = psi0.copy()
    rng = xp.random.default_rng()
    while t <= tmax:
        H, ns = transmon_H(EC, Em, n_cutoff=n_cutoff, n_g=n_g)
        Efull, Ufull = diagnolize_H(H)
        Es_t = Efull[:d]; U_t = Ufull[:,:d]
        Ht = xp.diag(Es_t)
        psi = psi - 1j * (Ht @ psi) * dt
        psi = psi / xp.linalg.norm(psi)
        ts.append(t); c01.append(2*xp.real(psi[0].conj() * psi[1]))
        n_g = ou_step(n_g, dt, tau_c, sigma_ng, rng)
        t += dt
    return xp.array(ts).get() if GPU else xp.array(ts), xp.array(c01).get() if GPU else xp.array(c01)

# %% [markdown]
# ## Let's make some plots!

# %%
def average_over_trajectories(run_fn, Ntraj=10):
    outs = [run_fn() for _ in range(Ntraj)]
    ts = outs[0][0]
    sx = xp.mean(xp.array([xp.asarray(y) for (_, y) in outs]), axis=0)
    return ts, sx

plt.figure(figsize=(12,6))
n_cut = 10
sigmas = [2e-4, 5e-4, 1e-3, 2e-3, 1e-2, 0.1]
for s in sigmas:
    ts, sx_avg = average_over_trajectories(
        lambda: ramsey_stochastic_ng(EC, Em, n_cut, Es, U, d,
                                     tmax=50e-6, dt=2e-8,
                                     tau_c=50e-5, sigma_ng=s,
                                     n_g0=0.0),
        Ntraj=10
    )
    sx_avg = sx_avg.get() if GPU else sx_avg
    plt.plot(ts*1e6, sx_avg, label=f"$\sigma_ng={s:.0e}$")
plt.xlabel("time ($\mu s$)"); plt.ylabel(r"$\langle\sigma_x\rangle$"); plt.legend(); #plt.title("Ramsey vs charge-noise amplitude")
plt.yscale('logit'); plt.savefig('ramsey_stochastic_ng.png', dpi=300);
plt.grid(); plt.show(); plt.close()


# %%
def transmon_hamiltonian_charge(n_cut: int, EJ_over_EC: float, n_g: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = 2 * n_cut + 1
    n_vals = np.arange(-n_cut, n_cut + 1, dtype=float)
    diag = 4.0 * (n_vals - n_g) ** 2
    offdiag = -0.5 * float(EJ_over_EC) * np.ones(N - 1, dtype=float)
    return diag, offdiag, n_vals


def diagonalize_tridiag(diag: np.ndarray, offdiag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = diag.size
    if GPU:
        evals, evecs = scipy.linalg.eigh_tridiagonal(diag, offdiag)
    else:
        H = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
        evals, evecs = xp.eigh(H)
        # eigh already returns ascending
    return evals, evecs


def eigensystem_transmon(n_cut: int, EJ_over_EC: float, n_g: float = 0.0) -> EigenSystem:
    diag, offdiag, n_vals = transmon_hamiltonian_charge(n_cut, EJ_over_EC, n_g)
    evals, evecs = diagonalize_tridiag(diag, offdiag)
    return EigenSystem(evals=evals, evecs=evecs, n_vals=n_vals)


def anharmonicity_from_evals(evals: np.ndarray) -> float:
    E0, E1, E2 = evals[0:3]
    return (E2 - E1) - (E1 - E0)


def n_matrix_element(es: EigenSystem, i: int, j: int) -> complex:
    ci = es.evecs[:, i]
    cj = es.evecs[:, j]
    return np.vdot(ci, es.n_vals * cj)

def validate_charge_basis(n_cuts: List[int], EJ_over_EC: float, n_g: float = 0.0) -> Dict[int, Dict[str, float]]:
    results = {}
    for nc in n_cuts:
        es = eigensystem_transmon(nc, EJ_over_EC, n_g)
        E01 = es.evals[1] - es.evals[0]
        alpha = anharmonicity_from_evals(es.evals)
        n01 = abs(n_matrix_element(es, 0, 1))
        n02 = abs(n_matrix_element(es, 0, 2))
        results[nc] = {
            'E01': float(E01),
            'alpha': float(alpha),
            'abs_n01': float(n01),
            'abs_n02': float(n02),
        }
    return results


def plot_charge_basis_convergence(results: Dict[int, Dict[str, float]]) -> None:
    ncs = np.array(sorted(results.keys()))
    E01 = np.array([results[nc]['E01'] for nc in ncs])
    alpha = np.array([results[nc]['alpha'] for nc in ncs])
    n01 = np.array([results[nc]['abs_n01'] for nc in ncs])

    fig, ax = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
    ax[0].plot(ncs, E01, marker='o')
    ax[0].set_ylabel('$E_{01} / E_C$')
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(ncs, alpha, marker='o')
    ax[1].set_ylabel('$\\alpha / E_{C}$')
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(ncs, n01, marker='o')
    ax[2].set_xlabel('$n_{cut}$')
    ax[2].set_ylabel('|$\\langle 0 | n | 1 \\rangle$|')
    ax[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('charge_basis_convergence.png', dpi=300)

def sweep_alpha_vs_ratio(ratios: np.ndarray, n_cut: int = 12, n_g: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays (ratios, E01, alpha) for plotting. n_cut chosen to be conservative."""
    E01s, alphas = [], []
    for r in ratios:
        es = eigensystem_transmon(n_cut, float(r), n_g)
        E01s.append(es.evals[1] - es.evals[0])
        alphas.append(anharmonicity_from_evals(es.evals))
    return ratios, np.array(E01s), np.array(alphas)


def plot_alpha_vs_ratio(ratios: np.ndarray, alphas: np.ndarray) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(ratios, alphas)
    plt.xlabel('$E_J$ / $E_C$')
    plt.ylabel('$\\alpha$ / $E_C$')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('anharmonicity_vs_ratio.png', dpi=300)


def sweep_matrix_elements_vs_ratio(ratios: np.ndarray, n_cut: int = 12, n_g: float = 0.0,
                                   pairs: List[Tuple[int, int]] = [(0,1), (1,2)]) -> Dict[Tuple[int, int], np.ndarray]:
    out: Dict[Tuple[int, int], List[float]] = {p: [] for p in pairs}
    for r in ratios:
        es = eigensystem_transmon(n_cut, float(r), n_g)
        for p in pairs:
            out[p].append(abs(n_matrix_element(es, p[0], p[1])))
    return {p: np.array(vals) for p, vals in out.items()}


def plot_matrix_elements_vs_ratio(ratios: np.ndarray, elem_dict: Dict[Tuple[int, int], np.ndarray]) -> None:
    plt.figure(figsize=(6, 4))
    for (i, j), vals in elem_dict.items():
        plt.plot(ratios, vals, label=f'|$\\langle {i} | n | {j} \\rangle$|')
    plt.xlabel('$E_J / E_C$')
    plt.ylabel('matrix element (dimensionless)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('matrix_elements_vs_ratio.png', dpi=300)

def charge_to_phase_wavefunctions(evecs: np.ndarray, n_vals: np.ndarray, n_states: int = 3, Nphi: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    phi = np.linspace(-np.pi, np.pi, Nphi, endpoint=False)
    psi = np.zeros((n_states, Nphi), dtype=complex)
    for m in range(n_states):
        c = evecs[:, m]
        phase = np.exp(1j * np.outer(n_vals, phi))
        psi[m, :] = (c @ phase) / np.sqrt(2.0 * np.pi)
    return phi, psi


def wavefunction_validity(evecs: np.ndarray, n_vals: np.ndarray, n_states: int = 3, Nphi: int = 2048) -> Dict[str, float]:
    phi, psi = charge_to_phase_wavefunctions(evecs, n_vals, n_states=n_states, Nphi=Nphi)
    dphi = (2.0 * np.pi) / Nphi
    norms = np.real(np.sum(np.abs(psi)**2, axis=1) * dphi)
    norm_err = float(np.max(np.abs(norms - 1.0)))
    overlaps = psi @ (np.conjugate(psi).T) * dphi
    offdiag = overlaps - np.diag(np.diag(overlaps))
    orth_err = float(np.max(np.abs(offdiag)))
    periodic_err = float(np.max(np.abs(psi[:, 0] - psi[:, -1])))

    return {
        'norm_err': norm_err,
        'orth_err': orth_err,
        'periodic_err': periodic_err,
    }


def plot_wavefunctions(phi: np.ndarray, psi: np.ndarray, m_max: int = 3) -> None:
    m_max = min(m_max, psi.shape[0])
    plt.figure(figsize=(7, 4))
    for m in range(m_max):
        plt.plot(phi, np.abs(psi[m])**2, label=f'|$\\psi$_{m}($\\phi$)|$^2$')
    plt.xlabel('$\\phi$ (rad)')
    plt.ylabel('Probability density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('wavefunctions_phase_space.png', dpi=300)


def plot_wavefunctions_with_potential(phi: np.ndarray, psi: np.ndarray, EJ_over_EC: float, scale: float = 0.15) -> None:
    pot = -EJ_over_EC * np.cos(phi)
    pot_scaled = (pot - pot.min())
    pot_scaled /= (pot_scaled.max() + 1e-12)
    pot_scaled *= scale
    plt.figure(figsize=(7, 4))
    for m in range(min(3, psi.shape[0])):
        plt.plot(phi, np.abs(psi[m])**2, label=f'|$\\psi$_{m}($\\phi$)|$^2$')
    plt.plot(phi, pot_scaled, linestyle='--', label='(shifted) $-(E_J/E_C) \\cos \\phi$')
    plt.xlabel('$\\phi$ (rad)')
    plt.ylabel('Probability density (arb.)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('wavefunctions_with_potential.png', dpi=300)

def asymptotic_errors(ratios, E01s, alphas):
    w01_asym = np.sqrt(8.0*ratios) - 1.0
    a_asym   = -np.ones_like(ratios)
    rel_w01  = (E01s - w01_asym) / w01_asym
    rel_a    = (alphas - a_asym) / a_asym
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2, figsize=(9,3.5))
    ax[0].plot(ratios, rel_w01); ax[0].axhline(0, ls='--', lw=1)
    ax[0].set_xlabel('$E_J/E_C$'); ax[0].set_ylabel('rel. error $\\omega_{01}$')
    ax[1].plot(ratios, rel_a);   ax[1].axhline(0, ls='--', lw=1)
    ax[1].set_xlabel('$E_J/E_C$'); ax[1].set_ylabel('rel. error $\\alpha$')
    plt.tight_layout()
    plt.savefig('asymptotic_errors.png', dpi=300)

def charge_dispersion(EJ_over_EC, n_cut=14, n_samples=101):
    ngs = np.linspace(0.0, 0.5, n_samples)
    E01 = []
    for ng in ngs:
        es = eigensystem_transmon(n_cut, EJ_over_EC, n_g=ng)
        E01.append(es.evals[1]-es.evals[0])
    E01 = np.array(E01)
    amp = 0.5*(E01.max() - E01.min())
    return ngs, E01, amp


# %%
EJ_over_EC = 50.0
n_cuts = [6, 8, 10, 12, 14]
conv = validate_charge_basis(n_cuts, EJ_over_EC, n_g=0.0)
print("Convergence table (E_C units):")
for nc in n_cuts:
    print(nc, conv[nc])
plot_charge_basis_convergence(conv)

ratios = np.linspace(10, 120, 56)
r, E01s, alphas = sweep_alpha_vs_ratio(ratios, n_cut=14, n_g=0.0)
plot_alpha_vs_ratio(r, alphas)

elems = sweep_matrix_elements_vs_ratio(r, n_cut=14, n_g=0.0, pairs=[(0,1), (1,2)])
plot_matrix_elements_vs_ratio(r, elems)

es = eigensystem_transmon(n_cut=14, EJ_over_EC=EJ_over_EC, n_g=0.0)
checks = wavefunction_validity(es.evecs, es.n_vals, n_states=3, Nphi=2048)
print("Wavefunction validity metrics:", checks)
phi, psi = charge_to_phase_wavefunctions(es.evecs, es.n_vals, n_states=3, Nphi=2048)
plot_wavefunctions_with_potential(phi, psi, EJ_over_EC=EJ_over_EC, scale=0.2)

ngs, E01_ng, A = charge_dispersion(50.0, n_cut=14)
print("charge-dispersion amplitude ΔE01/2 (E_C units):", A)
plt.figure(); plt.plot(ngs, E01_ng); plt.xlabel('$n_g$'); plt.ylabel('$E_{01} / E_C$'); plt.grid(True, alpha=0.3); plt.savefig('charge_dispersion.png', dpi=300);

plt.show()



# %%
def _bdg_chain(NL, NN, NR, t, mu, Delta, phi, Vbar=0.0):
    N = NL + NN + NR
    H0 = np.zeros((N, N), dtype=np.complex128)
    ons = -mu * np.ones(N)
    if NN > 0 and Vbar != 0.0:
        ons[NL:NL+NN] += Vbar
    np.fill_diagonal(H0, ons)

    for i in range(N-1):
        H0[i, i+1] = -t
        H0[i+1, i] = -t

    Delta_mat = np.zeros((N, N), dtype=np.complex128)
    if Delta != 0.0:
        if NL > 0:
            Delta_mat[:NL, :NL] = np.diag(Delta * np.exp(1j * phi/2) * np.ones(NL))
        if NR > 0:
            Delta_mat[-NR:, -NR:] = np.diag(Delta * np.exp(-1j * phi/2) * np.ones(NR))

    Hbdg = np.block([
        [H0,              Delta_mat],
        [Delta_mat.conj().T, -H0.T]
    ])
    return Hbdg

def compute_abs_spectrum_vs_phase(
    NL=60, NN=1, NR=60, *, t=1.0, mu=0.0, Delta=0.2,
    Vbar=0.0, n_phi=241, phi_min=0.0, phi_max=2*np.pi, ecut=None
):
    phis = np.linspace(phi_min, phi_max, n_phi)
    if ecut is None:
        ecut = 0.999 * abs(Delta) if Delta != 0 else 1e9

    subgap_E = []
    for phi in phis:
        Hbdg = _bdg_chain(NL, NN, NR, t, mu, Delta, phi, Vbar=Vbar)
        # Hermitian: use eigvalsh
        evals = np.linalg.eigvalsh(Hbdg)
        # Keep only subgap states (|E| < ecut)
        mask = np.abs(evals) < ecut
        subgap_E.append(evals[mask])
    return phis, subgap_E

def plot_abs_vs_phase(
    NL=60, NN=1, NR=60, *, t=1.0, mu=0.0, Delta=0.2, Vbar=0.0,
    n_phi=241, phi_min=0.0, phi_max=2*np.pi, overlay_tau=None, ax=None
):
    phis, subgap_E = compute_abs_spectrum_vs_phase(
        NL, NN, NR, t=t, mu=mu, Delta=Delta, Vbar=Vbar,
        n_phi=n_phi, phi_min=phi_min, phi_max=phi_max
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,4.2))

    for i, phi in enumerate(phis):
        if subgap_E[i].size:
            ax.plot([phi]*len(subgap_E[i]), subgap_E[i], '.', ms=2)

    ax.hlines([+abs(Delta), -abs(Delta)], phi_min, phi_max,
              linestyles='--', linewidth=1, alpha=0.5)

    if overlay_tau is not None:
        tau = float(overlay_tau)
        Es = abs(Delta) * np.sqrt(1.0 - tau * np.sin(phis/2.0)**2)
        ax.plot(phis, +Es, lw=2, alpha=0.8, label=fr'analytic +,  $\tau={tau:.2f}$')
        ax.plot(phis, -Es, lw=2, alpha=0.8, label=fr'analytic −,  $\tau={tau:.2f}$')

    ax.set_xlim(phi_min, phi_max)
    ax.set_xlabel(r'Phase difference $\phi$')
    ax.set_ylabel('Energy')
    ax.set_title('Andreev Bound States vs Phase (tight-binding BdG)')
    if overlay_tau is not None:
        ax.legend(frameon=False)
    return ax

ax = plot_abs_vs_phase(NL=80, NN=1, NR=80, t=1.0, mu=0.0, Delta=0.2, Vbar=1.5,
                       overlay_tau=0.1)  # small τ → nearly flat ABS near ±Δ
plt.savefig('abs_vs_phase_1.png', dpi=300)
ax = plot_abs_vs_phase(NL=80, NN=1, NR=80, t=1.0, mu=0.0, Delta=0.2, Vbar=0.2,
                       overlay_tau=0.8)
plt.savefig('abs_vs_phase_2.png', dpi=300)
ax = plot_abs_vs_phase(NL=80, NN=5, NR=80, t=1.0, mu=0.0, Delta=0.2, Vbar=0.5)
plt.savefig('abs_vs_phase_3.png', dpi=300)
plt.tight_layout()
plt.show()

# %%
def abs_positive_vs_phase(NL, NN, NR, *, t=1.0, mu=0.0, Delta=0.2, Vbar=0.0,
                          n_phi=401, phi_min=0.0, phi_max=2*np.pi, ecut=None):
    phis = np.linspace(phi_min, phi_max, n_phi)
    if ecut is None:
        ecut = 0.999*abs(Delta) if Delta != 0 else 1e9
    rows = []
    for phi in phis:
        H = _bdg_chain(NL, NN, NR, t, mu, Delta, phi, Vbar)
        ev = np.linalg.eigvalsh(H)
        pos = ev[(ev>0) & (np.abs(ev)<ecut)]
        rows.append(np.sort(pos))
    nlev = min(len(r) for r in rows)
    Epos = np.array([r[:nlev] for r in rows])   # shape (n_phi, nlev)
    return phis, Epos

def cpr_from_abs(NL, NN, NR, *, t=1.0, mu=0.0, Delta=0.2, Vbar=0.0,
                 n_phi=401, T=0.0, kB=1.0):
    phis, Epos = abs_positive_vs_phase(NL, NN, NR, t=t, mu=mu, Delta=Delta, Vbar=Vbar, n_phi=n_phi)
    dphi = phis[1]-phis[0]
    dE_dphi = np.gradient(Epos, dphi, axis=0)
    # thermal occupation factor
    if T <= 0:
        occ = 1.0
    else:
        occ = np.tanh(Epos/(2.0*kB*T))
    Iphi = 2.0 * np.sum(dE_dphi * occ, axis=1)
    return phis, Iphi

def josephson_energy_from_cpr(phis, Iphi):
    dE_dphi = 0.5 * Iphi
    EJ = np.zeros_like(phis)
    EJ[1:] = np.cumsum(0.5*(dE_dphi[1:]+dE_dphi[:-1]) * np.diff(phis))
    return EJ

def plot_cpr_and_ej(NL=80, NN=1, NR=80, *, t=1.0, mu=0.0, Delta=0.2, Vbar=0.5,
                    n_phi=401, T=0.0, kB=1.0):
    phis, Iphi = cpr_from_abs(NL, NN, NR, t=t, mu=mu, Delta=Delta, Vbar=Vbar,
                              n_phi=n_phi, T=T, kB=kB)
    EJ = josephson_energy_from_cpr(phis, Iphi)
    fig, ax = plt.subplots(1,2, figsize=(11,4))
    ax[0].plot(phis, Iphi, lw=2)
    ax[0].set_xlabel(r'$\phi$'); ax[0].set_ylabel(r'$I(\phi)$ (arb.)'); ax[0].set_title('Current–Phase Relation')
    ax[1].plot(phis, EJ, lw=2)
    ax[1].set_xlabel(r'$\phi$'); ax[1].set_ylabel(r'$E_J(\phi)$ (arb.)'); ax[1].set_title('Josephson Energy (integrated)')
    fig.tight_layout()
    Ic = np.max(np.abs(Iphi))
    return phis, Iphi, EJ, Ic

_ = plot_cpr_and_ej(NL=100, NN=1, NR=100, Delta=0.2, Vbar=1.5, T=0.0)   # tunnel-like, sinusoidal
plt.savefig('cpr_ej_tunnel.png', dpi=300)
# _ = plot_cpr_and_ej(NL=100, NN=1, NR=100, Delta=0.2, Vbar=0.2, T=0.0)   # transparent, forward-skewed
# plt.savefig('cpr_ej_transparent.png', dpi=300)
# _ = plot_cpr_and_ej(NL=100, NN=5, NR=100, Delta=0.2, Vbar=0.5, T=0.02)  # longer link + finite T
# plt.savefig('cpr_ej_longer_finiteT.png', dpi=300)


# %%
def Ic_vs_T(T_list, **kwargs):
    Ics = []
    for T in T_list:
        _, _, _, Ic = plot_cpr_and_ej(T=T, **kwargs)
        Ics.append(Ic)
    return np.array(T_list), np.array(Ics)

T_list = np.linspace(0.0, 0.15, 16)  # in units of energy scale/kB
Ts, Ics = Ic_vs_T(T_list, NL=120, NN=1, NR=120, Delta=0.2, Vbar=0.8, n_phi=501)
plt.figure(figsize=(5.0,3.8)); plt.plot(Ts, Ics, 'o-'); plt.xlabel('T'); plt.ylabel('$I_c$ (arb.)'); plt.title('$I_c(T)$'); plt.show()

# %%
def sweep_barrier_V(NL=100, NN=1, NR=100, Delta=0.2, t=1.0, mu=0.0,
                    V_list=np.linspace(0.0, 2.0, 9), n_phi=501, T=0.0, kB=1.0,
                    show_examples=(0.0, 0.5, 1.5)):
    Ics = []
    CPRs = {}
    phis_common = None
    for V in V_list:
        phis, Iphi = cpr_from_abs(NL, NN, NR, t=t, mu=mu, Delta=Delta, Vbar=V, n_phi=n_phi, T=T, kB=kB)
        Ic = np.max(np.abs(Iphi))
        Ics.append(Ic)
        if phis_common is None: phis_common = phis
        if np.isclose(V, show_examples).any():
            CPRs[V] = Iphi

    # Plot Ic(Vbar)
    plt.figure(figsize=(5.2,3.8))
    plt.plot(V_list, Ics, 'o-'); plt.xlabel('Barrier height $V_\\mathrm{bar}$'); plt.ylabel('$I_c$ (arb.)')
    plt.title('$I_c$ vs barrier (transparency)'); plt.tight_layout()

    # Example CPRs
    if CPRs:
        plt.figure(figsize=(6.2,3.8))
        for V, Iphi in CPRs.items():
            plt.plot(phis_common, Iphi, label=f'Vbar={V:g}')
        plt.xlabel(r'$\phi$'); plt.ylabel(r'$I(\phi)$ (arb.)'); plt.title('CPR at selected barriers')
        plt.legend(frameon=False); plt.tight_layout()
    return np.array(V_list), np.array(Ics)

# Example sweep
V_list, Ics = sweep_barrier_V(NL=120, NN=1, NR=120, Delta=0.2, V_list=np.linspace(0.0, 2.0, 11))
plt.savefig('ic_vs_barrier.png', dpi=300)

# %%
import numpy as np
import numpy.linalg as la


def current_operator_dH_dphi(NL, NN, NR, *, Delta=0.2, phi=0.0):
    N = NL + NN + NR
    Z = np.zeros((N, N), dtype=np.complex128)

    dDelta = np.zeros((N, N), dtype=np.complex128)
    if NL > 0:
        dDelta[:NL, :NL] = np.diag(1j * 0.5 * Delta * np.exp(1j * phi/2.0) * np.ones(NL))
    if NR > 0:
        dDelta[-NR:, -NR:] = np.diag(-1j * 0.5 * Delta * np.exp(-1j * phi/2.0) * np.ones(NR))

    dH = np.block([[Z, dDelta], [dDelta.conj().T, Z]])
    return dH

def _phase_align_columns(U_prev, U_cur):
    ov = np.sum(U_prev.conj() * U_cur, axis=0)
    phases = np.exp(-1j * np.angle(ov + 1e-30))
    return U_cur * phases

def cpr_hellmann_feynman_clean(
    NL=120, NN=1, NR=120, *, t=1.0, mu=0.0, Delta=0.2, Vbar=0.5,
    n_phi=501, T=0.0, kB=1.0, prefactor=2.0
):
    phis = np.linspace(0.0, 2*np.pi, n_phi)
    Iphi = np.zeros_like(phis, dtype=float)

    evecs_prev = None
    for j, phi in enumerate(phis):
        H = _bdg_chain(NL, NN, NR, t, mu, Delta, phi, Vbar)
        evals, evecs = la.eigh(H)
        if evecs_prev is not None:
            evecs = _phase_align_columns(evecs_prev, evecs)
        evecs_prev = evecs.copy()
        dH = current_operator_dH_dphi(NL, NN, NR, Delta=Delta, phi=phi)
        pos = evals > 0
        Epos = evals[pos]
        Vpos = evecs[:, pos]
        M = Vpos.conj().T @ dH @ Vpos
        dH_exp = np.real(np.diag(M))
        if T <= 0:
            occ = np.tanh(Epos / (2.0*kB*1e-12))
        else:
            occ = np.tanh(Epos / (2.0*kB*T))

        Iphi[j] = prefactor * np.sum(occ * dH_exp)

    return phis, Iphi

phis_hf, I_hf = cpr_hellmann_feynman_clean(NL=150, NN=1, NR=150, Delta=0.2, Vbar=0.0, n_phi=801) 
phis_abs, I_abs = cpr_from_abs(NL=150, NN=1, NR=150, Delta=0.2, Vbar=0.0, n_phi=801, T=0.0)

plt.figure(figsize=(6.2,4.0))
plt.plot(phis_abs, I_abs, label='ABS derivative', lw=2)
plt.plot(phis_hf,  I_hf,  '--', label='Hellmann–Feynman (clean)', lw=2)
plt.xlabel(r'$\phi$'); plt.ylabel(r'$I(\phi)$ (arb.)')
plt.legend(frameon=False); plt.tight_layout()
plt.title('CPR comparison: ABS derivative vs Hellmann–Feynman')
plt.savefig('cpr_hf_vs_abs.png', dpi=300)
plt.show()

# %%
def fourier_cpr_sine_coeffs(phis, Iphi, mmax=None):
    phis = np.asarray(phis)
    Iphi = np.asarray(Iphi)
    assert phis.ndim == 1 and Iphi.shape == phis.shape
    if mmax is None:
        mmax = len(phis)//4  # conservative default
    dphi = phis[1]-phis[0]
    if not np.isclose((phis[-1]-phis[0]+dphi), 2*np.pi, rtol=1e-3, atol=1e-6):
        raise ValueError("phis must span [0, 2π) with uniform spacing")

    Im = np.zeros(mmax, dtype=float)
    for m in range(1, mmax+1):
        Im[m-1] = (1/np.pi) * np.sum(Iphi * np.sin(m*phis)) * dphi
    return Im  # length mmax

def EJ_coeffs_from_cpr(phis, Iphi, mmax=None):
    Im = fourier_cpr_sine_coeffs(phis, Iphi, mmax=mmax)
    m = np.arange(1, len(Im)+1, dtype=float)
    EJm = 0.5 * Im / m
    return EJm, Im

phis, Iphi = cpr_from_abs(NL=150, NN=1, NR=150, Delta=0.2, Vbar=0.5, n_phi=1024)
EJm, Im = EJ_coeffs_from_cpr(phis, Iphi, mmax=15)
m = np.arange(1, len(Im)+1)
I_recon = np.sum(Im[:,None] * np.sin(m[:,None]*phis[None,:]), axis=0)

plt.figure(figsize=(6.2,4))
plt.plot(phis, Iphi, label='CPR (TB)', lw=2)
plt.plot(phis, I_recon, '--', label='Fourier recon (M=15)', lw=2)
plt.xlabel(r'$\phi$'); plt.ylabel(r'$I(\phi)$'); plt.legend(frameon=False); plt.tight_layout()
plt.title('CPR and Fourier reconstruction from harmonics')
plt.savefig('cpr_fourier_recon.png', dpi=300); plt.show()

# %%
def transmon_hamiltonian_from_EJm(EJm, EC, ncut=10):
    ns = np.arange(-ncut, ncut+1, dtype=float)
    dim = len(ns)
    H = np.zeros((dim, dim), dtype=np.complex128)
    H += np.diag(4*EC*ns**2)
    for m, EJm_m in enumerate(EJm, start=1):
        hop = -0.5*EJm_m
        for i,n in enumerate(ns):
            j = np.where(ns == n+m)[0]
            k = np.where(ns == n-m)[0]
            if j.size: H[i, j[0]] += hop
            if k.size: H[i, k[0]] += hop
    H = 0.5*(H + H.conj().T)
    return H

def synth_1overf(T, dt, A, fmin=1.0, fmax=1e6, rng=np.random.default_rng()):
    N = int(np.round(T/dt))
    freqs = np.fft.rfftfreq(N, dt)
    S = np.zeros_like(freqs); 
    band = (freqs>=fmin) & (freqs<=fmax)
    S[band] = A/np.clip(freqs[band], fmin, None)
    phases = rng.uniform(0, 2*np.pi, size=S.size) * 1j
    spec = np.sqrt(S) * (np.cos(phases.imag)+1j*np.sin(phases.imag))
    x = np.fft.irfft(spec, n=N)
    return x

def plot_cpr_and_fourier_recon(NL=150, NN=1, NR=150, *, Delta=0.2, Vbar=0.5,
                               n_phi=1024, M=15, method='abs'):
    if method == 'abs':
        phis, Iphi = cpr_from_abs(NL, NN, NR, Delta=Delta, Vbar=Vbar, n_phi=n_phi)
    else:
        phis, Iphi = cpr_hellmann_feynman_clean(NL, NN, NR, Delta=Delta, Vbar=Vbar, n_phi=n_phi)
    EJm, Im = EJ_coeffs_from_cpr(phis, Iphi, mmax=M)
    m = np.arange(1, len(Im)+1)
    I_recon = (Im[:,None] * np.sin(m[:,None]*phis[None,:])).sum(axis=0)

    fig, ax = plt.subplots(1,2, figsize=(11,4))
    ax[0].plot(phis, Iphi, label='CPR (TB)', lw=2)
    ax[0].plot(phis, I_recon, '--', label=f'Fourier recon (M={M})', lw=2)
    ax[0].set_xlabel(r'$\phi$'); ax[0].set_ylabel(r'$I(\phi)$ (arb.)'); ax[0].legend(frameon=False)
    ax[0].set_title('CPR vs Fourier reconstruction')

    ax[1].stem(m, Im, basefmt=" ")
    ax[1].set_xlabel('harmonic m'); ax[1].set_ylabel(r'$I_m$ (arb.)')
    ax[1].set_title('Sine-series coefficients of CPR')
    fig.tight_layout()
    return phis, Iphi, (EJm, Im)

def sweep_fourier_vs_barrier(V_list, NL=150, NN=1, NR=150, *, Delta=0.2,
                             n_phi=1024, M=6, method='abs'):
    Im_store = []
    for V in V_list:
        phis, Iphi = (cpr_from_abs if method=='abs' else cpr_hellmann_feynman_clean)(
            NL, NN, NR, Delta=Delta, Vbar=V, n_phi=n_phi)
        _, Im = EJ_coeffs_from_cpr(phis, Iphi, mmax=M)
        Im_store.append(Im)
    Im_store = np.array(Im_store) 

    plt.figure(figsize=(6.5,4.2))
    for m in range(M):
        plt.plot(V_list, np.abs(Im_store[:,m]), marker='o', label=f'm={m+1}')
    plt.xlabel(r'Barrier $V_{\rm bar}$'); plt.ylabel(r'$|I_m|$ (arb.)')
    plt.title('Fourier content vs transparency'); plt.legend(frameon=False, ncol=2); plt.tight_layout()
    return Im_store

def sweep_fourier_vs_length(NN_list, NL=150, NR=150, *, Delta=0.2, Vbar=0.5,
                            n_phi=1024, M=6, method='abs'):
    Im_store = []
    for NN in NN_list:
        phis, Iphi = (cpr_from_abs if method=='abs' else cpr_hellmann_feynman_clean)(
            NL, NN, NR, Delta=Delta, Vbar=Vbar, n_phi=n_phi)
        _, Im = EJ_coeffs_from_cpr(phis, Iphi, mmax=M)
        Im_store.append(Im)
    Im_store = np.array(Im_store)

    plt.figure(figsize=(6.5,4.2))
    for m in range(M):
        plt.plot(NN_list, np.abs(Im_store[:,m]), marker='o', label=f'm={m+1}')
    plt.xlabel('link length $N_N$ (sites)'); plt.ylabel(r'$|I_m|$ (arb.)')
    plt.title('Fourier content vs junction length'); plt.legend(frameon=False, ncol=2); plt.tight_layout()
    return Im_store

def spectrum_from_EJm(EJm, EC=1.0, ncut=12):
    H = transmon_hamiltonian_from_EJm(EJm, EC, ncut=ncut)
    evals, vecs = np.linalg.eigh(H)
    evals = np.real(evals)
    evals -= evals[0]
    w01 = evals[1]
    alpha = (evals[2]-evals[1]) - (evals[1]-evals[0])   # (E2-E1) - (E1-E0)
    return w01, alpha

def sweep_transmon_vs_barrier(V_list, EC=1.0, *, NL=150, NN=1, NR=150, Delta=0.2,
                              n_phi=1024, M=10, ncut=12, method='abs'):
    w01s, alphas = [], []
    for V in V_list:
        phis, Iphi = (cpr_from_abs if method=='abs' else cpr_hellmann_feynman_clean)(
            NL, NN, NR, Delta=Delta, Vbar=V, n_phi=n_phi)
        EJm, Im = EJ_coeffs_from_cpr(phis, Iphi, mmax=M)
        w01, alpha = spectrum_from_EJm(EJm, EC=EC, ncut=ncut)
        w01s.append(w01); alphas.append(alpha)
    w01s, alphas = np.array(w01s), np.array(alphas)

    fig, ax = plt.subplots(1,2, figsize=(11,4))
    ax[0].plot(V_list, w01s, 'o-'); ax[0].set_xlabel(r'$V_{\rm bar}$'); ax[0].set_ylabel(r'$\omega_{01}$ (arb.)')
    ax[0].set_title('TB-driven transmon ω01 vs barrier')
    ax[1].plot(V_list, alphas, 'o-'); ax[1].set_xlabel(r'$V_{\rm bar}$'); ax[1].set_ylabel(r'$\alpha$ (arb.)')
    ax[1].set_title('TB-driven transmon anharmonicity vs barrier')
    fig.tight_layout()
    return w01s, alphas

def transmon_vs_flux_from_singleJJ(EJm_single, EC=1.0, ncut=12, nPhi=121):
    Phi = np.linspace(0, 1.0, nPhi)  # Φ/Φ0
    w01s = np.zeros_like(Phi); alphas = np.zeros_like(Phi)
    m = np.arange(1, len(EJm_single)+1)[:,None]  # (M,1)
    for i,phi in enumerate(Phi):
        scale = 2.0 * np.cos(m*np.pi*phi)        # (M,1)
        EJm_flux = (EJm_single[:,None] * scale).ravel()
        w01, alpha = spectrum_from_EJm(EJm_flux, EC=EC, ncut=ncut)
        w01s[i] = w01; alphas[i] = alpha
    fig, ax = plt.subplots(1,2, figsize=(11,4))
    ax[0].plot(Phi, w01s); ax[0].set_xlabel(r'$\Phi/\Phi_0$'); ax[0].set_ylabel(r'$\omega_{01}$')
    ax[0].set_title('ω01(Flux) from TB-derived EJ harmonics')
    ax[1].plot(Phi, alphas); ax[1].set_xlabel(r'$\Phi/\Phi_0$'); ax[1].set_ylabel(r'$\alpha$')
    ax[1].set_title('α(Flux) from TB-derived EJ harmonics')
    fig.tight_layout()
    return Phi, w01s, alphas


_ = plot_cpr_and_fourier_recon(Vbar=0.2, M=20)   # transparent → many harmonics
_ = plot_cpr_and_fourier_recon(Vbar=1.5, M=6)    # tunneling → almost sinusoidal

V_list = np.linspace(0.0, 2.0, 11)
_ = sweep_fourier_vs_barrier(V_list, M=6)

NN_list = [1,2,3,4,5,8,10]
_ = sweep_fourier_vs_length(NN_list, M=6, Vbar=0.4)

V_list = np.linspace(0.0, 2.0, 11)
_ = sweep_transmon_vs_barrier(V_list, EC=1.0, M=12, ncut=14)

# get single-JJ EJm at some barrier (e.g., Vbar=0.5)
phis, Iphi = cpr_from_abs(150,1,150, Delta=0.2, Vbar=0.5, n_phi=1024)
EJm_single, _ = EJ_coeffs_from_cpr(phis, Iphi, mmax=12)
_ = transmon_vs_flux_from_singleJJ(EJm_single, EC=1.0, ncut=14)


# %% [markdown]
# ### Error Isolation Procedure
# Currently, the microscopic Hamiltonian is presented as:
# \begin{equation*}
# H(t;\theta(t)) = 4E_C n^2 - \sum_m E_J^{(m)} (\Phi(t) + \delta \Phi(t))\cos(m\phi) + H_{\mathrm{drive}}(t), \quad \theta(t) \in \{\delta \Phi, \delta E_J^{(m)} , \delta w_d, 1/\mathrm{f_{noise}} \},
# \end{equation*}
# where $\theta(t)$ represents the stochastic error parameter. We can project this into a small subspace $\mathcal{S}$ using the TB-derived $E_J^{(m)}$. During a gate on $[1,T]$, the propagater $\mathcal{S}$ is:
# \begin{equation*}
#     U_{\mathrm{act}} = \mathcal{T} \mathrm{exp}\left[-i \int_0^T H_\mathcal{S}(t;\theta(t))\, dt \right], \quad U_{\mathrm{ideal}} = \mathcal{T} \mathrm{exp}\left[H_\mathcal{S}(t;0)\, dt \right].
# \end{equation*}
# We define the error unitary operator in the computational subspace as:
# \begin{equation*}
#     U_{\mathrm{error}} = U_{\mathrm{act}} U_{\mathrm{ideal}}^\dagger.
# \end{equation*}
# If there is leakage, we also keep the block off the $\{\ket{0},\ket{1} \}$ subspace; that captures non-unitary errors we can't simply invert with a two-level correction.
# 
# To properly isolate and reverse this error, we follow the following procedure:
# - Build the reduced model by diagonalizing the static TB-driven transmon Hamiltonian at the working point to get $\ket{0},\ket{1},\ket{2}$, then precompute the matrices in the basis $n,\phi,\frac{\partial H}{\partial \Phi}, \frac{\partial H}{\partial E_J^{(m)}}$. This would give a fast time revolution while remaining microscopically physical.
# - Simulate a noisy gate by picking a control $H_{\mathrm{drive}}$ (Gaussian $\pi$ pulse, flux phase, etc) and a noise trajectory $\theta(t)$. 
# - Propagate to get $U_{\mathrm{act}}$ in $\mathcal{S}$ by a magnus expansion. 
# - Extract and parametrize the error by computing $U_{\mathrm{error}} = U_{\mathrm{act}}U_{\mathrm{ideal}}^\dagger$. We then write the $U_{\mathrm{erorr}}=e^{iK}$ in the qubit subspace with $K=\frac{\epsilon}{2} \hat{n}\cdot \vec{\sigma}$ where $K_z$ causes a detuning or phase error through a $Z$ rotation, and $K_{x,y}$ cause an amplitude or timing error through an axis tilt. 
# - Synthesize a correction by applying $U_{\mathrm{correct}}=U_{error}^\dagger$ as a short correcting pulse. To account for multiple errors, we can use a gradient-based pulse optimization (GRAPE) sequence since the TB pipeline gives analytical derivatives via $\frac{\partial H}{\partial \Phi}$ and $\frac{\partial H}{\partial E_J^{(m)}}$.
# 
# Another method to tune out noise into a correctable unitary operation would be by assuming quasi-static slow noise over a gate such that $\theta(t)\approx \theta_0$ as a constant. This would give:
# \begin{equation*}
#     U_{\mathrm{error}} \approx \mathrm{exp}\left[-i\theta_0\int_0^T G(t)\, dt\right], \quad G(t) = \frac{\partial H}{\partial \Phi}
# \end{equation*}
# such that $G(t)$ is a noise generator. In this case, we can either estimate $\theta_0$ through tomography, or dynamically correct the gates by minimizing $\int_0^T f(\omega) S_\theta(\omega)\, d\omega$.

# %%
# Some Helpers
def PO1(d):
    P = xp.zeros((d,d), dtype=xp.complex128)
    P[0,0] = 1.0; P[1,1] = 1.0
    return P

def truncate_to_qubit(rho_d):
    d = rho_d.shape[0]
    P = PO1(d)
    rho_2 = P[:2,:2] @ rho_d[:2,:2]; pop_01 = float(xp.trace(rho_2.real))
    return rho_2 / (pop_01+1e-18), pop_01

_SIG = [
    xp.array([[1,0],[0,1]], dtype=xp.complex128), # I
    xp.array([[0,1],[1,0]], dtype=xp.complex128), # X
    xp.array([[0,-1j],[1j,0]], dtype=xp.complex128), # Y
    xp.array([[1,0],[0,-1]], dtype=xp.complex128)  # Z
]

def vec(r):
    return r.reshape((-1,1))
def unvec(v):
    return v.reshape((2,2))

# Backend-safe dimension-aware vec/unvec
def vecN(rho):
    """Flatten dxd -> (d*d, 1)."""
    rho = xp.asarray(rho)
    return rho.reshape((-1, 1))

def unvecN(v):
    """Infer d from length and reshape (d*d,1) or (d*d,) -> (d,d)."""
    v = xp.asarray(v)
    n = int(v.size)
    d = int(round(n**0.5))
    if d * d != n:
        raise ValueError(f"unvecN: length {n} is not a perfect square")
    return v.reshape((d, d))

def pauli_transfer_matrix(E):
    B = [S/xp.sqrt(2.0) for S in _SIG]
    R = xp.zeros((4,4), dtype=xp.complex128)
    for i, Si in enumerate(B):
        vin = vec(Si)
        vout = E(vin)
        rho_out = unvec(vout)
        for j, Sj in enumerate(B):
            R[j,i] = xp.trace(Sj.conj().T @ rho_out)
    return R

# %%
def evolve_lindblad_fixedH(rho0,H,Ls,tmax,dt):
    def Ht(t,*args): return H
    t=0.0; rho=rho0.copy()
    while t < tmax - 1e-18:
        rho = rk4_step(rho,t,dt,Ht,tuple(),Ls)
        t += dt
    return rho

def evolve_stochastic_ng_pure(psi0,EC,Em,n_cut,tmax,dt,tau_c,sigma_ng,n_g0=0.0):
    d = psi0.shape[0]
    t = 0.0; n_g = n_g0; psi = psi0.copy()
    rng = xp.random.default_rng()
    while t <= tmax - 1e-18:
        H,ns = transmon_H(EC,Em,n_cut,n_g)
        Efull, Ufull = xp.linalg.eigh(H)
        Es_t = Efull[:d]; Ht = xp.diag(Es_t)
        psi = psi - 1j*dt*(Ht @ psi)
        psi = psi / xp.linalg.norm(psi)
        n_g = ou_step(n_g, dt, tau_c, sigma_ng, rng)
        t += dt
    return xp.outer(psi, psi.conj())

def average_channel_over_traj(make_rho1, N=16):
    def E(vin):
        # vin may be (d*d, ) or (d*d,1) and may be numpy or cupy -> use unvecN
        rho0 = unvecN(xp.asarray(vin))
        outs = []
        for _ in range(N):
            outs.append(make_rho1(rho0))
        rho = sum(outs) / float(N)
        return vecN(rho)
    return E

# %%
def qubit_channel_from_d_evolution(make_rho1_d):
    def bloch_vec(rho2):
        X = xp.array([[0,1],[1,0]], dtype=xp.complex128)
        Y = xp.array([[0,-1j],[1j,0]], dtype=xp.complex128)
        Z = xp.array([[1,0],[0,-1]], dtype=xp.complex128)
        return xp.array([
            float(xp.real(xp.trace(X @ rho2))),
            float(xp.real(xp.trace(Y @ rho2))),
            float(xp.real(xp.trace(Z @ rho2))),
        ])

    d = 3 
    e0 = xp.zeros(d, dtype=xp.complex128); e0[0]=1
    e1 = xp.zeros(d, dtype=xp.complex128); e1[1]=1
    ep = (e0+e1)/xp.sqrt(2); ei = (e0+1j*e1)/xp.sqrt(2)
    probes = [xp.outer(e0,e0.conj()), xp.outer(e1,e1.conj()),
              xp.outer(ep,ep.conj()), xp.outer(ei,ei.conj())]

    S_in = xp.array([[0,0,1],[0,0,-1],[1,0,0],[0,1,0]], dtype=xp.float64).T  # 3x4
    S_out = xp.zeros((3,4), dtype=xp.float64)
    pops01 = []

    for j, rho_d in enumerate(probes):
        rho1_d = make_rho1_d(rho_d)
        rho2_block = rho1_d[:2,:2]
        pop01 = float(xp.real(xp.trace(rho2_block)))
        pops01.append(pop01)
        rho2 = rho2_block / (pop01 + 1e-18)
        S_out[:, j] = bloch_vec(rho2)

    Sin_aug = xp.vstack([S_in, xp.ones((1,4), dtype=xp.float64)])  # 4x4
    Sout_aug = xp.vstack([S_out, xp.ones((1,4), dtype=xp.float64)])# 4x4
    M = Sout_aug @ xp.linalg.pinv(Sin_aug)
    A = xp.asarray(M[:3,:3]); t = xp.asarray(M[:3,3])

    R = xp.eye(4, dtype=xp.float64)
    R[1:,1:] = A; R[1:,0] = t
    leakage_metric = float(xp.mean(xp.array(pops01)))
    return R, leakage_metric

def channel_from_d_evolution(make_rho1_d):
    """Return (R, t, leaks_per_probe, leak_avg) for the induced qubit channel."""
    X = xp.array([[0,1],[1,0]], dtype=xp.complex128)
    Y = xp.array([[0,-1j],[1j,0]], dtype=xp.complex128)
    Z = xp.array([[1,0],[0,-1]], dtype=xp.complex128)
    def bloch_vec(rho2):
        return xp.array([
            float(xp.real(xp.trace(X @ rho2))),
            float(xp.real(xp.trace(Y @ rho2))),
            float(xp.real(xp.trace(Z @ rho2))),
        ])

    d = 3
    e0 = xp.zeros(d, dtype=xp.complex128); e0[0]=1
    e1 = xp.zeros(d, dtype=xp.complex128); e1[1]=1
    ep = (e0+e1)/xp.sqrt(2); ei = (e0+1j*e1)/xp.sqrt(2)
    probes = [xp.outer(e0,e0.conj()), xp.outer(e1,e1.conj()),
              xp.outer(ep,ep.conj()), xp.outer(ei,ei.conj())]

    S_in = xp.array([[0,0,1],[0,0,-1],[1,0,0],[0,1,0]], dtype=xp.float64).T  # 3x4
    S_out = xp.zeros((3,4), dtype=xp.float64)
    leaks = []

    for j, rho_d in enumerate(probes):
        rho1_d = make_rho1_d(rho_d)            # returns d×d
        rho2b = rho1_d[:2,:2]
        pop01 = float(xp.real(xp.trace(rho2b)))
        leaks.append(pop01)
        rho2 = rho2b / (pop01 + 1e-18)
        S_out[:, j] = bloch_vec(rho2)

    Sin_aug  = xp.vstack([S_in,  xp.ones((1,4), dtype=xp.float64)])  # 4×4
    Sout_aug = xp.vstack([S_out, xp.ones((1,4), dtype=xp.float64)])  # 4×4
    M = Sout_aug @ xp.linalg.pinv(Sin_aug)
    A = xp.asarray(M[:3,:3]); t = xp.asarray(M[:3,3])

    R = xp.eye(4, dtype=xp.float64)
    R[1:,1:] = A
    R[1:,0]  = t

    return R, t, leaks, float(xp.mean(xp.array(leaks)))

def build_R_for_segment_lindblad(H, Ls, dt_seg, dt_step):
    """Fixed-H Lindblad over duration dt_seg with RK4 step dt_step."""
    def evolve_lindblad_fixedH(rho0):
        def Ht(t, *args): return H
        tcur = 0.0; rho = rho0.copy()
        while tcur < dt_seg - 1e-18:
            rho = rk4_step(rho, tcur, dt_step, Ht, tuple(), Ls)
            tcur += dt_step
        return rho
    return channel_from_d_evolution(evolve_lindblad_fixedH)  # -> (R, t, leaks, leak_avg)

def evolve_stochastic_ng_pure(psi0, EC, Em, n_cut, dt_seg, dt_step, tau_c, sigma_ng, n_g0=0.0):
    t = 0.0; n_g = n_g0; psi = psi0.copy()
    rng = xp.random.default_rng()
    while t <= dt_seg + 1e-18:
        H, ns = transmon_H(EC, Em, n_cutoff=n_cut, n_g=n_g)
        Efull, Ufull = diagnolize_H(H)
        Es_t = Efull[:psi.size]
        Ht = xp.diag(Es_t)
        psi = psi - 1j * (Ht @ psi) * dt_step
        psi = psi / xp.linalg.norm(psi)
        n_g = ou_step(n_g, dt_step, tau_c, sigma_ng, rng)
        t += dt_step
    return xp.outer(psi, psi.conj())

def make_rho1_one_traj_factory(EC, Em, n_cut, dt_seg, dt_step, tau_c, sigma_ng, n_g0=0.0, d=3):
    def make_rho1_one_traj(rho_d):
        evals, evecs = xp.linalg.eigh(rho_d)
        out = xp.zeros_like(rho_d)
        for k in range(d):
            pk = float(xp.real(evals[k]))
            if pk > 1e-14:
                psi = evecs[:,k]
                out += pk * evolve_stochastic_ng_pure(psi, EC, Em, n_cut, dt_seg, dt_step, tau_c, sigma_ng, n_g0)
        return out
    return make_rho1_one_traj

def average_channel_over_trajectories(make_rho1, N=16):
    def E(rho0):
        return sum(make_rho1(rho0) for _ in range(N)) / float(N)
    return E

def build_R_for_segment_stochastic(EC, Em, n_cut, dt_seg, dt_step, tau_c, sigma_ng, n_g0=0.0, averaged=True, N=16):
    make_one = make_rho1_one_traj_factory(EC, Em, n_cut, dt_seg, dt_step, tau_c, sigma_ng, n_g0, d=3)
    make_rho1 = average_channel_over_trajectories(make_one, N) if averaged else make_one
    return channel_from_d_evolution(make_rho1)  # -> (R, t, leaks, leak_avg)


def isolate_coherent_from_R(R):
    # accept numpy or cupy arrays and handle complex numerical noise robustly
    if hasattr(R, "get"):
        R = R.get()
    R = np.asarray(R)
    # ensure shape is (4,4)
    if R.shape != (4,4):
        raise ValueError(f"Expected R shape (4,4), got {R.shape}")
    # Extract the 3x3 block and force it to a real matrix (drop tiny numerical imag parts)
    A = R[1:4, 1:4].astype(np.complex128)
    if np.iscomplexobj(A):
        imag_max = np.max(np.abs(A.imag))
        if imag_max > 1e-8:
            pass
        A = np.real(A)
    # SVD and project to nearest orthogonal matrix O in SO(3)
    U, S, VT = np.linalg.svd(A, full_matrices=True)
    O = U @ VT
    if np.linalg.det(O) < 0:
        U[:, -1] *= -1.0
        O = U @ VT
    O = np.asarray(O, dtype=float)  # ensure plain float ndarray
    # rotation angle from trace
    trace_term = 0.5 * (np.trace(O) - 1.0)
    trace_term = float(np.real(trace_term))
    trace_term = np.clip(trace_term, -1.0, 1.0)
    angle = float(np.arccos(trace_term))
    # extract ZYZ Euler angles robustly (all args must be real scalars)
    if abs(angle) < 1e-12:
        rz1 = ry = rz2 = 0.0
    else:
        rz1 = float(np.arctan2(O[1, 2], O[0, 2]))
        ry  = float(np.arccos(np.clip(O[2, 2], -1.0, 1.0)))
        rz2 = float(np.arctan2(O[2, 1], -O[2, 0]))
    RU = np.eye(4, dtype=float)
    RU[1:4, 1:4] = O
    # compute incoherent remainder (keep original dtype of R for that)
    Rincoh = R @ np.linalg.inv(RU)
    return {"RU": RU, "angles_zyz": (float(rz1), float(ry), float(rz2)),
            "R_incoh": Rincoh, "angle": float(angle)}


def su2_from_zyz(rz1, ry, rz2):
    a1 = float(rz1); a2 = float(ry); a3 = float(rz2)
    Rz1 = np.array([[np.exp(-1j*a1/2), 0.0],
                    [0.0, np.exp(1j*a1/2)]], dtype=np.complex128)
    Ry  = np.array([[np.cos(a2/2), -np.sin(a2/2)],
                    [np.sin(a2/2),  np.cos(a2/2)]], dtype=np.complex128)
    Rz2 = np.array([[np.exp(-1j*a3/2), 0.0],
                    [0.0, np.exp(1j*a3/2)]], dtype=np.complex128)
    return Rz1 @ Ry @ Rz2

def correction_unitary_from_RU(RU):
    out = isolate_coherent_from_R(RU)
    rz1, ry, rz2 = out["angles_zyz"]
    U_err = su2_from_zyz(rz1, ry, rz2)
    U_corr = U_err.conj().T
    return U_corr

def coarse_incoherent_params(R,leakage_avg):
    R = np.asarray(R)
    A = R[1:,1:]
    s_xy = 0.5*(A[0,0]+A[1,1])
    s_z = A[2,2]
    return {"s_xy": float(s_xy), "s_z": float(s_z), "leakage": float(leakage_avg)}

def make_rho1_d_lindblad_factory(H,Ls,tmax,dt):
    def make_rho1_d(rho0_d): return evolve_lindblad_fixedH(rho0_d,H,Ls,tmax,dt)
    return make_rho1_d

def bloch_vec(rho2):
    X = xp.array([[0,1],[1,0]], dtype=xp.complex128)
    Y = xp.array([[0,-1j],[1j,0]], dtype=xp.complex128)
    Z = xp.array([[1,0],[0,-1]], dtype=xp.complex128)
    return xp.array([
        float(xp.real(xp.trace(X @ rho2))),
        float(xp.real(xp.trace(Y @ rho2))),
        float(xp.real(xp.trace(Z @ rho2))),
    ])

def qubit_channel_affine_from_physical_probes(make_rho1_d):
    ket0 = xp.array([1,0,0], dtype=xp.complex128)
    ket1 = xp.array([0,1,0], dtype=xp.complex128)
    ketp = (xp.array([1,1,0], dtype=xp.complex128) / xp.sqrt(2)).astype(xp.complex128)
    keti = (xp.array([1,1j,0], dtype=xp.complex128) / xp.sqrt(2)).astype(xp.complex128)

    probes = [
        xp.outer(ket0, ket0.conj()),
        xp.outer(ket1, ket1.conj()),
        xp.outer(ketp, ketp.conj()),
        xp.outer(keti, keti.conj()),
    ]
    # input Bloch vectors for these probes
    S_in = xp.array([
        [0,  0,  1],   # |0>
        [0,  0, -1],   # |1>
        [1,  0,  0],   # |+>
        [0,  1,  0],   # |+i>
    ], dtype=xp.float64).T  # shape 3x4

    S_out = xp.zeros((3,4), dtype=xp.float64)
    pops01 = []
    for j, rho_d in enumerate(probes):
        rho1_d = make_rho1_d(rho_d)
        # project to qubit (top-left 2x2) WITHOUT renormalization (for leakage calc)
        rho2_block = rho1_d[:2,:2]
        pop01 = float(xp.real(xp.trace(rho2_block)))
        pops01.append(pop01)
        # now renormalize ONLY for Bloch-vector evaluation
        rho2 = rho2_block / (pop01 + 1e-18)
        S_out[:, j] = bloch_vec(rho2)

    Sin_aug = xp.vstack([S_in, xp.ones((1,4), dtype=xp.float64)])  # 4x4
    Sout_aug = xp.vstack([S_out, xp.ones((1,4), dtype=xp.float64)])  # 4x4
    M = Sout_aug @ xp.linalg.pinv(Sin_aug)  # 4x4 homogeneous; last row ~ [0,0,0,1]
    A = xp.asarray(M[:3,:3])
    t = xp.asarray(M[:3,3])

    # Build 4x4 PTM in (1, X, Y, Z) ordering for convenience (unital shift in first column)
    R = xp.eye(4, dtype=xp.float64)
    R[1:,1:] = A
    R[1:,0]  = t  # translation
    leakage_metric = float(xp.mean(xp.array(pops01)))  # average pop remaining in {|0,1>}
    return R, leakage_metric



# %%
d = 3
EC, EJ = 0.5, 20.0
Em = {1: EJ}
n_cut = 10
H3, ns = transmon_H(EC, Em, n_cutoff=n_cut, n_g=0.0)
evals, U = diagnolize_H(H3)
Es = evals[:d]
Ured = U[:,:d]
n_mat = matrix_elements(ns, Ured)
T1 = 20e-6; Tphi = 30e-6
Ls = collapse_ops(T1=T1, Tphi=Tphi, n_mat=n_mat)

tmax = 2.0e-6; dt = 2.0e-9
make_rho1_d = make_rho1_d_lindblad_factory(xp.diag(Es), Ls, tmax, dt)

R, leak = qubit_channel_affine_from_physical_probes(make_rho1_d)
iso = isolate_coherent_from_R(R)      # as before
U_corr = correction_unitary_from_RU(iso["RU"])
A = np.asarray(R.get() if hasattr(R, "get") else R)[1:,1:]
print("A singular values:", np.linalg.svd(A, compute_uv=False))
print("leakage (avg pop kept in {|0,1>} for |0>,|1|,|+>,|+i>):", leak)

# %%
def embed_qubit_unitary(U2, d=3, xp_mod=xp):
    U = xp_mod.eye(d, dtype=xp_mod.complex128)
    U2_xp = to_xp(U2, xp_mod)
    if U2_xp.shape != (2,2):
        raise ValueError(f"U2 must be 2x2, got {U2_xp.shape}")
    U[:2,:2] = U2_xp
    return U

# Example, correct-before evolution:
U_corr = to_xp(U_corr, xp)
Uemb   = embed_qubit_unitary(U_corr, d=3, xp_mod=xp)

def make_rho1_d_corrected(rho_d):
    rho_d = to_xp(rho_d, xp)
    rho_d_corr = Uemb @ rho_d @ Uemb.conj().T
    return make_rho1_d(rho_d_corr)

R_corr, leak_corr = qubit_channel_from_d_evolution(make_rho1_d_corrected)

# Average the stochastic channel over trajectories, then isolate & correct
def make_rho1_one_traj_factory(EC, Em, n_cut, tmax, dt, tau_c, sigma_ng, n_g0=0.0, d=3):
    def make_rho1_one_traj(rho_d):
        # draw a pure-state decomposition of rho_d to propagate with your pure-state loop, then mix
        evals, evecs = xp.linalg.eigh(rho_d)
        outs = []
        for k in range(d):
            pk = float(xp.real(evals[k]))
            if pk > 1e-12:
                psi = evecs[:,k]
                rho_out = evolve_stochastic_ng_pure(psi, EC, Em, n_cut, tmax, dt, tau_c, sigma_ng, n_g0)
                outs.append(pk * rho_out)
        return sum(outs) if outs else rho_d
    return make_rho1_one_traj

make_one = make_rho1_one_traj_factory(EC, Em, n_cut, tmax=2e-6, dt=2e-9, tau_c=50e-6, sigma_ng=2e-4, n_g0=0.0, d=3)
E_avg = average_channel_over_traj(make_one, N=16)

def make_rho1_d_avg(rho_d):
    v_out = E_avg(vecN(rho_d))
    return unvecN(v_out)

Rst, leak_st = qubit_channel_from_d_evolution(make_rho1_d_avg)
iso_st = isolate_coherent_from_R(Rst)
Ucorr_st = correction_unitary_from_RU(iso_st["RU"])
noise_st = coarse_incoherent_params(iso_st["R_incoh"], leak_st)
print("stochastic angles_zyz (rad):", iso_st["angles_zyz"])
print("stochastic incoherent:", noise_st)

# %%
def to_np(a):  # safe caster for cupy/numpy
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray): return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)

def bloch_block(R):   # 3x3 A from 4x4 R
    return to_np(R)[1:,1:]

def plot_bloch_map(R, title="Bloch map A (X,Y,Z → X',Y',Z')"):
    A = bloch_block(R)
    plt.figure()
    plt.imshow(A, vmin=-1, vmax=1)
    plt.title(title)
    plt.xticks([0,1,2], ["X","Y","Z"])
    plt.yticks([0,1,2], ["X'","Y'","Z'"])
    plt.colorbar()
    plt.show()

def plot_singular_values(R, title="Bloch singular values"):
    s = np.linalg.svd(bloch_block(R), compute_uv=False)
    plt.figure()
    plt.bar(["σ₁","σ₂","σ₃"], s)
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.show()

def plot_translation(t_vec, title="Bloch translation t"):
    t = to_np(t_vec).ravel()
    plt.figure()
    try:
        plt.stem(["tₓ","t_y","t_z"], t, use_line_collection=True)
    except TypeError:
        plt.stem(["tₓ","t_y","t_z"], t)
    plt.title(title)
    plt.show()

def plot_rotation_axis_angle(R, title="Residual coherent rotation"):
    A = bloch_block(R)
    ang = float(np.arccos(np.clip((np.trace(A)-1)/2, -1, 1)))
    axis = np.array([A[2,1]-A[1,2], A[0,2]-A[2,0], A[1,0]-A[0,1]])
    axis = axis / (np.linalg.norm(axis) + 1e-18)
    print(f"θ = {ang:.3e} rad, axis n = [{axis[0]:.3e}, {axis[1]:.3e}, {axis[2]:.3e}]")
    plt.figure()
    plt.quiver(0, 0, axis[0], axis[1], angles='xy', scale_units='xy', scale=1)
    plt.xlim(-1,1); plt.ylim(-1,1); plt.gca().set_aspect('equal','box')
    plt.title(title + " (XY projection)")
    plt.show()

def plot_before_after(R_before, R_after):
    Ab, Aa = bloch_block(R_before), bloch_block(R_after)
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1); plt.imshow(Ab, vmin=-1, vmax=1); plt.title("A before"); plt.colorbar()
    plt.subplot(1,2,2); plt.imshow(Aa, vmin=-1, vmax=1); plt.title("A after");  plt.colorbar()
    plt.tight_layout(); plt.show()

def plot_leakage(leaks, title="Leakage per probe (pop in {|0,1⟩})"):
    labels = ["|0⟩","|1⟩","|+⟩","|+i⟩"]
    y = to_np(leaks)
    plt.figure()
    plt.bar(labels[:len(y)], y)
    plt.ylim(0,1.05)
    plt.title(f"{title} — avg={y.mean():.3f}")
    plt.show()

def plot_shrink_vs_dt(dts, build_R_for_segment_fn, title="Shrink vs segment time"):
    sxy, sz = [], []
    for dt_seg in dts:
        R_dt, t_dt, _, _ = build_R_for_segment_fn(dt_seg)
        s = np.linalg.svd(bloch_block(R_dt), compute_uv=False)
        sxy.append(0.5*(s[0]+s[1]))
        sz.append(s[2])
    dtu = to_np(dts)*1e6
    plt.figure()
    plt.step(dtu, sxy, where='post', label="⟨σₓ,σ_y⟩")
    plt.step(dtu, sz,  where='post', label="σ_z")
    plt.xlabel("segment Δt (µs)"); plt.ylabel("shrink factor"); plt.ylim(0,1.05)
    plt.title(title); plt.legend(); plt.show()

def plot_bloch_ellipsoid(R, title="Bloch ellipsoid"):
    A = bloch_block(R)
    phi = np.linspace(0, 2*np.pi, 80)
    theta = np.linspace(0, np.pi, 40)
    xs, ys, zs = [], [], []
    for th in theta:
        c, s = np.cos(th), np.sin(th)
        for ph in phi:
            v = np.array([s*np.cos(ph), s*np.sin(ph), c])
            w = A @ v
            xs.append(w[0]); ys.append(w[1]); zs.append(w[2])
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, s=3)
    ax.set_title(title)
    plt.show()

# %%
R, t, leaks, leak_avg = build_R_for_segment_stochastic(
    EC, Em, n_cut=10, dt_seg=2e-6, dt_step=2e-9, tau_c=50e-6, sigma_ng=2e-4, averaged=True, N=16
)
plot_bloch_map(R)
plot_singular_values(R)
plot_translation(t)
plot_rotation_axis_angle(R)
plot_leakage(leaks)

dts = np.linspace(0.2e-6, 8e-6, 12)
def build_for_dt(dt_seg):
    return build_R_for_segment_stochastic(EC, Em, 10, dt_seg, 2e-9, 50e-6, 1e-2, averaged=True, N=16)
plot_shrink_vs_dt(dts, build_for_dt)

# %%
def collapse_ops_qubit_T1_Tphi(T1=None, Tphi=None, d=3, xp_mod=xp):
    Ls = []
    if T1 is not None and T1 > 0:
        gamma1 = 1.0 / T1
        L10 = xp_mod.zeros((d,d), dtype=xp_mod.complex128)
        L10[0,1] = xp_mod.sqrt(gamma1)
        Ls.append(L10)
    if Tphi is not None and Tphi > 0:
        gamma_phi = 1.0 / Tphi
        Lphi = xp_mod.zeros((d,d), dtype=xp_mod.complex128)
        Lphi[1,1] =  1.0
        Lphi[0,0] = -1.0
        Lphi *= xp_mod.sqrt(gamma_phi/2.0)
        Ls.append(Lphi)
    return Ls

def collapse_ops_multilevel_T1s(T1_10=None, T1_21=None, Tphi=None, d=3, xp_mod=xp):
    Ls = collapse_ops_qubit_T1_Tphi(T1_10, Tphi, d=d, xp_mod=xp_mod)
    if T1_21 is not None and T1_21 > 0 and d >= 3:
        g21 = 1.0 / T1_21
        L21 = xp_mod.zeros((d,d), dtype=xp_mod.complex128)
        L21[1,2] = xp_mod.sqrt(g21)
        Ls.append(L21)
    return Ls

def ohmic_psd(omega, eta=1.0, wc=2*np.pi*5e9):
    w = abs(float(omega))
    return 2*eta*w / (1 + (w/wc)**2)

def collapse_ops_from_n_psd(evals, n_mat, psd_fn=ohmic_psd, d=3, xp_mod=xp):
    Ls = []
    for m in range(d):
        for n in range(m):
            omega_mn = evals[m] - evals[n]
            rate = psd_fn(omega_mn) * float(abs(n_mat[m,n])**2)
            if rate > 0:
                L = xp_mod.zeros((d,d), dtype=xp_mod.complex128)
                L[n,m] = xp_mod.sqrt(rate)
                Ls.append(L)
    n_diag = xp_mod.diag(n_mat)
    n0 = float(xp_mod.real(n_diag[0]))
    for m in range(1, min(d,3)):
        delta = float(xp_mod.real(n_diag[m]) - n0)
        rate_phi = psd_fn(0.0) * (delta**2)
        if rate_phi > 0:
            Lm = xp_mod.zeros((d,d), dtype=xp_mod.complex128)
            Lm[m,m] = xp_mod.sqrt(rate_phi/2.0)
            L0 = xp_mod.zeros((d,d), dtype=xp_mod.complex128)
            L0[0,0] = -xp_mod.sqrt(rate_phi/2.0)
            Ls += [Lm, L0]
    return Ls

# %%
H3, ns = transmon_H(EC, Em, n_cutoff=n_cut, n_g=0.0)
evals, U = diagnolize_H(H3)
Es = evals[:3]
H_diag = xp.diag(Es)

Ls = collapse_ops_qubit_T1_Tphi(T1=20e-6, Tphi=30e-6, d=3, xp_mod=xp)

R, t, leaks, leak_avg = build_R_for_segment_lindblad(H=H_diag, Ls=Ls, dt_seg=2e-6, dt_step=2e-9)
print("Lindblad channel leakage (avg pop in {|0>,|1>} for |0>,|1>,|+>,|+i>):", leak_avg)

def evolve_stochastic_ng_lindblad(rho0, EC, Em, n_cut, dt_seg, dt_step, tau_c, sigma_ng, Ls, n_g0=0.0):
    t = 0.0; n_g = n_g0; rho = rho0.copy()
    rng = xp.random.default_rng()
    while t < dt_seg - 1e-18:
        Ht_full, ns = transmon_H(EC, Em, n_cutoff=n_cut, n_g=n_g)
        evals, U = diagnolize_H(Ht_full)
        Ht = xp.diag(evals[:rho.shape[0]])  # 3×3
        def Ht_fn(tt,*args): return Ht
        rho = rk4_step(rho, t, dt_step, Ht_fn, tuple(), Ls)
        n_g = ou_step(n_g, dt_step, tau_c, sigma_ng, rng)
        t += dt_step
    return rho

def make_rho1_one_traj_dm_factory(EC, Em, n_cut, dt_seg, dt_step, tau_c, sigma_ng, Ls, n_g0=0.0, d=3):
    def make_rho1(rho_d):
        return evolve_stochastic_ng_lindblad(rho_d, EC, Em, n_cut, dt_seg, dt_step, tau_c, sigma_ng, Ls, n_g0)
    return make_rho1

def bloch_block(R):
    return to_np(R)[1:,1:]

def theory_shrink(dt, T1, Tphi):
    T2 = 1.0 / (0.5/T1 + 1.0/Tphi) if (T1 and Tphi) else (
          2*T1 if (T1 and not Tphi) else np.inf)
    s_xy = np.exp(-dt / T2)
    s_z  = np.exp(-dt / T1) if T1 else np.ones_like(dt)
    return s_xy, s_z

def theory_translation_tz(dt, T1):
    return 1.0 - np.exp(-dt / T1)

def sweep_channel_vs_dt(dts, build_R_for_dt, T1=None, Tphi=None, title="Shrink vs segment time"):
    sxy_num, sz_num, tz_num = [], [], []
    for dt in dts:
        R, t_vec, _, _ = build_R_for_dt(dt)
        A = bloch_block(R)
        s = np.linalg.svd(A, compute_uv=False)
        sxy_num.append(0.5*(s[0]+s[1]))
        sz_num.append(s[2])
        tz_num.append(to_np(t_vec)[2])

    dtu = to_np(dts)*1e6
    sxy_num = np.array(sxy_num); sz_num = np.array(sz_num); tz_num = np.array(tz_num)

    if T1 is not None:
        sxy_th, sz_th = theory_shrink(to_np(dts), T1, Tphi if Tphi else np.inf)
        tz_th = theory_translation_tz(to_np(dts), T1)
    else:
        sxy_th = sz_th = tz_th = None

    plt.figure()
    plt.step(dtu, sxy_num, where='post', label="$\langle \\sigma_x,\\sigma_y\\rangle$ (num)")
    plt.step(dtu, sz_num,  where='post', label="$\\sigma_z$ (num)")
    if sxy_th is not None:
        plt.plot(dtu, sxy_th, label="$\langle \\sigma_x,\\sigma_y\\rangle$ (theory)")
        plt.plot(dtu, sz_th,  label="$\\sigma_z$ (theory)")
    plt.xlabel("segment $\\Delta t$ ($\\mu s$)"); plt.ylabel("shrink factor"); plt.ylim(0, 1.05)
    plt.title(title); plt.legend(); plt.show()

    plt.figure()
    plt.step(dtu, tz_num, where='post', label="$t_z$ (num)")
    if sxy_th is not None:
        plt.plot(dtu, tz_th, label="$t_z$ (theory)")
    plt.xlabel("segment $\\Delta t$ ($\\mu s$)"); plt.ylabel("$t_z$"); plt.ylim(-0.05, 1.05)
    plt.title("Bloch translation $t_z$ (amplitude damping)"); plt.legend(); plt.show()
def sweep_leakage_vs_dt(dts, build_R_for_dt, title="Leakage vs segment time"):
    avg_leak, max_leak = [], []
    for dt in dts:
        _, _, leaks, leak_avg = build_R_for_dt(dt)
        leaks = to_np(leaks)
        avg_leak.append(leak_avg)
        max_leak.append(leaks.max())
    dtu = to_np(dts)*1e6
    plt.figure()
    plt.step(dtu, avg_leak, where='post', label="avg pop in {$|0,1\\rangle$}")
    plt.step(dtu, max_leak, where='post', label="max probe pop in {$|0,1\\rangle$}")
    plt.xlabel("segment $\\Delta t ($\\mu s$)$"); plt.ylabel("population retained (0-1)")
    plt.title(title); plt.ylim(0, 1.05); plt.legend(); plt.show()


# %%
T1  = 20e-6
Tphi = 30e-6
def build_for_dt(dt_seg):
    return build_R_for_segment_lindblad(H=xp.diag(Es), Ls=collapse_ops_qubit_T1_Tphi(T1, Tphi, d=3, xp_mod=xp),
                                        dt_seg=dt_seg, dt_step=2e-9)

dts = np.linspace(0.2e-6, 20e-6, 20)
sweep_channel_vs_dt(dts, build_for_dt, T1=T1, Tphi=Tphi)
sweep_leakage_vs_dt(dts, build_for_dt)

T1, Tphi = 20e-6, 30e-6
Ls = collapse_ops_qubit_T1_Tphi(T1, Tphi, d=3, xp_mod=xp)

# def build_for_dt(dt_seg):
#     make_one = make_rho1_one_traj_dm_factory(EC, Em, n_cut=10, dt_seg=dt_seg, dt_step=2e-9,
#                                              tau_c=50e-6, sigma_ng=2e-4, Ls=Ls, n_g0=0.0)
#     make_avg = average_channel_over_trajectories(make_one, N=16)
#     return channel_from_d_evolution(make_avg)

# dts = np.linspace(0.2e-6, 20e-6, 20)
# sweep_channel_vs_dt(dts, build_for_dt, T1=T1, Tphi=Tphi)
# sweep_leakage_vs_dt(dts, build_for_dt)

# %% [markdown]
# ## Quantum Transport and DOS Investigation
# Now that we have built a formalisim for a TB Transmon Qubit Hamiltonian and proven its accuracy and stability, we can utilize the TB model to look into the properties of Quantum Transport and Density of States in the Transmon Qubit.
# 
# ### Using KWANT:
# We can use KWANT to build a spatially resolved Local DOS, and linear-response transport of a single particle model of the junction. We can build the TB Hamiltonian in particle-hole space, double the basis, and add onsite pairing:
# \begin{equation*}
#     H = \begin{pmatrix}
#     H_0 - \mu & \Delta(r) \\
#     \Delta^*(r) & -[H_0 - \mu]^T
#     \end{pmatrix}, \quad \Delta = \begin{cases} 
#     \Delta_L e^{i\theta},& \mathrm{~left~lead}\\
#     \Delta_R e^{i\varphi},& \mathrm{~right~lead}\\
#     0 & \mathrm{~constriction}
#     \end{cases}
# \end{equation*}
# We attach the superconducting leads with phases $\theta$ and $\varphi$, then compute the following:
# - LSO/DOS through: $\rho(\mathbf{r},E) = \frac{-1}{\pi}\Im [\mathrm{Tr}[G^r(r,r',E)]]$.
# - Subgap Andreev bound states by either treating the device as a short finite wire or scanning energy and looking for peaks in LDOS. 
# - Phase-dependent spectra through sweeping $\varphi$.
# 
# The issue with KWANT is that there is no $\Delta$ self-consistency, which is good for linear transport but not for transmon level quantization.
# 
# ### Non-Equilibrium Green's Functions (NEGF) + Equilibrium Josephson
# We use NEGF when we want currents, CPR, DOS with explicit lead self-energies, equilibrium or finite bias. We use the same device from KWANT with the same lead self-energies. We then compute:
# - Retarded Green's Functions: $G^r(E) = [E^+ - H_D - \Sigma_L^r - \Sigma_R^r]^{-1}$.
# - DOS from the same $\rho$ function shown above. 
# - Josephson Current at equilibrium: $I(\varphi) = \frac{2e}{\hbar} \int \frac{dE}{2\pi} \tanh\left(\frac{E}{2k_B T} \right) \Im[\mathrm{Tr}[\tau_z \Sigma_L^r G^r]]$. 

# %%
import kwant
import importlib, kwant
kwant = importlib.reload(kwant)
from kwant import ldos as kwant_ldos
tau0 = np.eye(2)
taux = np.array([[0,1],[1,0]], dtype=np.complex128)
tauy = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
tauz = np.array([[1,0],[0,-1]], dtype=np.complex128)

def bdg_onsite(mu,t,Delta=0.0,phase=0.0):
    H0 = (0.0 - mu)
    Re, Im = Delta * np.cos(phase), Delta * np.sin(phase)
    M = H0*tauz + Re*taux - Im*tauy
    M = 0.5 * (M + M.conj().T)
    return M

def bdg_hop(t):
    M = -t * tauz
    M = 0.5 * (M + M.conj().T)
    return M

def onsite_left(site, mu, t, Delta, phi):
    return bdg_onsite(mu, t, Delta=Delta, phase=0.0)

def onsite_right(site, mu, t, Delta, phi):
    return bdg_onsite(mu, t, Delta=Delta, phase=phi)


def make_JJ_WL(W=30,L=30,a=1.0,t=1.0,mu=2.0,Delta=0.2,barrier=0.0):
    lat = kwant.lattice.square(a,norbs=2)
    syst = kwant.Builder()

    def in_central(pos):
        x,y = pos
        return (0<=x<L) and (0<=y<W)
    
    for x in range(0,L):
        for y in range(0,W):
            onsite_shift = barrier if (L//3 <= x < 2*L//3) else 0.0
            syst[lat(x,y)] = bdg_onsite(mu+onsite_shift,t,Delta,phase=0.0)

    for x in range(0,L):
        for y in range(0,W):
            if x+1 < L: syst[lat(x,y), lat(x+1,y)] = bdg_hop(t)
            if y+1 < W: syst[lat(x,y), lat(x,y+1)] = bdg_hop(t)

    sym_left  = kwant.TranslationalSymmetry((-1, 0))
    left_lead = kwant.Builder(sym_left)
    for y in range(W):
        left_lead[lat(0, y)] = onsite_left
    for y in range(W-1):
        left_lead[lat(0, y+1), lat(0, y)] = bdg_hop(t)
    left_lead[lat(0, 0), lat(-1, 0)] = bdg_hop(t)
    syst.attach_lead(left_lead)

    sym_right = kwant.TranslationalSymmetry((1, 0))
    right_lead = kwant.Builder(sym_right)
    for y in range(W):
        right_lead[lat(0, y)] = onsite_right
    for y in range(W-1):
        right_lead[lat(0, y+1), lat(0, y)] = bdg_hop(t)
    right_lead[lat(0, 0), lat(1, 0)] = bdg_hop(t)
    syst.attach_lead(right_lead)
    fsys = syst.finalized()

    constriction_sites = [s for s in fsys.sites if (0 <= s.tag[0]<L) and (W//3 <= s.tag[1] < 2*W//3)]
    center_line = [s for s in fsys.sites if (0<= s.tag[0]<L) and (s.tag[1] == W//2)]
    tags = {"constriction": constriction_sites, "center_line": center_line}
    return fsys, tags

import warnings

def ldos_average(fsys, E, phi, region_sites, mu=2.0, t=1.0, Delta=0.2,
                 broadening=1e-9, retry_broadening=1e-6):
    try:
        ldos_array = kwant_ldos(fsys, energy=E, args=(mu, t, Delta, phi), check_hermiticity=False)
    except RuntimeError as exc:
        warnings.warn(f"kwant.ldos failed at E={E:.6g}, phi={phi:.3g}: {exc}. Retrying with broadening={broadening}.")
        try:
            ldos_array = kwant_ldos(fsys, energy=E + 1j * broadening, args=(mu, t, Delta, phi), check_hermiticity=False)
        except RuntimeError:
            warnings.warn(f"Retry with broadening={broadening} failed; retrying with larger broadening={retry_broadening}.")
            try:
                ldos_array = kwant_ldos(fsys, energy=E + 1j * retry_broadening, args=(mu, t, Delta, phi), check_hermiticity=False)
            except RuntimeError as exc2:
                warnings.warn(f"kwant.ldos persistent failure at E={E:.6g}, phi={phi:.3g}: {exc2}. Returning NaN for this point.")
                return float(np.nan)

    id_map = fsys.id_by_site
    vals = []
    for s in region_sites:
        sid = id_map.get(s, None)
        if sid is None:
            warnings.warn(f"site {s} not found in finalized system id map; skipping.")
            continue
        vals.append(ldos_array[sid])
    if not vals:
        return float(np.nan)
    return float(np.mean(vals))


def abs_map_vs_phase(fsys, tags, phis, E_vals, mu=2.0, t=1.0, Delta=0.2, broadening=1e-9):
    grid = np.zeros((len(phis), len(E_vals)), dtype=float)
    region = tags["constriction"]
    for i, phi in enumerate(phis):
        for j, E in enumerate(E_vals):
            grid[i, j] = ldos_average(fsys, E, phi, region, mu=mu, t=t, Delta=Delta,
                                      broadening=broadening)
    return grid

def plot_abs_map(phis, energies, ldos_grid, Delta):
    plt.figure()
    extent = [energies[0], energies[-1], phis[0], phis[-1]]
    plt.imshow(ldos_grid, aspect='auto', origin='lower', extent=extent,
               vmin=np.percentile(ldos_grid, 5), vmax=np.percentile(ldos_grid, 95))
    plt.xlabel("Energy E (arb.)"); plt.ylabel("Phase $\\varphi$ (rad)")
    plt.title("ABS intensity (LDOS avg in constriction)")
    plt.colorbar(label="LDOS")
    plt.hlines([0], xmin=energies[0], xmax=energies[-1], linestyles='dotted')
    plt.show()
    
def plot_ldos_linecut(fsys, tags, E, phi, mu=2.0, t=1.0, Delta=0.2, check_hermiticity=False):
    import warnings
    line = tags["center_line"]
    xs = [s.tag[0] for s in line]
    ldos_array = kwant_ldos(fsys, energy=E, args=(mu, t, Delta, phi), check_hermiticity=check_hermiticity)
    id_map = fsys.id_by_site
    vals = []
    for s in line:
        sid = id_map.get(s, None)
        if sid is None:
            warnings.warn(f"site {s} not found in finalized system id map; skipping.")
            continue
        vals.append(ldos_array[sid])

    if len(vals) == 0:
        raise ValueError("No LDOS values collected for linecut (no matching sites).")

    vals = np.array(vals)
    order = np.argsort(xs)
    plt.figure()
    plt.step(np.array(xs)[order], vals[order], where='mid')
    plt.xlabel("$x$ (sites)"); plt.ylabel("LDOS")
    plt.title(f"LDOS linecut at E={E:.3g}, $\\varphi$={phi:.2f}")
    plt.show()

# %%
# fsys, tags = make_JJ_WL(W=40, L=50, a=1.0, t=1.0, mu=2.5, Delta=0.3, barrier=0.6)
# _ = kwant.ldos(fsys, energy=0.0, args=(2.5, 1.0, 0.3, 0.0))
# phis = np.linspace(0, 2*np.pi, 81)
# energies = np.linspace(-0.6, 0.6, 401)
# ldos_grid = abs_map_vs_phase(fsys, tags, phis, energies, mu=2.5, t=1.0, Delta=0.3)
# plot_abs_map(phis, energies, ldos_grid, Delta=0.3)
# plot_ldos_linecut(fsys, tags, E=0.05, phi=np.pi/2, mu=2.5, t=1.0, Delta=0.3)

# %% [markdown]
# Alright so, unfortunetly KWANT isn't fit to model this device. Let's just move onto modeling the NEGF.

# %%
tau0 = np.eye(2)
taux = np.array([[0, 1],[1, 0]], complex)
tauy = np.array([[0,-1j],[1j, 0]], complex)
tauz = np.array([[1, 0],[0,-1 ]], complex)

Delta = 0.3
ETA   = 3e-3*Delta
kBT   = 0.05*Delta
fE    = lambda E: np.tanh(E/(2*kBT))

def make_device_chain(N=5, t=1.0, mu=2.5):
    H0   = -mu*tauz
    Hhop = -t*tauz
    H = np.kron(np.eye(N), H0)
    off = np.kron(np.diag(np.ones(N-1), 1), Hhop)
    H += off + off.conj().T
    return H

def embed_sigmas_chain(N, SigmaL2, SigmaR2):
    dim = 2*N
    SL = np.zeros((dim, dim), dtype=complex)
    SR = np.zeros((dim, dim), dtype=complex)
    SL[0:2, 0:2] = SigmaL2
    i0 = 2*(N-1)
    SR[i0:i0+2, i0:i0+2] = SigmaR2
    return SL, SR

def green_ret(E, H, SigmaL, SigmaR, eta=ETA):
    z = E + 1j*eta
    return np.linalg.inv(z*np.eye(H.shape[0]) - H - SigmaL - SigmaR)

def super_sigma(E, Delta, phi, Gamma, eta=ETA):
    z = E + 1j*eta
    root = np.lib.scimath.sqrt(Delta**2 - z**2)
    Delta_mat = Delta*(np.cos(phi)*taux - np.sin(phi)*tauy)
    return -1j * Gamma * (z*tau0 + Delta_mat) / root

def josephson_current(H, Delta, Gamma, phi, Egrid, fE, eta=ETA):
    N = H.shape[0] // 2
    tauzN = np.kron(np.eye(N), tauz)
    IE_L, IE_R = [], []
    for E in Egrid:
        SigmaL2 = super_sigma(E, Delta, 0.0, Gamma, eta)
        SigmaR2 = super_sigma(E, Delta, phi,  Gamma, eta)
        SigmaL, SigmaR = embed_sigmas_chain(N, SigmaL2, SigmaR2)

        G = green_ret(E, H, SigmaL, SigmaR, eta)
        IE_L.append(np.imag(np.trace(tauzN @ (SigmaL @ G))) * fE(E))
        IE_R.append(np.imag(np.trace(tauzN @ (SigmaR @ G))) * fE(E))
    return 0.5*(np.trapezoid(IE_L, Egrid) - np.trapezoid(IE_R, Egrid))

def current_phase(mu=2.5, t=1.0, Delta=0.3, Gamma=0.2,
                  Egrid=None, phis=None, eta=ETA, Nchain=5):
    if Egrid is None:
        Egrid = np.linspace(-2*Delta, 2*Delta, 8001)
    if phis is None:
        phis = np.linspace(0, 2*np.pi, 181)

    H = make_device_chain(N=Nchain, t=t, mu=mu)
    N = H.shape[0] // 2
    Iphi, dos_phi = [], []

    for phi in phis:
        Iphi.append(josephson_current(H, Delta, Gamma, phi, Egrid, fE, eta))
        DE = []
        for E in Egrid:
            SigmaL2 = super_sigma(E, Delta, 0.0, Gamma, eta)
            SigmaR2 = super_sigma(E, Delta, phi,  Gamma, eta)
            SigmaL, SigmaR = embed_sigmas_chain(N, SigmaL2, SigmaR2)

            G = green_ret(E, H, SigmaL, SigmaR, eta)
            DE.append(-np.imag(np.trace(G))/np.pi)
        dos_phi.append(np.trapezoid(DE, Egrid))

    return np.array(phis), np.array(Iphi), np.array(dos_phi)

phis, Iphi, dos_phi = current_phase(mu=2.5, t=1.0, Delta=Delta, Gamma=0.2, eta=ETA, Nchain=5)

plt.figure()
plt.plot(phis, dos_phi/np.max(dos_phi))
plt.xlabel("phase φ"); plt.ylabel("DOS (norm.)")
plt.title("Total DOS vs φ")
plt.show()

I0 = Iphi - Iphi.mean()
In = I0 / (np.max(np.abs(I0)) + 1e-15)
plt.figure()
plt.plot(phis, In)
plt.xlabel("phase φ"); plt.ylabel("Josephson current (norm.)")
plt.title("Josephson CPR (de-biased)")
plt.show()

# %%
def local_dos_map(H, Delta, Gamma, phi, E, eta=ETA):
    N = H.shape[0] // 2
    SigmaL2 = super_sigma(E, Delta, 0.0, Gamma, eta)
    SigmaR2 = super_sigma(E, Delta, phi,  Gamma, eta)
    SigmaL, SigmaR = embed_sigmas_chain(N, SigmaL2, SigmaR2)
    G = green_ret(E, H, SigmaL, SigmaR, eta)
    LDOS = -np.imag(np.diag(G)) / np.pi
    return LDOS[::2] + LDOS[1::2]

def transmission(H, Delta, Gamma, phi, Egrid, eta=ETA):
    N = H.shape[0] // 2
    T_E = []
    for E in Egrid:
        SigmaL2 = super_sigma(E, Delta, 0.0, Gamma, eta)
        SigmaR2 = super_sigma(E, Delta, phi,  Gamma, eta)
        SigmaL, SigmaR = embed_sigmas_chain(N, SigmaL2, SigmaR2)
        G = green_ret(E, H, SigmaL, SigmaR, eta)
        GammaL = 1j*(SigmaL - SigmaL.conj().T)
        GammaR = 1j*(SigmaR - SigmaR.conj().T)
        T_E.append(np.real(np.trace(GammaL @ G @ GammaR @ G.conj().T)))
    return np.array(Egrid), np.array(T_E)

plt.figure()
H = make_device_chain(N=5, t=1.0, mu=2.5)
Egrid = np.linspace(-0.6, 0.6, 1001)
E_vals, T_E = transmission(H, Delta, Gamma=0.2, phi=np.pi/2, Egrid=Egrid, eta=ETA)
plt.plot(E_vals, T_E)
plt.xlabel("Energy E (arb.)"); plt.ylabel("Transmission T(E)")
plt.title("Transmission vs Energy at $\\varphi=\\pi/2$")
plt.show()
plt.close()

plt.figure()
plt.colormaps['viridis']
ldos_map = local_dos_map(H, Delta, Gamma=0.2, phi=np.pi/2, E=0.0, eta=ETA)
plt.step(range(len(ldos_map)), ldos_map, where='mid')
plt.xlabel("Site index")
plt.ylabel("Local DOS")
plt.title("Local DOS at E=0, $\\varphi=\\pi/2$")
plt.show()

# %%
def embed_sigmas_chain(N, SigmaL2, SigmaR2):
    dim = 2*N
    SL = np.zeros((dim, dim), dtype=complex)
    SR = np.zeros((dim, dim), dtype=complex)
    SL[0:2, 0:2] = SigmaL2
    i0 = 2*(N-1)
    SR[i0:i0+2, i0:i0+2] = SigmaR2
    return SL, SR

def tauz_big(N):
    return np.kron(np.eye(N), tauz)

def grid_spectral_current(H, Delta, Gamma, phis, Egrid, kBT, eta=ETA):
    N = H.shape[0] // 2
    tauzN = tauz_big(N)
    tanh = (lambda E: np.tanh(E/(2*kBT))) if kBT > 0 else (lambda E: np.sign(E))
    J = np.zeros((len(phis), len(Egrid)))
    for ip, phi in enumerate(phis):
        for ie, E in enumerate(Egrid):
            SigmaL2 = super_sigma(E, Delta, 0.0, Gamma, eta)
            SigmaR2 = super_sigma(E, Delta, phi,  Gamma, eta)
            SigmaL, SigmaR = embed_sigmas_chain(N, SigmaL2, SigmaR2)
            G = green_ret(E, H, SigmaL, SigmaR, eta)
            J[ip, ie] = np.imag(np.trace(tauzN @ (SigmaL @ G))) * tanh(E)
    return J

def grid_transmission(H, Delta, Gamma, phis, Egrid, eta=ETA):
    N = H.shape[0] // 2
    T = np.zeros((len(phis), len(Egrid)))
    for ip, phi in enumerate(phis):
        for ie, E in enumerate(Egrid):
            SigmaL2 = super_sigma(E, Delta, 0.0, Gamma, eta)
            SigmaR2 = super_sigma(E, Delta, phi,  Gamma, eta)
            SigmaL, SigmaR = embed_sigmas_chain(N, SigmaL2, SigmaR2)
            G = green_ret(E, H, SigmaL, SigmaR, eta)
            GammaL = 1j*(SigmaL - SigmaL.conj().T)
            GammaR = 1j*(SigmaR - SigmaR.conj().T)
            T[ip, ie] = np.real(np.trace(GammaL @ G @ GammaR @ G.conj().T))
    return T

def grid_ldos_vs_E_and_x(H, Delta, Gamma, phi, Egrid, eta=ETA):
    N = H.shape[0] // 2
    LD = np.zeros((len(Egrid), N))
    for ie, E in enumerate(Egrid):
        SigmaL2 = super_sigma(E, Delta, 0.0, Gamma, eta)
        SigmaR2 = super_sigma(E, Delta, phi,  Gamma, eta)
        SigmaL, SigmaR = embed_sigmas_chain(N, SigmaL2, SigmaR2)
        G = green_ret(E, H, SigmaL, SigmaR, eta)
        local = -np.imag(np.diag(G))/np.pi
        LD[ie, :] = local[0::2] + local[1::2]
    return LD

def plot_heatmap(Z, xvals, yvals, xlabel, ylabel, title, percent_clip=1.0, cmap="viridis", filename='file'):
    Z = np.asarray(Z)
    vmin = np.percentile(Z, percent_clip)
    vmax = np.percentile(Z, 100 - percent_clip)
    plt.figure(figsize=(11,9))
    extent = [xvals[0], xvals[-1], yvals[0], yvals[-1]]
    plt.imshow(Z, aspect='auto', origin='lower', extent=extent, cmap=cmap,
               vmin=vmin, vmax=vmax)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title,fontsize=16)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)
    plt.show()

# %%
H = make_device_chain(N=5, t=1.0, mu=2.5)
phis  = np.linspace(0, 2*np.pi, 161)
Egrid = np.linspace(-2*Delta, 2*Delta, 1201)

J = grid_spectral_current(H, Delta, Gamma=0.2, phis=phis, Egrid=Egrid, kBT=kBT, eta=ETA)
plot_heatmap(J, Egrid, phis, xlabel="Energy $E$ (arb.)", ylabel="phase $\\varphi$",
             title="Spectral current density J($E$, $\\varphi$)", percent_clip=2.0, filename='spectral_current_density')

Iphi_from_J = np.array([np.trapezoid(J[i,:], Egrid) for i in range(len(phis))])
T = grid_transmission(H, Delta, Gamma=0.2, phis=phis, Egrid=Egrid, eta=ETA)
plot_heatmap(T, Egrid, phis, xlabel="Energy $E$ (arb.)", ylabel="phase $\\varphi$",
             title="Transmission T($E$, $\\varphi$)", percent_clip=2.0, filename='transmission')

phi_fixed = np.pi
LD = grid_ldos_vs_E_and_x(H, Delta, Gamma=0.2, phi=phi_fixed, Egrid=Egrid, eta=ETA)
x_sites = np.arange(H.shape[0]//2)
plot_heatmap(LD, x_sites, Egrid, xlabel="site index", ylabel="Energy $E$ (arb.)",
             title=f"LDOS($E$, $x$) at $\\varphi$ = {phi_fixed:.2f}", percent_clip=1.0, filename='ldos_fixed_phi')

region = slice(1, (H.shape[0]//2)-1)
LD_phiE = []
for phi in phis:
    LD = grid_ldos_vs_E_and_x(H, Delta, Gamma=0.2, phi=phi, Egrid=Egrid, eta=ETA)
    LD_phiE.append(LD[:, region].mean(axis=1))
LD_phiE = np.array(LD_phiE)
plot_heatmap(LD_phiE, Egrid, phis, xlabel="Energy $E$ (arb.)", ylabel="phase $\\varphi$",
             title="Average LDOS in constriction vs ($E$, $\\varphi$)", percent_clip=2.0, filename='average_ldos_constriction')

# %%
def make_device_chain_hetero(
    N=5, tL=1.0, tC=1.0, tR=1.0,
    muL=2.5, muC=2.5, muR=2.5,
    Vbar=0.0, i0=None, w=1,
    disorder_std=0.0, seed=None
):
    rng = np.random.default_rng(seed)
    mu = np.full(N, muC, dtype=float)
    t  = np.full(N-1, tC,  dtype=float)
    mu[0]     = muL
    mu[-1]    = muR
    t[0]      = tL
    t[-1]     = tR

    if i0 is None: i0 = N//2
    j0, j1 = max(0, i0 - w//2), min(N, i0 + (w+1)//2)
    mu[j0:j1] = mu[j0:j1] + Vbar
    if disorder_std > 0:
        mu[1:-1] += rng.normal(0.0, disorder_std, size=N-2)

    H = np.zeros((2*N, 2*N), complex)
    for i in range(N):
        H[2*i:2*i+2, 2*i:2*i+2] = -mu[i]*tauz
        if i+1 < N:
            H[2*i:2*i+2, 2*(i+1):2*(i+1)+2] = -t[i]*tauz
            H[2*(i+1):2*(i+1)+2, 2*i:2*i+2] = -t[i]*tauz
    return H

def build_sigmas(E, Delta, phi, Gamma, eta, N):
    ΣL2 = super_sigma(E, Delta, 0.0, Gamma, eta)
    ΣR2 = super_sigma(E, Delta, phi,  Gamma, eta)
    return embed_sigmas_chain(N, ΣL2, ΣR2)

def spectral_current_grid(H, Delta, Gamma, phis, Egrid, kBT, eta):
    N = H.shape[0]//2
    tauzN = np.kron(np.eye(N), tauz)
    tanh = (lambda E: np.tanh(E/(2*kBT))) if kBT>0 else (lambda E: np.sign(E))
    J = np.zeros((len(phis), len(Egrid)))
    for ip, phi in enumerate(phis):
        for ie, E in enumerate(Egrid):
            ΣL, ΣR = build_sigmas(E, Delta, phi, Gamma, eta, N)
            G = green_ret(E, H, ΣL, ΣR, eta)
            J[ip, ie] = np.imag(np.trace(tauzN @ (ΣL @ G))) * tanh(E)
    return J

def ldos_phiE_avg_constriction(H, Delta, Gamma, phis, Egrid, eta, region=None):
    N = H.shape[0]//2
    if region is None:
        region = slice(1, N-1)
    Z = np.zeros((len(phis), len(Egrid)))
    for ip, phi in enumerate(phis):
        for ie, E in enumerate(Egrid):
            ΣL, ΣR = build_sigmas(E, Delta, phi, Gamma, eta, N)
            G = green_ret(E, H, ΣL, ΣR, eta)
            local = -np.imag(np.diag(G))/np.pi
            ldos_sites = local[0::2] + local[1::2]
            Z[ip, ie] = ldos_sites[region].mean()
    return Z

# %%
Gamma = 1.25
ETA   = 1e-3*Delta
kBT   = 0.03*Delta
Egrid = np.linspace(-Delta*1.2, Delta*1.2, 500)
phis  = np.linspace(0, 2*np.pi, 500)

H = make_device_chain_hetero(N=3, tL=1.0, tC=1.0, tR=1.0, muL=1.5, muC=1.5, muR=1.5,
                             Vbar=0.0, disorder_std=0.0)

J = spectral_current_grid(H, Delta, Gamma, phis, Egrid, kBT, ETA)
# interpolate J, phi, and Egrid:
new_Egrid = np.linspace(Egrid[0], Egrid[-1],10000)
new_phis  = np.linspace(phis[0], phis[-1], 10000)
from scipy.interpolate import RegularGridInterpolator
interp_J = RegularGridInterpolator((phis, Egrid), J)
plot_heatmap(J, Egrid, phis, "Energy $E$ (arb.)", "phase $\\varphi$", "Spectral current $J(E, \\varphi)$", percent_clip=2.0,filename='spectral_current_density_hetero')
plot_heatmap(interp_J(np.array(np.meshgrid(new_phis, new_Egrid)).T.reshape(-1, 2)).reshape(len(new_phis), len(new_Egrid)), new_Egrid, new_phis, "Energy $E$ (arb.)", "phase $\\varphi$", "Interpolated spectral current $J(E, \\varphi)$", percent_clip=2.0,filename='spectral_current_density_hetero_interpolated')
LD_phiE = ldos_phiE_avg_constriction(H, Delta, Gamma, phis, Egrid, ETA)
interp_LD = RegularGridInterpolator((phis, Egrid), LD_phiE)
plot_heatmap(interp_LD(np.array(np.meshgrid(new_phis, new_Egrid)).T.reshape(-1, 2)).reshape(len(new_phis), len(new_Egrid)), new_Egrid, new_phis, "Energy $E$ (arb.)", "phase $\\varphi$", "Interpolated $\\braket{LDOS}$ in constriction vs (E, $\\varphi$)", percent_clip=2.0, filename='average_ldos_constriction_hetero_interpolated')
plot_heatmap(LD_phiE, Egrid, phis, "Energy $E$ (arb.)", "phase $\\varphi$", "$\\braket{LDOS}$ in constriction vs (E, $\\varphi$)", percent_clip=2.0, filename='average_ldos_constriction')

# %%
def extract_abs_from_grid(Z, Egrid, phis, prefer="min", Emin=1e-3, Emax=None):
    if Emax is None: Emax = 0.98*np.max(np.abs(Egrid))
    pos = np.where((Egrid > Emin) & (Egrid < Emax))[0]
    Epos = Egrid[pos]
    Eabs = np.zeros(len(phis))
    for i in range(len(phis)):
        row = Z[i, pos]
        idx = np.argmin(row) if prefer=="min" else np.argmax(row)
        Eabs[i] = Epos[idx]
    return Eabs

Eabs = extract_abs_from_grid(J, Egrid, phis, prefer="min", Emin=1e-3, Emax=Delta*0.98)
plt.figure(figsize=(10,9))
extent = [Egrid[0], Egrid[-1], phis[0], phis[-1]]
vmin, vmax = np.percentile(J, 2), np.percentile(J, 98)
plt.imshow(J, aspect='auto', origin='lower', extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
plt.plot(Eabs, phis, 'r-', lw=2, label="ABS (extracted)")
plt.plot(-Eabs, phis, 'r-', lw=2)
plt.xlabel("Energy $E$"); plt.ylabel("Phase $\\varphi$"); plt.title("$J(E, \\varphi)$ with ABS overlay")
plt.legend(); plt.tight_layout(); plt.show()

# %%
Iphi = np.array([josephson_current(H, Delta, Gamma, φ, Egrid, fE=lambda E: np.tanh(E/(2*kBT)), eta=ETA) for φ in phis])
I0   = Iphi - Iphi.mean()
Ephi = np.cumsum(np.trapezoid(I0, phis))
EJ   = 0.5*(Ephi.max() - Ephi.min())
print(f"Effective Josephson energy EJ (arb.): {EJ:.3e}")

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def _percentile_minmax(stack, pct):
    vmin = np.percentile(stack, pct)
    vmax = np.percentile(stack, 100 - pct)
    if np.isclose(vmin, vmax):
        pad = 1e-12 if vmax == 0 else 1e-3*abs(vmax)
        vmin, vmax = vmax - pad, vmax + pad
    return vmin, vmax

def _imshow_grid(maps, xvals, yvals, title, xlab, ylab, pct_clip=2.0, cmap="viridis"):
    nrows, ncols = len(maps), len(maps[0])
    all_data = np.concatenate([np.concatenate(row, axis=1) for row in maps], axis=0)
    vmin, vmax = _percentile_minmax(all_data, pct_clip)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0*ncols, 3.3*nrows), squeeze=False)
    extent = [xvals[0], xvals[-1], yvals[0], yvals[-1]]
    norm = Normalize(vmin=vmin, vmax=vmax)

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            im = ax.imshow(maps[i][j], aspect='auto', origin='lower',
                           extent=extent, cmap=cmap, norm=norm)
            # tidy labels
            if i == nrows - 1:
                ax.set_xlabel(xlab)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(ylab)
            else:
                ax.set_yticklabels([])
    fig.suptitle(title, y=0.97, fontsize=20, weight='bold')
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(),
                        shrink=0.95, pad=0.02)
    return fig, axes, cbar
def parameter_sweep_heatmaps(
    param1_vals, param2_vals, param1_name, param2_name,
    make_H_func,
    Delta, Gamma, Egrid, phis, kBT, eta,
    percent_clip=2.0
):
    J_maps = []
    LD_maps = []
    for p1 in param1_vals:
        J_row, LD_row = [], []
        for p2 in param2_vals:
            H  = make_H_func(p1, p2)
            J  = spectral_current_grid(H, Delta, Gamma, phis, Egrid, kBT, eta)
            LD = ldos_phiE_avg_constriction(H, Delta, Gamma, phis, Egrid, eta)
            J_row.append(J)
            LD_row.append(LD)
        J_maps.append(J_row)
        LD_maps.append(LD_row)

    def annotate_panels(axes):
        for i, p1 in enumerate(param1_vals):
            for j, p2 in enumerate(param2_vals):
                axes[i, j].set_title(f"{param1_name}={p1}, {param2_name}={p2}", fontsize=10)

    figJ, axesJ, cbarJ = _imshow_grid(
        J_maps, Egrid, phis,
        title=r"Spectral current density $J(E,\varphi)$",
        xlab="Energy $E$ (arb.)", ylab="phase $\\varphi$",
        pct_clip=percent_clip, cmap="viridis"
    )
    annotate_panels(axesJ)

    figL, axesL, cbarL = _imshow_grid(
        LD_maps, Egrid, phis,
        title=r"Average LDOS in constriction vs $(E,\varphi)$",
        xlab="Energy $E$ (arb.)", ylab="phase $\\varphi$",
        pct_clip=percent_clip, cmap="viridis"
    )
    annotate_panels(axesL)

    plt.show()
    return (figJ, axesJ, cbarJ), (figL, axesL, cbarL)

barrier_vals  = [0.0,0.25,0.5,0.75,1.0]
disorder_vals = [0.0, 0.025, 0.05, 0.075, 0.1]

def make_H_barrier_disorder(Vbar, disorder_std):
    return make_device_chain_hetero(
        N=5, tL=1.0, tC=1.0, tR=1.0,
        muL=1.5, muC=1.5, muR=1.5,
        Vbar=Vbar, disorder_std=disorder_std
    )

_ = parameter_sweep_heatmaps(
    barrier_vals, disorder_vals,
    "$V_{bar}$", "$\\sigma_{dis}$",
    make_H_barrier_disorder,
    Delta, Gamma=0.9, Egrid=np.linspace(-4*Delta, 4*Delta, 600),
    phis=np.linspace(0, 2*np.pi, 161),
    kBT=0.03*Delta, eta=1e-3*Delta,
    percent_clip=2.0
)

# %%
def make_device_from_ky(Nx=10, t_x=1.0, t_y=1.0, mu=2.5, ky=0.0):
    eps_ky = -(mu + 2*t_y*np.cos(ky))
    H0   = -eps_ky * tauz
    Hhop = -t_x * tauz
    H = np.kron(np.eye(Nx), H0)
    off = np.kron(np.diag(np.ones(Nx-1), 1), Hhop)
    H += off + off.conj().T
    return H 

def ky_spectral_map(
    Nx=10, t_x=1.0, t_y=1.0, mu=2.0,
    Delta=0.3, Gamma=0.5, phi=np.pi/2,
    ky_grid=None, Egrid=None, eta=1e-3, average_region=True
):
    if ky_grid is None: ky_grid = np.linspace(-np.pi, np.pi, 241)
    if Egrid  is None:  Egrid  = np.linspace(-1.2*Delta, 1.2*Delta, 1201)

    A_kE = np.zeros((len(ky_grid), len(Egrid)))

    for ik, ky in enumerate(ky_grid):
        H  = make_device_from_ky(Nx, t_x, t_y, mu, ky)
        N  = H.shape[0] // 2
        region = slice(1, N-1) if average_region else slice(0, N)
        for ie, E in enumerate(Egrid):
            ΣL2 = super_sigma(E, Delta, 0.0,  Gamma, eta)
            ΣR2 = super_sigma(E, Delta, phi,  Gamma, eta)
            ΣL, ΣR = embed_sigmas_chain(N, ΣL2, ΣR2)
            G  = green_ret(E, H, ΣL, ΣR, eta)
            local = -np.imag(np.diag(G))/np.pi
            ldos_sites = local[0::2] + local[1::2]
            A_kE[ik, ie] = np.mean(ldos_sites[region])
    return A_kE, ky_grid, Egrid

A_kE, ky_grid, Egrid = ky_spectral_map(
    Nx=14, t_x=1.0, t_y=1.0, mu=1.5, Delta=0.3, Gamma=0.6, phi=np.pi/2, eta=1e-3*0.3
)

plt.figure(figsize=(7.2,4.5))
vmin, vmax = np.percentile(A_kE, 2), np.percentile(A_kE, 98)
extent = [ky_grid[0], ky_grid[-1], Egrid[0], Egrid[-1]]
plt.imshow(A_kE.T, origin='lower', aspect='auto', extent=extent,
           vmin=vmin, vmax=vmax, cmap='viridis')
plt.xlabel(r'$k_y$'); plt.ylabel('Energy E (arb.)')
plt.title(r'$k_y$-resolved spectral function $A(k_y,E)$ at $\varphi=\pi/2$')
plt.colorbar(label='spectral weight')
plt.tight_layout(); plt.show()

DOS_E = np.trapezoid(A_kE, ky_grid, axis=0)
plt.figure(); plt.plot(Egrid, DOS_E/DOS_E.max()); plt.xlabel('E'); plt.ylabel('DOS (norm.)'); plt.show()

# %%
def make_sns_supercell_params(
    Nx=8, Ny=3,            # supercell size
    NL=3, NN=2, NR=3,      # left-S, normal, right-S widths (NL+NN+NR == Nx)
    t=1.0, muS=1.5, muN=1.5,
    Delta=0.3, phi=np.pi/2,
    Vbar=0.0,
    disorder_std=0.0, seed=None
):
    assert NL + NN + NR == Nx, "NL+NN+NR must equal Nx"
    rng = np.random.default_rng(seed)
    mu = np.full((Nx, Ny), muN, float)
    Delta_c = np.zeros((Nx, Ny), complex)
    mu[:NL, :] = muS
    Delta_c[:NL, :] = Delta*np.exp(-1j*phi/2)
    mu[NL:NL+NN, :] = muN + Vbar
    Delta_c[NL:NL+NN, :] = 0.0
    mu[NL+NN:, :] = muS
    Delta_c[NL+NN:, :] = Delta*np.exp(+1j*phi/2)

    if disorder_std > 0:
        mu += rng.normal(0.0, disorder_std, size=mu.shape)

    return dict(Nx=Nx, Ny=Ny, NL=NL, NN=NN, NR=NR,
                t=t, mu=mu, Delta_c=Delta_c, phi=phi)

def site_index(ix, iy, Nx, Ny):
    return iy + iy*0 + ix*Ny + 0 

def Hk_BdG(kx, ky, pars):
    Nx, Ny = pars["Nx"], pars["Ny"]
    t = pars["t"]; mu = pars["mu"]; Delta_c = pars["Delta_c"]
    Ns = Nx*Ny
    dim = 2*Ns
    H = np.zeros((dim, dim), complex)

    def put_block(i, j, B):
        H[2*i:2*i+2, 2*j:2*j+2] += B

    for ix in range(Nx):
        for iy in range(Ny):
            i = site_index(ix, iy, Nx, Ny)
            mu_ij = mu[ix, iy]
            Δ = Delta_c[ix, iy]
            onsite = -mu_ij * tauz + (Δ.real)*taux - (Δ.imag)*tauy
            put_block(i, i, onsite)

    hop = -t * tauz
    for ix in range(Nx):
        ixp = (ix + 1) % Nx
        phase_x = 1.0 if ix+1 < Nx else np.exp(1j*kx)
        for iy in range(Ny):
            i = site_index(ix,  iy, Nx, Ny)
            j = site_index(ixp, iy, Nx, Ny)
            put_block(i, j, phase_x * hop)
            put_block(j, i, np.conj(phase_x) * hop)

    for iy in range(Ny):
        iyp = (iy + 1) % Ny
        phase_y = 1.0 if iy+1 < Ny else np.exp(1j*ky)
        for ix in range(Nx):
            i = site_index(ix, iy,  Nx, Ny)
            j = site_index(ix, iyp, Nx, Ny)
            put_block(i, j, phase_y * hop)
            put_block(j, i, np.conj(phase_y) * hop)
    return H

def kpath_GXM(n_per_segment=60):
    Γ = np.array([0.0, 0.0])
    X = np.array([np.pi, 0.0])
    M = np.array([np.pi, np.pi])
    pts = [Γ, X, M, Γ]
    labels = [r"$\Gamma$", "X", "M", r"$\Gamma$"]
    klist = []
    for a,b in zip(pts[:-1], pts[1:]):
        for s in range(n_per_segment):
            klist.append(a + (b-a)*s/(n_per_segment-1))
    klist = np.array(klist)
    dk = np.linalg.norm(np.diff(klist, axis=0), axis=1)
    x = np.concatenate([[0], np.cumsum(dk)])
    ticks = [0]
    acc = 0.0
    for seg in range(3):
        acc += np.sum(dk[seg*(n_per_segment-1):(seg+1)*(n_per_segment-1)])
        ticks.append(acc)
    return klist, x, labels, ticks

def bandstructure(pars, n_bands_to_plot=None, n_per_segment=60):
    klist, kx_axis, labels, ticks = kpath_GXM(n_per_segment=n_per_segment)

    Nx, Ny = pars["Nx"], pars["Ny"]
    Ns  = Nx*Ny
    dim = 2*Ns
    if n_bands_to_plot is None or n_bands_to_plot > dim:
        n_bands_to_plot = dim

    evals = np.zeros((len(klist), n_bands_to_plot))
    for i, (kx,ky) in enumerate(klist):
        Hk = Hk_BdG(kx, ky, pars)
        w = np.linalg.eigvalsh(Hk)
        mid = dim//2
        if n_bands_to_plot < dim:
            half = n_bands_to_plot//2
            wsel = w[mid-half:mid+half] if n_bands_to_plot%2==0 else w[mid-half:mid+half+1]
        else:
            wsel = w
        evals[i,:len(wsel)] = wsel

    plt.figure(figsize=(6.0, 6.5))
    for n in range(n_bands_to_plot):
        plt.plot(kx_axis, evals[:, n], 'k-', lw=1.0)
    for t in ticks:
        plt.axvline(t, color='k', lw=0.6, ls='--')
    plt.axhline(0, color='k', ls='--', lw=0.8)
    plt.xticks(ticks, labels)
    plt.ylabel(r"$E - E_F$ (arb.)")
    plt.title("BdG Andreev minibands of an SNS superlattice")
    plt.savefig("sns_superlattice_bandstructure.png", dpi=300)
    plt.tight_layout()
    plt.show()


Delta = 0.3
phi   = np.pi/2
pars = make_sns_supercell_params(
    Nx=15, Ny=5, NL=5, NN=5, NR=5,
    t=1.0, muS=1.5, muN=1.5,
    Delta=Delta, phi=phi,
    Vbar=0.3, disorder_std=0.0
)
bandstructure(pars, n_bands_to_plot=24, n_per_segment=80)


