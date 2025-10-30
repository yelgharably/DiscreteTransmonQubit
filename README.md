# **Modeling Transmon Qubits as Tight-Binding Hamiltonians**
## QSEG610 - Engineering The Quantum Revolution Final Project
## Youssef El Gharably -- Fall 2025
### Instructors: Dr Garcia-Frias, Dr Vishal Saxena

Superconducting Qubits -especially Transmon qubits- are the workhorses of modern quantum processors. They are built from the simplest possible superconducting circuit, a Josephson Junction (JJ) shunted by a large capacitance. The configuration is able to make a nonlinear quantum oscillator whose lowest two energy levels form an effective qubit. 

The Transmon Qubit sits at the intersection of:
- Quantum Mechanics; it provides the Hamiltonian and its Quantization rules:
$$
    \hat{H} = 4E_C (\hat{n}-n_g)^2 - E_J \cos(\hat{\phi}),
$$
- Tight-Binding Model; it provides a mean of reducing the continous Schrodinger equation** to a discrete tight-binding eigenproblem solvable by matrix methods,
- Electronics; it connects the circuit parameters (capacitance, inductance, critical current) to the Quantum Energies $E_C$ and $E_J$ by desigining control/readout hardware that couples via voltage and flux lines.

### The Physical Picture of the Transmon Qubit as a Quantum Circuit:
A JJ behaves like a nonlinear, nondissipative inductor whose current-phase relation is $I(\phi) = I_C \sin(\phi)$. In circuit quantization, the Lagrangian of a node flux $\Phi=\frac{\Phi_0}{2\pi}\phi$ is:
$$
    L=\frac{1}{2} C_\Sigma \dot{\Phi}^2 + E_J \cos(\phi).
$$
The conjugate charge $q=C_\Sigma \dot{\Phi}=2en$ leads to the Hamiltonian:
$$
    \hat{H}= 4E_C (\hat{n}-n_g)^2 - E_J \cos(\hat{\phi}), \quad [\hat{\phi},\hat{n}]=i\hbar,
$$
where $E_C=e^2/2C_\Sigma$ represents the charging energy or the electrostatic term, $E_J=\Phi_0 I_C/2\pi$ represents the Josephson Energy for the nonlinear inductive term, and $n_g$ represents an offset charge controlled by a gate electrode. Physically, this Hamiltonian describes a quantum harmonic osciallator of a particle of mass $1/8E_C$ moving in a periodic potential $U(\phi)=-E_J\cos(\phi)$.

### The Current-Phase Relationship (CPR):
At the junction level, the current-phase relationship (CPR) reflects the microscopic materials' properties:
$$
    I(\phi) = \sum_{m\geq 1} I_m \sin(m\phi).
$$
- For a tunnel junction (Superconductor-Insulator-Superconductor/SIS): $I_1$ dominates so $I(\phi)$ is nearly sinusodial.
- For a metallic constriction (Superconductor-Constriction-Superconductor/SCS): higher harmonics tend to appear in the series. 
Integrating $I(\phi)$ over all $\phi$ gives the Josephson Potential:
$$
    U(\phi) = -\sum_m E_m \cos(m\phi), \quad E_m = \frac{\Phi_0 I_m}{2\pi m}.
$$
So, the potential is able to influence the transmon's anharmonicity and noise sensitivity.

### Discretizing the Continous Hamiltonian into a Tight-Binding Hamiltonian:
We pick the charge basis $\ket{n}$ (Cooper-pair number). Using the identity $e^{\pm i\hat{\phi}}\ket{n}=\ket{n\mp 1}$, we have:
$$
    \cos(m\hat{\phi}) = \frac{1}{2} \sum_n \left(\ket{n+m}\bra{n}+\ket{n}\bra{n+m} \right),
$$
and the Hamiltonian becomes a 1-D tight-binding lattice in $n$:
$$
    H_{nm'} = 4E_C (n-n_g)^2 \delta_{n,n'} - \frac{1}{2} E_m (\delta_{n',n+m} + \delta_{n',n-m}).
$$
Each harmonic $m$ in the CPR adds a hopping range $m$ to the lattice. Diagnolizing this Hermitian matrix gives the eigenvalues $E_i$ and eigenvectors $U_{ni}$. From $E_i$, we can compute key metrics:
- Transition Frequency: $\omega_{01}=E_1-E_0$ for qubit operating frequency;
- Anharmonicity: $\alpha(E_2 - E_1)-(E_1-E_0)$ for gate selectivity and leakage;
- Charge Dispersion: variation between $\omega_{01}$ and $n_g$ for charge-noise sensitivity.

### Open System Dynamics and Decoherence:
In realistic environments, the transmon interacts with external electromagnetic fields. Mathematically, we can describe this as an open quantum system using the Lindblad master equation:
$$
    \frac{\partial \rho}{\partial t} = -i \hbar[\hat{H},\hat{\rho}] + \sum_k \left(L_k \hat{\rho} L_l^\dagger - \frac{1}{2} \{L_k^\dagger L_k, \hat{\rho} \} \right),
$$
with operators $L_k$ encoding:
- Energy Relaxation $(T_1)$: $L_1 = \sqrt{1/T_1} \ket{0}\bra{1}$;
- Pure dephasing $(T_\varphi)$: $L_\varphi = \sqrt{1/T_\varphi} \hat{\sigma}_z$.

For slow, correlated noise, we could also replace the Lindblad master equation with a stochastic Hamiltonian evolution:
$$
    i\hbar \frac{\partial}{\partial t} \ket{\psi} = \left(\hat{H}_0 + \zeta(t) \hat{V} \right)\ket{\psi},
$$
for the noise potential contribution $\hat{V}$ which averages over many noise realizations. This should produce a Gaussian like coherence decay.

## **Building the Transmon Qubit computationally:**
