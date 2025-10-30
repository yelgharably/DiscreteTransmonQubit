# **Modeling Transmon Qubits as Tight-Binding Hamiltonians**
## QSEG610 - Engineering The Quantum Revolution Final Project
## Youssef El Gharably -- Fall 2025
### Instructors: Dr Garcia-Frias, Dr Vishal Saxena

Superconducting Qubits -especially Transmon qubits- are the workhorses of modern quantum processors. They are built from the simplest possible superconducting circuit, a Josephson Junction (JJ) shunted by a large capacitance. The configuration is able to make a nonlinear quantum oscillator whose lowest two energy levels form an effective qubit. 

The Transmon Qubit sits at the intersection of:
- Quantum Mechanics; it provides the Hamiltonian and its Quantization rules:
```math
    \hat{H} = 4E_C (\hat{n}-n_g)^2 - E_J \cos(\hat{\phi}),
```
- Tight-Binding Model; it provides a mean of reducing the continous Schrodinger equation** to a discrete tight-binding eigenproblem solvable by matrix methods,
- Electronics; it connects the circuit parameters (capacitance, inductance, critical current) to the Quantum Energies $E_C$ and $E_J$ by desigining control/readout hardware that couples via voltage and flux lines.

### The Physical Picture of the Transmon Qubit as a Quantum Circuit:
A JJ behaves like a nonlinear, nondissipative inductor whose current-phase relation is $I(\phi) = I_C \sin(\phi)$. In circuit quantization, the Lagrangian of a node flux $\Phi=\frac{\Phi_0}{2\pi}\phi$ is:
```math
    L=\frac{1}{2} C_\Sigma \dot{\Phi}^2 + E_J \cos(\phi).
```
The conjugate charge $q=C_\Sigma \dot{\Phi}=2en$ leads to the Hamiltonian:
```math
    \hat{H}= 4E_C (\hat{n}-n_g)^2 - E_J \cos(\hat{\phi}), \quad [\hat{\phi},\hat{n}]=i\hbar,
```
where $E_C=e^2/2C_\Sigma$ represents the charging energy or the electrostatic term, $E_J=\Phi_0 I_C/2\pi$ represents the Josephson Energy for the nonlinear inductive term, and $n_g$ represents an offset charge controlled by a gate electrode. Physically, this Hamiltonian describes a quantum harmonic osciallator of a particle of mass $1/8E_C$ moving in a periodic potential $U(\phi)=-E_J\cos(\phi)$.

### The Current-Phase Relationship (CPR):
At the junction level, the current-phase relationship (CPR) reflects the microscopic materials' properties:
```math
    I(\phi) = \sum_{m\geq 1} I_m \sin(m\phi).
```
- For a tunnel junction (Superconductor-Insulator-Superconductor/SIS): $I_1$ dominates so $I(\phi)$ is nearly sinusodial.
- For a metallic constriction (Superconductor-Constriction-Superconductor/SCS): higher harmonics tend to appear in the series. 
Integrating $I(\phi)$ over all $\phi$ gives the Josephson Potential:
```math
    U(\phi) = -\sum_m E_m \cos(m\phi), \quad E_m = \frac{\Phi_0 I_m}{2\pi m}.
```
So, the potential is able to influence the transmon's anharmonicity and noise sensitivity.

### Discretizing the Continous Hamiltonian into a Tight-Binding Hamiltonian:
We pick the charge basis $\ket{n}$ (Cooper-pair number). Using the identity $e^{\pm i\hat{\phi}}\ket{n}=\ket{n\mp 1}$, we have:
```math
    \cos(m\hat{\phi}) = \frac{1}{2} \sum_n \left(\ket{n+m}\bra{n}+\ket{n}\bra{n+m} \right),
```
and the Hamiltonian becomes a 1-D tight-binding lattice in $n$:
```math
    H_{nm'} = 4E_C (n-n_g)^2 \delta_{n,n'} - \frac{1}{2} E_m (\delta_{n',n+m} + \delta_{n',n-m}).
```
Each harmonic $m$ in the CPR adds a hopping range $m$ to the lattice. Diagnolizing this Hermitian matrix gives the eigenvalues $E_i$ and eigenvectors $U_{ni}$. From $E_i$, we can compute key metrics:
- Transition Frequency: $\omega_{01}=E_1-E_0$ for qubit operating frequency;
- Anharmonicity: $\alpha(E_2 - E_1)-(E_1-E_0)$ for gate selectivity and leakage;
- Charge Dispersion: variation between $\omega_{01}$ and $n_g$ for charge-noise sensitivity.

### Open System Dynamics and Decoherence:
In realistic environments, the transmon interacts with external electromagnetic fields. Mathematically, we can describe this as an open quantum system using the Lindblad master equation:
```math
    \frac{\partial \rho}{\partial t} = -i \hbar[\hat{H},\hat{\rho}] + \sum_k \left(L_k \hat{\rho} L_l^\dagger - \frac{1}{2} \{L_k^\dagger L_k, \hat{\rho} \} \right),
```
with operators $L_k$ encoding:
- Energy Relaxation $(T_1)$: $L_1 = \sqrt{1/T_1} \ket{0}\bra{1}$;
- Pure dephasing $(T_\varphi)$: $L_\varphi = \sqrt{1/T_\varphi} \hat{\sigma}_z$.

For slow, correlated noise, we could also replace the Lindblad master equation with a stochastic Hamiltonian evolution:
```math
    i\hbar \frac{\partial}{\partial t} \ket{\psi} = \left(\hat{H}_0 + \zeta(t) \hat{V} \right)\ket{\psi},
```
for the noise potential contribution $\hat{V}$ which averages over many noise realizations. This should produce a Gaussian like coherence decay.


The output of the cells modeling flux dependence shows proper behavior of a Transmon qubit where $\omega_{01}\approx \sqrt{8E_J E_C}$ and both $\braket{0|n|1}$ and $\braket{1|n|2}$ are non-zero. We also have the off-diagonal components of the TB Hamiltonian as $-10$, which is the hopping term. 
Furthermore, the plot showing the flux dependence on $\omega_{01}$ shows a maximum at $\Phi=0$ when $E_J=E_{J,\Sigma}$, so the qubit freuqency is highest. It also shows a minimum at $\Phi=\Phi_0/2$ when $E_J=0$. This is due to the symmetric SQUID limit when the Josephson potential nearly vanishes causing $\omega_{01}$ to collapse to a small value. This shape matches the experimental data acquired by Koch et al (2007): Transmon Frequency vs Flux Bias.

Looking at the Anharmonity plot, we see some interesting behavior due to the current-phase relationship (CPR):
```math
    I(\phi) = I_1 \sin(\phi) + I_2 \sin(2\phi) + I_3 \sin(3\phi) + I_4 \sin(4\phi)
```
which would correspond to the potential:
```math
    U(\phi) = -E_1 \cos(\phi) - E_2 \cos(2\phi) - E_3 \cos(3\phi) - E_4 \cos(4\phi).
```

By expanding into more harmonics, the JJ behaves in a non-sinusodial manner. This is typical for metallic constriction junctions (ScS), graphene-based junctions, and high-transperancy 2D weak links where more harmonics are measurable and significant. The plots validate this physical behavior through a TB hamiltonian representation.

## **Markovian and Non-Markovian System Evolution**:
In order to properly set this up, we need to pick a bias, pre-build the Hamiltonian and eigenbasis, then truncate to the lowest $d$ levels.

The output of the previous cell shows proper behavior of a Transmon qubit where $\omega_{01}\approx \sqrt{8E_J E_C}$ and both $\braket{0|n|1}$ and $\braket{1|n|2}$ are non-zero. We also have the off-diagonal components of the TB Hamiltonian as $-10$, which is the hopping term. 
Furthermore, the plot showing the flux dependence on $\omega_{01}$ shows a maximum at $\Phi=0$ when $E_J=E_{J,\Sigma}$, so the qubit freuqency is highest. It also shows a minimum at $\Phi=\Phi_0/2$ when $E_J=0$. This is due to the symmetric SQUID limit when the Josephson potential nearly vanishes causing $\omega_{01}$ to collapse to a small value. This shape matches the experimental data acquired by Koch et al (2007): Transmon Frequency vs Flux Bias.

Looking at the Anharmonity plot, we see some interesting behavior due to the current-phase relationship (CPR):
```math
    I(\phi) = I_1 \sin(\phi) + I_2 \sin(2\phi) + I_3 \sin(3\phi) + I_4 \sin(4\phi)
```
which would correspond to the potential:
```math
    U(\phi) = -E_1 \cos(\phi) - E_2 \cos(2\phi) - E_3 \cos(3\phi) - E_4 \cos(4\phi).
```

By expanding into more harmonics, the JJ behaves in a non-sinusodial manner. This is typical for metallic constriction junctions (ScS), graphene-based junctions, and high-transperancy 2D weak links where more harmonics are measurable and significant. The plots validate this physical behavior through a TB hamiltonian representation.

## **Markovian and Non-Markovian System Evolution**:
In order to properly set this up, we need to pick a bias, pre-build the Hamiltonian and eigenbasis, then truncate to the lowest $d$ levels.


We then evolve the density operator such that: 
```math
    \frac{\partial}{\partial t}\rho = -i\hbar [\hat{H},\hat{\rho}] + \sum_k \mathcal{D}[L_k]\hat{\rho}, \quad \mathcal{D}[L]\hat{\rho} = L\rho L^\dagger - \frac{1}{2} \{L^\dagger L,\hat{\rho} \}.
```
We collapse the operators such that:
- Relaxation $T_1$ between adjacent levels is $L_{10} = \sqrt{\Gamma_{10}} \ket{0}\bra{1}$, and $L_{21}=\sqrt{\Gamma_{21}}\ket{1}\bra{2}$, where $\Gamma$ is the relaxation rate or energy decay rate. We can model the relaxation $T_1$ in a more physical sense using the spectral density of fluctuations at that frequency $S_n(\omega_{ij})$ such that for relaxation:
```math
    \Gamma_{ij} = |\braket{i|\hat{n}|j}|^2 S_n(\omega_{ij}), \quad S_n(\omega) = \int_{-\infty}^\infty \braket{\delta n(t)\delta n(0)}e^{i\omega t} \, dt.
```
- Dephasing $T_\varphi$ between adjacent levels is similarly represented where $L_\varphi = -\sqrt{\gamma_\varphi}\hat{\sigma}_z = \sqrt{\gamma_\varphi} (\ket{1}\bra{1}-\ket{0}\bra{0})$ for $\gamma_\varphi=1/T_\varphi$. A more accurate physical description of this can also be formed using $S_n(\omega)$ such that:
```math
    \gamma_\varphi = |\braket{1|\hat{n}|1} - \braket{0|\hat{n}|0}|^2 S_n(0),
```
since zero-frequency noise causes random shifts of the energy splitting.

We will represent noise in three forms:
- Ohmic: $S(\omega) = \eta \omega e^{-\omega/\omega_c} \coth\left(\frac{\hbar\omega}{2k_B T} \right)$
- White Noise: $S(\omega) = S_0$
- 1/$f$ regularized noise: $S(\omega) = \frac{A}{|\omega|+\omega_{min}}$

We then compute the Markovian rates from Fermi's golden rule, downward emission and upward absorption:
```math
    \Gamma_{i\rightarrow j} = |A_{ij}|^2 S(+\omega_{ij}), \quad \Gamma_{j\rightarrow i} = |A_{ij}|^2 S(-\omega_{ij}), \quad A_{ij} = \braket{i|\hat{A}|j}, \quad \omega_{ij} = E_i - E_j,
```
where $\hat{A}=\hat{n}$ represents the charge noise or the flux operator.

### Modeling the Stochastic Hamiltonian (Non-Markovian)
To capture slow drift, 1/$f$ charge noise, or flux noise, we need to make the parameters a random process and evolve pure states then average at the end. 
The Hamiltonian on-site term contains $4E_C (n-n_g)^2$, we need to rebuild H each step with an updated $n_g(t)$ value, or we can add the linearized perturbation $\delta H \approx -8E_C(\hat{n}-n_g)\delta n_g$. We do so using the Ornstein-Uhlenbeck with mean 0, correlation time $\tau_c$ and asymptotic standard deviation $\sigma$. The definition of a Ornstein-Uhlenbeck process is as follow:

>"The Ornstein-Uhlenbeck process is a stochastic process that models the tendency of a variable to revert to a long-term mean, while also experiencing random fluctuations. It is defined by a stochastic differential equation where a drift term pulls the process toward its mean, and a diffusion term introduces random noise." ~ Wikipedia

### Error Isolation Procedure
Currently, the microscopic Hamiltonian is presented as:
```math
H(t;\theta(t)) = 4E_C n^2 - \sum_m E_J^{(m)} (\Phi(t) + \delta \Phi(t))\cos(m\phi) + H_{\mathrm{drive}}(t), \quad \theta(t) \in \{\delta \Phi, \delta E_J^{(m)} , \delta w_d, 1/\mathrm{f_{noise}} \},
```
where $\theta(t)$ represents the stochastic error parameter. We can project this into a small subspace $\mathcal{S}$ using the TB-derived $E_J^{(m)}$. During a gate on $[1,T]$, the propagater $\mathcal{S}$ is:
```math
    U_{\mathrm{act}} = \mathcal{T} \mathrm{exp}\left[-i \int_0^T H_\mathcal{S}(t;\theta(t))\, dt \right], \quad U_{\mathrm{ideal}} = \mathcal{T} \mathrm{exp}\left[H_\mathcal{S}(t;0)\, dt \right].
```
We define the error unitary operator in the computational subspace as:
```math
    U_{\mathrm{error}} = U_{\mathrm{act}} U_{\mathrm{ideal}}^\dagger.
```
If there is leakage, we also keep the block off the $\{\ket{0},\ket{1} \}$ subspace; that captures non-unitary errors we can't simply invert with a two-level correction.

To properly isolate and reverse this error, we follow the following procedure:
- Build the reduced model by diagonalizing the static TB-driven transmon Hamiltonian at the working point to get $\ket{0},\ket{1},\ket{2}$, then precompute the matrices in the basis $n,\phi,\frac{\partial H}{\partial \Phi}, \frac{\partial H}{\partial E_J^{(m)}}$. This would give a fast time revolution while remaining microscopically physical.
- Simulate a noisy gate by picking a control $H_{\mathrm{drive}}$ (Gaussian $\pi$ pulse, flux phase, etc) and a noise trajectory $\theta(t)$. 
- Propagate to get $U_{\mathrm{act}}$ in $\mathcal{S}$ by a magnus expansion. 
- Extract and parametrize the error by computing $U_{\mathrm{error}} = U_{\mathrm{act}}U_{\mathrm{ideal}}^\dagger$. We then write the $U_{\mathrm{erorr}}=e^{iK}$ in the qubit subspace with $K=\frac{\epsilon}{2} \hat{n}\cdot \vec{\sigma}$ where $K_z$ causes a detuning or phase error through a $Z$ rotation, and $K_{x,y}$ cause an amplitude or timing error through an axis tilt. 
- Synthesize a correction by applying $U_{\mathrm{correct}}=U_{error}^\dagger$ as a short correcting pulse. To account for multiple errors, we can use a gradient-based pulse optimization (GRAPE) sequence since the TB pipeline gives analytical derivatives via $\frac{\partial H}{\partial \Phi}$ and $\frac{\partial H}{\partial E_J^{(m)}}$.

Another method to tune out noise into a correctable unitary operation would be by assuming quasi-static slow noise over a gate such that $\theta(t)\approx \theta_0$ as a constant. This would give:
```math
    U_{\mathrm{error}} \approx \mathrm{exp}\left[-i\theta_0\int_0^T G(t)\, dt\right], \quad G(t) = \frac{\partial H}{\partial \Phi}
```
such that $G(t)$ is a noise generator. In this case, we can either estimate $\theta_0$ through tomography, or dynamically correct the gates by minimizing $\int_0^T f(\omega) S_\theta(\omega)\, d\omega$.


## Quantum Transport and DOS Investigation
Now that we have built a formalisim for a TB Transmon Qubit Hamiltonian and proven its accuracy and stability, we can utilize the TB model to look into the properties of Quantum Transport and Density of States in the Transmon Qubit.

### Using KWANT:
We can use KWANT to build a spatially resolved Local DOS, and linear-response transport of a single particle model of the junction. We can build the TB Hamiltonian in particle-hole space, double the basis, and add onsite pairing:
```math
    H = \begin{pmatrix}
    H_0 - \mu & \Delta(r) \\
    \Delta^*(r) & -[H_0 - \mu]^T
    \end{pmatrix}, \quad \Delta = \begin{cases} 
    \Delta_L e^{i\theta},& \mathrm{~left~lead}\\
    \Delta_R e^{i\varphi},& \mathrm{~right~lead}\\
    0 & \mathrm{~constriction}
    \end{cases}
```
We attach the superconducting leads with phases $\theta$ and $\varphi$, then compute the following:
- LSO/DOS through: $\rho(\mathbf{r},E) = \frac{-1}{\pi}\Im [\mathrm{Tr}[G^r(r,r',E)]]$.
- Subgap Andreev bound states by either treating the device as a short finite wire or scanning energy and looking for peaks in LDOS. 
- Phase-dependent spectra through sweeping $\varphi$.

The issue with KWANT is that there is no $\Delta$ self-consistency, which is good for linear transport but not for transmon level quantization.

### Non-Equilibrium Green's Functions (NEGF) + Equilibrium Josephson
We use NEGF when we want currents, CPR, DOS with explicit lead self-energies, equilibrium or finite bias. We use the same device from KWANT with the same lead self-energies. We then compute:
- Retarded Green's Functions: $G^r(E) = [E^+ - H_D - \Sigma_L^r - \Sigma_R^r]^{-1}$.
- DOS from the same $\rho$ function shown above. 
- Josephson Current at equilibrium: $I(\varphi) = \frac{2e}{\hbar} \int \frac{dE}{2\pi} \tanh\left(\frac{E}{2k_B T} \right) \Im[\mathrm{Tr}[\tau_z \Sigma_L^r G^r]]$.

However, KWANT behaved poorly on this sort of system due to its non-hermitian nature. So I decided to use a manual NEGF calculation as shown in the last few cells of the code.
