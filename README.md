# The Bioenergetic Stability Index (ISB) Pipeline: A Deterministic Multi-Scale Thermodynamic Framework

## Overview
This repository contains the deterministic computational architecture (ODE-PDE framework) modeling the bioenergetic thermodynamics of major depressive phenotypes. The pipeline executes the integration of multi-modal empirical matrices (rs-fMRI, DTI spatial coordinates, AHBA transcriptomics, Neuromaps PET, and gnomAD genetics) into a coupled deterministic system to simulate bioenergetic spatiotemporal dynamics across 148 cortical and subcortical nodes.

The primary objective of this architecture is to quantify the precise critical thresholds ($\theta_c$) at which cumulative allostatic load (internal visceral stress and external psychosocial strain) converges to induce focal glutamate clearance failure. This failure drives the neural circuit toward a localized **Saddle-Node Bifurcation** ($[ATP] < 0.5 \text{ mM}$), which is postulated to serve as the fundamental thermodynamic substrate for depressive phenotypes.

## Thermodynamic Parsimony & Boundary Conditions
The execution of this model is strictly anchored in the principle of thermodynamic parsimony. The system is rigidly calibrated at steady-state ($d[ATP]/dt = 0$) prior to the introduction of any allostatic load. This strict homeostatic calibration ensures that subsequent topological phase transitions—establishing a lower-energy equilibrium—are purely physical manifestations of network demand-supply disparity rather than computational artifacts. The absolute metabolic capacity of the astrocytic network is hard-coded to an empirically bounded cerebral metabolic rate ceiling ($4.0\times$ baseline).

### Mechanistic Causal Cascade
The phase transitions within the pipeline are governed by the following deterministic differential cascade:
1. Cumulative Allostatic Load ($\Omega_{ext}$)
2. Vagal Afferent Hyperfiring (Node 0 Epicenter)
3. Focal Glutamate Accumulation ($[Glu]_{syn}$)
4. Dose-Dependent Transporter (EAAT) Downregulation via RNA Velocity Kinetics ($\lambda$)
5. Focal Bioenergetic Depletion (Saddle-Node Bifurcation at $[ATP] < 0.5 \text{ mM}$)

## Pipeline Architecture
The execution is distributed across 4 modular Python scripts, adhering to exact computational reproducibility standards:

* **Script 01 (Offline Empirical Ingestion):** Extracts connectomic matrices (Destrieux Atlas), transcriptomic profiles (AHBA), and receptor densities (PET). 
  * *Methodological Note:* Structural adjacency is derived strictly via 3D Euclidean geometric distance ($1 / d_{ij}$) to ensure authentic spatial diffusion limits. Transcriptomic and kinetic arrays are mapped to the 148-node topology utilizing continuous linear interpolation to prevent baseline averaging artifacts.
* **Script 02 (Multi-Scale Integration Framework):** Executes scale integration mapping genetic vulnerability to kinetic equilibrium, culminating in a Laplacian PDE diffusion network. The solver simulates continuous spatiotemporal trajectories across varying tiers of senescence entropy ($\Delta S_{mito}$) and cross-ancestry relative $mtDNA$-CN synthesis.
* **Script 03 (Large-Scale Dynamic Cohort):** A deterministic Monte Carlo simulation engine evaluating $40,000$ virtual patients over $1000$ biological days. Computational termination mapping is executed focally—a transition is recorded when the absolute minimum node breaches the bifurcation threshold, effectively isolating *Time-to-Bifurcation* kinetics.
* **Script 04 (High-Fidelity Visual Rendering):** Transforms spatial matrix outputs and survival dynamics into publication-ready, mathematically annotated figures (Un-truncated Kaplan-Meier kinetics, Topographical Heatmaps, and Phase Trajectories).

## Limitations & Structural Acknowledgment
This deterministic system is subject to specific methodological boundaries. We acknowledge the inherent representation heterogeneity within the gnomAD database, which influences the baseline resolution of the $mtDNA$-CN buffering matrix. Furthermore, this iteration operates as a pure deterministic ODE-PDE model; it does not account for intrinsic cellular Langevin noise ($dW_t$) or stochastic environmental fluctuations, which may theoretically accelerate phase transitions prior to the absolute deterministic limit.

## Future Directions
This deterministic framework provides a structurally sound and highly testable baseline. Future architectural iterations will prioritize the integration of Stochastic Differential Equations (SDE) to evaluate Noise-Induced Phase Transitions (Critical Slowing Down) and explore computational pharmacodynamics targeting the structural epicenter prior to systemic network failure.

## Academic Correspondence
For inquiries regarding the computational pipeline, differential parameters, data reproduction, or collaborative extensions of this thermodynamic framework, please direct correspondence to:

**Cefiyana Cefiyana** *Lead Investigator / Independent Researcher* Email: leafcloverfive@gmail.com  

---
*Repository Status: Deterministic Code Freeze.* *Execution Note: Script 03 requires parallel computational resolution (joblib/LokyBackend) and entails propagation times exceeding 90 minutes on standard multi-core hardware environments.*
