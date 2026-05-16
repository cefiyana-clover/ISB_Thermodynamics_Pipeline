# The Bioenergetic Stability Index (ISB) Pipeline: A Multi-Scale Thermodynamic Framework

## Overview
This repository contains the deterministic computational architecture (a *Zero-Assumption Architecture*) modeling the *Universal Thermodynamic Theory of Depression*. The framework executes the integration of 6-layer empirical matrices (fMRI, DTI, AHBA, Neuromaps PET, gnomAD) into a coupled Ordinary and Partial Differential Equation (ODE-PDE) system to simulate bioenergetic dynamics across 148 cortical and subcortical nodes.

The primary objective of this pipeline is to quantify the precise thresholds at which *Cumulative Allostatic Load* (internal visceral stress and external psychosocial strain) converges to induce glutamate clearance failure, thereby driving the neural circuit toward a focal *Saddle-Node Bifurcation*. This focal bioenergetic depletion is postulated to serve as the fundamental thermodynamic substrate for major depressive phenotypes.

## Epistemological Stance & The Zero-Assumption Declaration
The execution of this model is strictly anchored in the principle of *Thermodynamic Parsimony*. The system is rigidly calibrated at steady-state ($d[ATP]/dt = 0$) prior to the introduction of any *Allostatic Load*. This *Homeostatic Calibration* ensures that subsequent parameter shifts establishing a *lower-energy equilibrium* are purely physical manifestations of network demand-supply disparity, rather than computational artifacts. The absolute metabolic capacity of the brain is hard-coded to a *Cerebral Metabolic Rate Ceiling* ($4.0\times$ from baseline).

### Causal Chain Determinism
The mechanistic phase transitions within the pipeline are formulated following this deterministic cascade:
`Cumulative Allostatic Load (Internal + External) → Vagal Afferent Hyperfiring → Glutamate Accumulation at NTS/Insula Epicenter → Dose-Dependent Astrocytic EAAT Downregulation → Focal Bioenergetic Depletion (ATP < 0.5 mM).`

## Pipeline Architecture
The execution is distributed across 4 modular Python scripts, adhering to Q1 reproducibility standards:

* **Script 01: Offline Empirical Ingestion:** Extracts connectomic matrices (Destrieux Atlas), transcriptomic profiles (AHBA), and receptor densities (Neuromaps). This module definitively locks the *Cumulative Allostatic Proxies* and the $4.0$ *Metabolic Ceiling* into a central JSON registry.
* **Script 02: Multi-Scale Integration Framework:** Executes scale integration (Genetics $\rightarrow$ Kinetic Equilibrium $\rightarrow$ Laplacian PDE Diffusion). The system simulates spatiotemporal trajectories across varying tiers of senescence entropy and cross-ancestry relative $mtDNA$-CN baseline synthesis.
* **Script 03: Large-Scale Dynamic Cohort:** A Monte Carlo simulation engine evaluating $40,000$ virtual patients over $1000$ biological days. Computational termination is executed focally—a transition is recorded when the absolute minimum node breaches the *Saddle-Node Bifurcation* threshold ($0.5$ mM), demonstrating the phenomenon of *Focal vs. Systemic Lag*.
* **Script 04: High-Fidelity Visual Rendering:** Transforms spatial matrix outputs and survival analyses into publication-ready figures (Kaplan-Meier estimates, Network Topologies, and Bifurcation Curves).

## Limitations & Systemic Bias Acknowledgment
This computational system is inherently subject to empirical confounding variables. We explicitly acknowledge the heterogeneity of population representation within the gnomAD database, which may influence the resolution of the $mtDNA$-CN buffering matrix. Furthermore, the application of the *Graph Laplacian* models diffusion propagation linearly across structural parcellations; adaptive post-trauma neuroplasticity and dynamic receptor regeneration have not yet been quantified within this iteration.

## Future Directions
This deterministic ODE-PDE framework provides a highly testable paradigm. Future architectural iterations should prioritize the injection of *Stochastic Differential Equations* (SDE) to accommodate biological synaptic noise and explore targeted pharmacodynamic interventions (*targeted energy substrates*) aimed at stabilizing the NTS epicenter before systemic network depletion catalyzes.

## License
This project and its associated computational architecture are released under the **Creative Commons Attribution 4.0 International (CC BY NC 4.0)** License. You are free to share and adapt the material for any purpose, provided appropriate credit is given to the original author.

## Contact & Correspondence
For academic inquiries regarding the computational pipeline, thermodynamic parameters, or data reproduction, please contact:
**Cefiyana Cefiyana** (Independent Researcher)
Email: leafcloverfive@gmail.com

---
*Repository Status: Code Freeze (Deterministic)* *Execution Warning: Script 03 requires parallel computational resolution and may entail propagation times exceeding 80 minutes on a standard 2-vCPU environment.*
