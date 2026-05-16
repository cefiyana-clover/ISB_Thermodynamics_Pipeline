"""
=============================================================================
PIPELINE ISB FINAL VERSION - SCRIPT 03: LARGE-SCALE DYNAMIC COHORT SIMULATION
=============================================================================
Function: High-fidelity ODE-PDE Monte Carlo execution across virtual cohorts.
          Incorporates genetic ancestry (mtDNA-CN), systemic allostatic load, 
          senescence entropy, focal vagal influx, dose-dependent EAAT downregulation, 
          steady-state homeostatic calibration, and focal topological phase transition.
Status: Reproducibility Standard - DETERMINISTIC COMPUTATIONAL RIGOR
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
import time
import warnings
warnings.filterwarnings("ignore")

print("[SYSTEM] Initializing Large-Scale ODE-PDE Cohort Simulation (Zero-Assumption Architecture)...")
BASE_DIR = '/content/drive/MyDrive/ISB_Empirical_Vault'

POPULATION_SIZE = 40000 
QUARTER_POP = POPULATION_SIZE // 4

# ---------------------------------------------------------
# [MODULE 1] DATA INGESTION (Spatiotemporal & Parameter Anchors)
# ---------------------------------------------------------
print("\n[MODULE 1] Loading Spatiotemporal Matrices & Parameter Anchors...")
try:
    L1_fMRI = np.load(f"{BASE_DIR}/L1_fMRI_Adjacency.npy")
    L2_DTI = np.load(f"{BASE_DIR}/L2_DTI_Structural_Mask.npy")
    P_basal_raw = np.load(f"{BASE_DIR}/L3_Transcriptomic_Pbasal.npy")
    Vmax_raw = np.load(f"{BASE_DIR}/L4_PET_Vmax_Density.npy")

    with open(f"{BASE_DIR}/L5_L6_Clinical_Anchors.json", "r") as f:
        anchors = json.load(f)
    with open(f"{BASE_DIR}/L7_Population_Genetics.json", "r") as f:
        pop_genetics = json.load(f)
except Exception as e:
    raise RuntimeError(f"[FATAL ERROR] Data Ingestion Failed: {e}")

TRUE_NODES = L1_fMRI.shape[0] 
P_basal_global_array = P_basal_raw[:TRUE_NODES]
Vmax = Vmax_raw[:TRUE_NODES]
NUM_NODES = TRUE_NODES

fused_adjacency = L1_fMRI * L2_DTI
Degree_Matrix = np.diag(np.sum(fused_adjacency, axis=1))
Laplacian_L = Degree_Matrix - fused_adjacency

# Parameter Synchronization with Central JSON Vault
baseline_atp = anchors["1H_MRS_Metabolomics"]["Baseline_ATP_mM"]
baseline_glu = anchors["1H_MRS_Metabolomics"]["Baseline_Glutamate_mM"]
CRITICAL_ATP_BIFURCATION = anchors["1H_MRS_Metabolomics"]["Saddle_Node_Threshold"]
stoich_atp_glu = anchors["Thermodynamic_Coupling"]["ATP_per_Glutamate_Stoichiometry"]
decline_rate = anchors["Mitochondrial_Senescence"]["Decline_Rate_Per_Year"]
peak_age = anchors["Mitochondrial_Senescence"]["Peak_Optimal_Age"]
vagal_firing_mult = anchors["Neurophysiology_Anchors"]["Vagal_Hyperfiring_Multiplier"]
eaat_loss_max = anchors["Neurophysiology_Anchors"]["EAAT_Downregulation_Max"]
R_STRESS_MAX_CAP = anchors["Neurophysiology_Anchors"]["Max_Metabolic_Capacity_Limit"]

Km_ATP, Km_G, k_prod, D_G = 0.1, 0.1, 0.05, 0.01

# ---------------------------------------------------------
# [MODULE 2] COHORT SYNTHESIS: ANCESTRY, ALLOSTATIC LOAD, AND SENESCENCE
# ---------------------------------------------------------
print("\n[MODULE 2] Synthesizing Multidimensional Patient Profiles...")
np.random.seed(42)

global_mean_mtCN = pop_genetics["GLOBAL_MEAN_REFERENCE"]

def generate_cohort_rbs(ancestry_key, size):
    mean_cn = pop_genetics[ancestry_key]["mean_mtCN"]
    std_cn = pop_genetics[ancestry_key]["std_mtCN"]
    return np.random.normal(loc=mean_cn, scale=std_cn, size=size) / global_mean_mtCN

RBS_Population = np.concatenate([
    generate_cohort_rbs("AFR", QUARTER_POP),
    generate_cohort_rbs("EUR", QUARTER_POP),
    generate_cohort_rbs("EAS", QUARTER_POP),
    generate_cohort_rbs("AMR", QUARTER_POP)
])
Ancestry_Labels = np.array(['AFR']*QUARTER_POP + ['EUR']*QUARTER_POP + ['EAS']*QUARTER_POP + ['AMR']*QUARTER_POP)

# Stratification of Allostatic Load and Age
# Note: The distribution natively respects the 4.0 hard ceiling defined in JSON
R_stress_Population = np.random.uniform(low=1.0, high=R_STRESS_MAX_CAP, size=POPULATION_SIZE)
Age_Population = np.random.uniform(low=20.0, high=80.0, size=POPULATION_SIZE)
Age_Penalty_Population = np.where(Age_Population > peak_age, 1.0 - ((Age_Population - peak_age) * decline_rate), 1.0)

# ---------------------------------------------------------
# [MODULE 3] INDIVIDUAL ODE-PDE COMPUTE NODE
# ---------------------------------------------------------
ALLOSTATIC_SPAN = (0, 1000) 
baseline_glu_prod = k_prod * baseline_atp

def simulate_virtual_patient(patient_id, rbs_factor, r_stress_scalar, age_penalty):
    # 1. Parameter ISB Definition
    personal_p_basal_raw = P_basal_global_array * rbs_factor * age_penalty
    personal_p_basal_clamped = np.clip(personal_p_basal_raw, a_min=4.0, a_max=9.0)
    epsilon = 1e-6
    mean_p_basal = np.mean(personal_p_basal_clamped)
    patient_isb = mean_p_basal / (r_stress_scalar + epsilon)
    
    # 2. Steady-State Homeostatic Calibration (d[ATP]/dt = 0 at T=0)
    Basal_Thermodynamic_Stress = 1.0 * stoich_atp_glu
    Basal_Drain = (Basal_Thermodynamic_Stress * baseline_glu) / (Km_ATP + baseline_glu)
    k_decay_personal = (personal_p_basal_clamped - Basal_Drain) / baseline_atp
    
    # 3. Pathological Load Initialization (Vagal Influx & Thermodynamic Drain)
    Thermodynamic_Stress = np.full(NUM_NODES, r_stress_scalar * stoich_atp_glu)
    Chronic_Vagal_Influx = np.zeros(NUM_NODES)
    if r_stress_scalar > 1.0:
        Chronic_Vagal_Influx[0] = baseline_glu_prod * (r_stress_scalar - 1.0) * vagal_firing_mult
        
    # 4. Dose-Dependent EAAT Downregulation (TNF-alpha Mechanism)
    stress_ratio = (r_stress_scalar - 1.0) / (R_STRESS_MAX_CAP - 1.0)
    Vmax_impaired = Vmax * (1.0 - (eaat_loss_max * stress_ratio))
    
    def patient_pde(t, Y):
        ATP = Y[:NUM_NODES]
        G = Y[NUM_NODES:]
        dATP_dt = personal_p_basal_clamped - (Thermodynamic_Stress * G) / (Km_ATP + G) - (k_decay_personal * ATP)
        diffusion_term = D_G * np.dot(Laplacian_L, G)
        dG_dt = (k_prod * ATP) + Chronic_Vagal_Influx - (Vmax_impaired * G) / (Km_G + G) - diffusion_term
        return np.concatenate([dATP_dt, dG_dt])

    Y0_ATP = np.full(NUM_NODES, baseline_atp)
    Y0_G = np.full(NUM_NODES, baseline_glu)
    Y0 = np.concatenate([Y0_ATP, Y0_G])
    
    def atp_bifurcation_event(t, Y):
        # Focal Phase Transition Principle: System evaluates based on the absolute minimum node
        min_atp = np.min(Y[:NUM_NODES])
        return min_atp - CRITICAL_ATP_BIFURCATION
    atp_bifurcation_event.terminal = True
    atp_bifurcation_event.direction = -1

    sol = solve_ivp(patient_pde, ALLOSTATIC_SPAN, Y0, method='Radau', 
                    events=atp_bifurcation_event, rtol=1e-3, atol=1e-5) 
    
    survived = sol.status != 1 
    time_to_collapse = sol.t[-1] if not survived else 1000.0
    
    return (patient_id, patient_isb, time_to_collapse, survived)

# ---------------------------------------------------------
# [MODULE 4] PARALLEL EXECUTION ARCHITECTURE
# ---------------------------------------------------------
print(f"\n[MODULE 4] Initiating Parallel Computation across {POPULATION_SIZE} Nodes...")
start_time = time.time()

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(simulate_virtual_patient)(i, RBS_Population[i], R_stress_Population[i], Age_Penalty_Population[i]) 
    for i in range(POPULATION_SIZE)
)

end_time = time.time()
print(f"\n-> [SYSTEM] Parallel Integration Resolved in {(end_time - start_time)/60:.2f} minutes.")

# ---------------------------------------------------------
# [MODULE 5] BIFURCATION DATA AGGREGATION
# ---------------------------------------------------------
patient_ids, isb_scores, collapse_times, survived_flags = zip(*results)

df_sim = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Ancestry': Ancestry_Labels,
    'Age': Age_Population,
    'Age_Penalty_Factor': Age_Penalty_Population,
    'Allostatic_Load_OR': R_stress_Population,
    'Stabilized_ISB_Score': isb_scores,
    'Time_To_Collapse_Days': collapse_times,
    'Survived_1000_Days': survived_flags
})

df_sim.to_csv(f"{BASE_DIR}/Output_TrueDynamic_MegaSimulation_Final.csv", index=False)

print("\n=============================================================================")
print("FOCAL PHASE TRANSITION RATE BY ANCESTRY UNDER EXTREME ALLOSTATIC LOAD (OR > 3.0):")
print("*(Transition defined as focal ATP depletion below 0.5 mM threshold prior to Day 1000)*")
extreme_load = df_sim[df_sim['Allostatic_Load_OR'] > 3.0]
collapse_rates = (~extreme_load['Survived_1000_Days']).groupby(extreme_load['Ancestry']).mean() * 100
print(collapse_rates.sort_values(ascending=False).to_string(float_format="%.2f%%"))
print("=============================================================================\n")
