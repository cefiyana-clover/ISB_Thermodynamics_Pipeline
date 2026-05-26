"""
=============================================================================
PIPELINE ISB - SCRIPT 03: LARGE-SCALE DYNAMIC COHORT SIMULATION (ODE-PDE)
=============================================================================
Function: Deterministic ODE-PDE Monte Carlo execution across virtual cohorts.
          Incorporates genetic ancestry (mtDNA-CN), systemic allostatic load, 
          senescence entropy, focal vagal influx, dose-dependent EAAT downregulation, 
          and structural phase transition event mapping.
Status: Reproducibility Standard - Deterministic Computational Rigor
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

print("[SYSTEM] Initializing large-scale deterministic ODE-PDE cohort simulation...")
BASE_DIR = '/content/drive/MyDrive/ISB_Empirical_Vault'

# Enforce high-density population configuration
POPULATION_SIZE = 40000 
QUARTER_POP = POPULATION_SIZE // 4

# ---------------------------------------------------------
# [MODULE 1] DATA INGESTION & NETWORK TOPOLOGY
# ---------------------------------------------------------
print("\n[MODULE 1] Loading spatiotemporal matrices and structural anchors...")
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
    raise RuntimeError(f"[PIPELINE ERROR] Data ingestion failed: {e}")

NUM_NODES = L1_fMRI.shape[0] 
P_basal_global_array = P_basal_raw[:NUM_NODES]
Vmax = Vmax_raw[:NUM_NODES]

# Constructing Spatiotemporal Graph Laplacian (L = D - W)
Degree_Matrix = np.diag(np.sum(L2_DTI, axis=1))
Laplacian_L = Degree_Matrix - L2_DTI

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
# [MODULE 2] COHORT SYNTHESIS
# ---------------------------------------------------------
print("\n[MODULE 2] Synthesizing multidimensional patient profiles...")
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

R_stress_Population = np.random.uniform(low=1.0, high=R_STRESS_MAX_CAP, size=POPULATION_SIZE)
Age_Population = np.random.uniform(low=20.0, high=80.0, size=POPULATION_SIZE)
Age_Penalty_Population = np.where(Age_Population > peak_age, 1.0 - ((Age_Population - peak_age) * decline_rate), 1.0)

# ---------------------------------------------------------
# [MODULE 3] INDIVIDUAL INTEGRATION COMPUTE NODE
# ---------------------------------------------------------
ALLOSTATIC_SPAN = (0, 1000) 
baseline_glu_prod = k_prod * baseline_atp

def simulate_virtual_patient(patient_id, rbs_factor, r_stress_scalar, age_penalty):
    personal_p_basal_raw = P_basal_global_array * rbs_factor * age_penalty
    personal_p_basal_clamped = np.clip(personal_p_basal_raw, a_min=4.0, a_max=9.0)
    epsilon = 1e-6
    mean_p_basal = np.mean(personal_p_basal_clamped)
    patient_isb = mean_p_basal / (r_stress_scalar + epsilon)
    
    # [DOUBLE STEADY-STATE CALIBRATION]
    Basal_Thermodynamic_Stress = 1.0 * stoich_atp_glu
    Basal_Drain = (Basal_Thermodynamic_Stress * baseline_glu) / (Km_ATP + baseline_glu)
    k_decay_personal = (personal_p_basal_clamped - Basal_Drain) / baseline_atp
    
    # Chronic Vagal Influx Formulation (Focal to Node 0)
    Chronic_Vagal_Influx = np.zeros(NUM_NODES)
    if r_stress_scalar > 1.0:
        Chronic_Vagal_Influx[0] = baseline_glu_prod * (r_stress_scalar - 1.0) * vagal_firing_mult
        
    # Dose-Dependent EAAT Downregulation bounded by maximum constraints
    stress_ratio = (r_stress_scalar - 1.0) / (R_STRESS_MAX_CAP - 1.0)
    Vmax_impaired = Vmax * np.maximum((1.0 - (eaat_loss_max * stress_ratio)), (1.0 - eaat_loss_max))
    
    def patient_pde(t, Y):
        ATP = Y[:NUM_NODES]
        G = Y[NUM_NODES:]
        
        # [THERMODYNAMIC CORRECTION] Restoring exact modular coupling kinetics
        Dynamic_Thermo_Stress = r_stress_scalar * stoich_atp_glu
        dATP_dt = personal_p_basal_clamped - (Dynamic_Thermo_Stress * G) / (Km_ATP + G) - (k_decay_personal * ATP)
        
        # Anisotropic network dispersion mapping
        diffusion_term = D_G * np.dot(Laplacian_L, G)
        dG_dt = (k_prod * ATP) + Chronic_Vagal_Influx - (Vmax_impaired * G) / (Km_G + G) - diffusion_term
        
        return np.concatenate([dATP_dt, dG_dt])

    Y0_ATP = np.full(NUM_NODES, baseline_atp)
    Y0_G = np.full(NUM_NODES, baseline_glu)
    Y0 = np.concatenate([Y0_ATP, Y0_G])
    
    # Structural Event Detection for Saddle-Node Bifurcation
    def atp_bifurcation_event(t, Y):
        return np.min(Y[:NUM_NODES]) - CRITICAL_ATP_BIFURCATION
    atp_bifurcation_event.terminal = True
    atp_bifurcation_event.direction = -1

    sol = solve_ivp(patient_pde, ALLOSTATIC_SPAN, Y0, method='Radau', 
                    events=atp_bifurcation_event, rtol=1e-3, atol=1e-5) 
    
    # [LOGIC CORRECTION] Restoring exact state mapping flags
    survived = sol.status != 1 
    time_to_bifurcation = sol.t[-1] if not survived else 1000.0
    
    return (patient_id, patient_isb, time_to_bifurcation, survived)

# ---------------------------------------------------------
# [MODULE 4] PARALLEL EXECUTION ARCHITECTURE
# ---------------------------------------------------------
print(f"\n[MODULE 4] Initiating parallel SDE computation across {POPULATION_SIZE} nodes...")
start_time = time.time()

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(simulate_virtual_patient)(i, RBS_Population[i], R_stress_Population[i], Age_Penalty_Population[i]) 
    for i in range(POPULATION_SIZE)
)

end_time = time.time()
print(f"\n-> [SYSTEM] Parallel integration completed in {(end_time - start_time)/60:.2f} minutes.")

# ---------------------------------------------------------
# [MODULE 5] BIFURCATION DATA AGGREGATION
# ---------------------------------------------------------
patient_ids, isb_scores, bifurcation_times, survived_flags = zip(*results)

df_sim = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Ancestry': Ancestry_Labels,
    'Age': Age_Population,
    'Age_Penalty_Factor': Age_Penalty_Population,
    'Allostatic_Load_OR': R_stress_Population,
    'Stabilized_ISB_Score': isb_scores,
    'Time_To_Bifurcation_Days': bifurcation_times,
    'Survived_1000_Days': survived_flags
})

# Secure synchronization of file artifact outputs
df_sim.to_csv(f"{BASE_DIR}/Output_TrueDynamic_MegaSimulation_Final.csv", index=False)

print("\n=============================================================================")
print("FOCAL PHASE TRANSITION RATE BY ANCESTRY UNDER EXTREME ALLOSTATIC LOAD (OR > 3.0):")
print("*(Transition defined as focal ATP depletion below 0.5 mM threshold prior to Day 1000)*")
extreme_load = df_sim[df_sim['Allostatic_Load_OR'] > 3.0]
if not extreme_load.empty:
    collapse_rates = (~extreme_load['Survived_1000_Days']).groupby(extreme_load['Ancestry']).mean() * 100
    print(collapse_rates.sort_values(ascending=False).to_string(float_format="%.2f%%"))
else:
    print("No subjects meeting the criteria OR > 3.0 identified in this simulation run.")
print("=============================================================================\n")
