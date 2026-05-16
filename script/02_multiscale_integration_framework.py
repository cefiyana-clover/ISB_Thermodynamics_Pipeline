"""
=============================================================================
PIPELINE ISB FINAL VERSION - SCRIPT 02: MULTI-SCALE INTEGRATION FRAMEWORK
=============================================================================
Function: Integrates 6-layer empirical matrices into a coupled ODE-PDE system.
          Incorporates pan-ancestry relative basal synthesis (RBS), cellular 
          senescence, chronic vagal influx, dose-dependent EAAT downregulation, 
          and strict steady-state homeostatic calibration.
Status: Reproducibility Standard - DETERMINISTIC COMPUTATIONAL CORE
=============================================================================
"""

import os
import json
import numpy as np
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings("ignore")

print("[SYSTEM] Initializing Multi-Scale PDE Integration Framework...")
BASE_DIR = '/content/drive/MyDrive/ISB_Empirical_Vault'

# ---------------------------------------------------------
# [MODULE 1 & 2] EMPIRICAL DATA INGESTION & LAPLACIAN CONSTRUCTION
# ---------------------------------------------------------
print("\n[MODULE 1 & 2] Loading Empirical Matrices & Structural Anchors...")
try:
    L1_fMRI = np.load(f"{BASE_DIR}/L1_fMRI_Adjacency.npy")
    L2_DTI = np.load(f"{BASE_DIR}/L2_DTI_Structural_Mask.npy")
    P_basal_raw = np.load(f"{BASE_DIR}/L3_Transcriptomic_Pbasal.npy")
    Vmax_raw = np.load(f"{BASE_DIR}/L4_PET_Vmax_Density.npy")
    
    with open(f"{BASE_DIR}/L5_L6_Clinical_Anchors.json", "r") as f:
        anchors = json.load(f)
    with open(f"{BASE_DIR}/L7_Population_Genetics.json", "r") as f:
        pop_genetics = json.load(f)
        
    NUM_NODES = L1_fMRI.shape[0] 
    P_basal_global = P_basal_raw[:NUM_NODES]
    Vmax = Vmax_raw[:NUM_NODES]
    
    # Constructing Spatiotemporal Graph Laplacian
    fused_adjacency = L1_fMRI * L2_DTI
    Degree_Matrix = np.diag(np.sum(fused_adjacency, axis=1))
    Laplacian_L = Degree_Matrix - fused_adjacency
    
    print(f"-> [SUCCESS] Spatiotemporal Graph Laplacian constructed ({NUM_NODES} nodes).")

except Exception as e:
    raise RuntimeError(f"[FATAL ERROR] Empirical Data Ingestion Failed: {e}")

# ---------------------------------------------------------
# [MODULE 3] MULTI-MODAL PARAMETERIZATION
# ---------------------------------------------------------
print("\n[MODULE 3] Parameterizing Multi-Modal Bioenergetic Anchors...")

baseline_atp = anchors["1H_MRS_Metabolomics"]["Baseline_ATP_mM"]
baseline_glu = anchors["1H_MRS_Metabolomics"]["Baseline_Glutamate_mM"]

# Synchronized Hierarchical JSON Extraction
or_allostatic = anchors["Epidemiological_Allostatic_Proxies"]["Internal_Visceral_Load"]["Systemic_Pain_OR"]
vagal_coef = anchors["Epidemiological_Allostatic_Proxies"]["Vagal_Transfer_Coefficient"]

stoich_atp_glu = anchors["Thermodynamic_Coupling"]["ATP_per_Glutamate_Stoichiometry"]
decline_rate = anchors["Mitochondrial_Senescence"]["Decline_Rate_Per_Year"]
peak_age = anchors["Mitochondrial_Senescence"]["Peak_Optimal_Age"]
vagal_firing_mult = anchors["Neurophysiology_Anchors"]["Vagal_Hyperfiring_Multiplier"]
eaat_loss_max = anchors["Neurophysiology_Anchors"]["EAAT_Downregulation_Max"]
R_STRESS_MAX_CAP = anchors["Neurophysiology_Anchors"]["Max_Metabolic_Capacity_Limit"]

Km_ATP, Km_G, k_prod, D_G = 0.1, 0.1, 0.05, 0.01

# Allostatic Load Calculation
R_baseline = 1.0
R_stress_scalar = R_baseline + (or_allostatic * vagal_coef) 
Thermodynamic_Stress_Global = R_stress_scalar * stoich_atp_glu
R_stress = np.full(NUM_NODES, Thermodynamic_Stress_Global)

# Chronic Vagal Influx Formulation
baseline_glu_prod = k_prod * baseline_atp
Chronic_Vagal_Influx = np.zeros(NUM_NODES)
if R_stress_scalar > 1.0:
    Chronic_Vagal_Influx[0] = baseline_glu_prod * (R_stress_scalar - 1.0) * vagal_firing_mult

# Dose-Dependent EAAT Downregulation
stress_ratio = (R_stress_scalar - 1.0) / (R_STRESS_MAX_CAP - 1.0)
Vmax_impaired = Vmax * (1.0 - (eaat_loss_max * stress_ratio))

print(f"-> [SUCCESS] Kinetic limits defined. Upper R_stress boundary = {R_stress_scalar:.2f} (Ceiling: {R_STRESS_MAX_CAP})")

# ---------------------------------------------------------
# [MODULE 4] COUPLED ODE-PDE KINETIC FRAMEWORK
# ---------------------------------------------------------
def bioenergetic_pde(t, Y, Laplacian, P_basal_current, Vmax_imp, R_stress_arr, Vagal_Influx_Arr, k_decay_arr):
    ATP = Y[:NUM_NODES]
    G = Y[NUM_NODES:]
    
    # Metabolic turnover and thermodynamic drain
    dATP_dt = P_basal_current - (R_stress_arr * G) / (Km_ATP + G) - (k_decay_arr * ATP)
    
    # Spatial diffusion via Graph Laplacian
    diffusion_term = D_G * np.dot(Laplacian, G)
    
    # Glutamate clearance bottleneck and focal influx
    dG_dt = (k_prod * ATP) + Vagal_Influx_Arr - (Vmax_imp * G) / (Km_G + G) - diffusion_term
    
    return np.concatenate([dATP_dt, dG_dt])

# ---------------------------------------------------------
# [MODULE 5 & 6] 4D SPATIOTEMPORAL EXECUTION (Ancestry x Age)
# ---------------------------------------------------------
print("\n[MODULE 5 & 6] Computing 4D Spatiotemporal Trajectories (Ancestry x Senescence Tiers)...")

ALLOSTATIC_SPAN = (0, 1000)
t_eval = np.linspace(ALLOSTATIC_SPAN[0], ALLOSTATIC_SPAN[1], 500)
global_mean_mtCN = pop_genetics["GLOBAL_MEAN_REFERENCE"]

ANCESTRIES = ["AFR", "EUR", "EAS", "AMR"]
COHORT_AGES = [20.0, 30.0, 40.0, 50.0, 60.0]

for age in COHORT_AGES:
    # Senescence entropy factor
    age_penalty = 1.0 - ((age - peak_age) * decline_rate) if age > peak_age else 1.0
    print(f"\n=========================================================")
    print(f"---> [SENESCENCE STRATIFICATION] Age: {int(age)} | Entropy Coefficient: {age_penalty:.4f}")
    
    for ancestry in ANCESTRIES:
        ancestry_mean_mtCN = pop_genetics[ancestry]["mean_mtCN"]
        rbs_factor = ancestry_mean_mtCN / global_mean_mtCN
        
        P_basal_scaled = P_basal_global * rbs_factor * age_penalty
        P_basal_clamped = np.clip(P_basal_scaled, a_min=4.0, a_max=9.0)
        
        # Steady-State Homeostatic Calibration
        Basal_Thermodynamic_Stress = 1.0 * stoich_atp_glu
        Basal_Drain = (Basal_Thermodynamic_Stress * baseline_glu) / (Km_ATP + baseline_glu)
        k_decay_personal = (P_basal_clamped - Basal_Drain) / baseline_atp
        
        # Initial Conditions
        Y0_ATP = np.full(NUM_NODES, baseline_atp)
        Y0_G = np.full(NUM_NODES, baseline_glu)
        Y0 = np.concatenate([Y0_ATP, Y0_G])
        
        print(f"     -> Integrating PDE for Cohort: {ancestry} | RBS: {rbs_factor:.4f}...", end=" ")
        
        solution = solve_ivp(
            fun=bioenergetic_pde,
            t_span=ALLOSTATIC_SPAN,
            y0=Y0,
            t_eval=t_eval,
            args=(Laplacian_L, P_basal_clamped, Vmax_impaired, R_stress, Chronic_Vagal_Influx, k_decay_personal),
            method='Radau', 
            rtol=1e-6, 
            atol=1e-8
        )
        
        if solution.success:
            ATP_Spatiotemporal = solution.y[:NUM_NODES, :]
            G_Spatiotemporal = solution.y[NUM_NODES:, :]
            np.save(f"{BASE_DIR}/Output_ATP_Spatiotemporal_{ancestry}_Age{int(age)}.npy", ATP_Spatiotemporal)
            np.save(f"{BASE_DIR}/Output_G_Spatiotemporal_{ancestry}_Age{int(age)}.npy", G_Spatiotemporal)
            print("[SUCCESS]")
        else:
            print(f"[FAILED] {solution.message}")

np.save(f"{BASE_DIR}/Output_Time_Vector.npy", t_eval)
print("\n=============================================================================")
print("[STATUS] SCRIPT 02 COMPLETE. SPATIOTEMPORAL MATRICES COMPUTED AND ARCHIVED.")
print("=============================================================================\n")
