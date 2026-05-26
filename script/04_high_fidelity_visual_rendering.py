"""
=============================================================================
PIPELINE ISB - SCRIPT 04: HIGH-FIDELITY VISUAL RENDERING ENGINE (ODE-PDE)
=============================================================================
Function: Ingests deterministic spatiotemporal .npy matrices and .csv outputs 
          to render high-resolution, publication-ready analytical figures.
          Engineered to expose topological phase transitions and structural limits.
Status: Reproducibility Standard - Deterministic Visualization Engine
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from nilearn import datasets, plotting
from nilearn.image import load_img
from matplotlib.colors import LinearSegmentedColormap, Normalize
import warnings
warnings.filterwarnings("ignore")

print("[SYSTEM] Initializing high-fidelity deterministic visual rendering engine...")
BASE_DIR = '/content/drive/MyDrive/ISB_Empirical_Vault'
OUTPUT_DIR = f"{BASE_DIR}/Manuscript_Figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Q1 Aesthetics setup with integrated LaTeX formatting
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks")
plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# ---------------------------------------------------------
# [FIGURE 1] FOCAL NETWORK DYNAMICS & ASTROCYTIC COUPLING 
# ---------------------------------------------------------
print("\n[MODULE 1] Synthesizing Figure 1: Focal network dynamics...")
try:
    ATP_data = np.load(f"{BASE_DIR}/Output_ATP_Spatiotemporal_AFR_Age60.npy")
    G_data = np.load(f"{BASE_DIR}/Output_G_Spatiotemporal_AFR_Age60.npy")
    t_eval = np.load(f"{BASE_DIR}/Output_Time_Vector.npy")
    
    with open(f"{BASE_DIR}/L5_L6_Clinical_Anchors.json", "r") as f:
        anchors = json.load(f)
    
    collapse_thresh = anchors["1H_MRS_Metabolomics"]["Saddle_Node_Threshold"]
    Global_Mean_ATP = np.mean(ATP_data, axis=0)
    Global_Mean_G = np.mean(G_data, axis=0)
    
    fig1 = plt.figure(figsize=(14, 10))
    gs = fig1.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.35)
    
    # [Panel 1A] - Deterministic Trajectory Mapping
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.plot(t_eval, ATP_data[0, :], color='#8c1515', linewidth=3, label=r'Epicenter Node ($\eta_0$)')
    ax1.plot(t_eval, ATP_data[147, :], color='#9467bd', linewidth=1.5, linestyle='--', label=r'Distant Node ($\eta_{147}$)')
    ax1.plot(t_eval, Global_Mean_ATP, color='#1f77b4', linewidth=3, label=r'Global Brain $\mu_{ATP}$')
    ax1.axhline(y=collapse_thresh, color='black', linestyle=':', linewidth=2, label=r'Bifurcation Threshold ($\theta_c = 0.5$)')
    
    collapse_indices = np.where(ATP_data[0, :] <= collapse_thresh)[0]
    
    # [VISUAL PATCH] Implementing Bounding Boxes and Safe Coordinate Thresholds
    bbox_style = dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.85)
    
    if len(collapse_indices) > 0:
        collapse_day = t_eval[collapse_indices[0]]
        safe_x = max(50, collapse_day - 250)
        ax1.annotate(f'Saddle-Node Bifurcation\n(Day {collapse_day:.0f})', 
                     xy=(collapse_day, collapse_thresh), 
                     xytext=(safe_x, 1.2),
                     bbox=bbox_style,
                     arrowprops=dict(arrowstyle="->", color='black', lw=1.5, connectionstyle="arc3,rad=-0.1"))
    else:
        min_atp_idx = np.argmin(ATP_data[0, :])
        min_atp_day = t_eval[min_atp_idx]
        min_atp_val = ATP_data[0, min_atp_idx]
        safe_x = max(50, min_atp_day - 350)
        ax1.annotate(f'Stable Attractor Equilibrium\n({min_atp_val:.2f} mM)', 
                     xy=(min_atp_day, min_atp_val), 
                     xytext=(safe_x, min_atp_val + 0.8),
                     bbox=bbox_style,
                     arrowprops=dict(arrowstyle="->", color='black', lw=1.5, connectionstyle="arc3,rad=0.1"))
    
    ax1.set_ylabel(r'$[ATP]_{astrocytic} \ (mM)$', fontweight='bold')
    ax1.set_xlabel(r'$t \ (Days)$', fontweight='bold')
    ax1.set_title('A. Bioenergetic Depletion: Focal vs Systemic Lag', fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim(0, 3.5)

    # [Panel 1B] - Glutamate Core Dynamics
    ax2 = fig1.add_subplot(gs[0, 1])
    ax2.plot(t_eval, G_data[0, :], color='#d62728', linewidth=3, label=r'Epicenter $[Glu]_{syn}$')
    ax2.plot(t_eval, Global_Mean_G, color='#1f77b4', linewidth=2, linestyle='-', label=r'Global Mean $\mu_{Glu}$')
    
    ax2.set_ylabel(r'$[Glu]_{synaptic} \ (mM)$', fontweight='bold')
    ax2.set_xlabel(r'$t \ (Days)$', fontweight='bold')
    ax2.set_title('B. Vagal Influx & Astrocytic EAAT Bottleneck', fontweight='bold')
    ax2.legend(loc='upper left', framealpha=0.9)

    # [Panel 1C] - Resilience Scatter Matrix
    df_sim = pd.read_csv(f"{BASE_DIR}/Output_TrueDynamic_MegaSimulation_Final.csv")
    failed_patients = df_sim[df_sim['Survived_1000_Days'] == False]
    
    ax3 = fig1.add_subplot(gs[1, :])
    sns.scatterplot(data=failed_patients, x='Time_To_Bifurcation_Days', y='Stabilized_ISB_Score', 
                    hue='Ancestry', palette='Set1', s=50, alpha=0.7, edgecolor='black', ax=ax3)
    
    ax3.set_xlabel(r'Time-to-Phase-Transition $t_c$ (Biological Days)', fontweight='bold')
    ax3.set_ylabel(r'Basal ISB Score $\Omega_{basal}$', fontweight='bold')
    ax3.set_title('C. Correlation: Bioenergetic Resilience vs Phase Transition Acceleration', fontweight='bold')
    
    plt.suptitle('Figure 1: Multiscale Dynamics of Connectome-Mediated Bioenergetic Failure', fontweight='bold', fontsize=18, y=1.02)
    fig1.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_1_Complexity.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("   -> [VALIDATED] Figure 1 archived.")
except Exception as e:
    import traceback
    print(f"   -> [PIPELINE ERROR] Figure 1 rendering failed: {e}")
    traceback.print_exc()

# ---------------------------------------------------------
# [FIGURE 2] PAN-ANCESTRY SPATIOTEMPORAL CONNECTOME MAP
# ---------------------------------------------------------
print("\n[MODULE 2] Synthesizing Figure 2: Pan-ancestry map connectome...")
try:
    destrieux = datasets.fetch_atlas_destrieux_2009()
    atlas_img = load_img(destrieux['maps'])
    coords = plotting.find_parcellation_cut_coords(atlas_img)
    
    L2_DTI_Mask = np.load(f"{BASE_DIR}/L2_DTI_Structural_Mask.npy")
    coords = coords[:L2_DTI_Mask.shape[0]] 
    
    ATP_EUR = np.load(f"{BASE_DIR}/Output_ATP_Spatiotemporal_EUR_Age60.npy")
    ATP_EAS = np.load(f"{BASE_DIR}/Output_ATP_Spatiotemporal_EAS_Age60.npy")
    ATP_AMR = np.load(f"{BASE_DIR}/Output_ATP_Spatiotemporal_AMR_Age60.npy")
    ATP_AFR = np.load(f"{BASE_DIR}/Output_ATP_Spatiotemporal_AFR_Age60.npy")
    
    TIME_IDX = -1 
    
    fig2 = plt.figure(figsize=(20, 12))
    gs = fig2.add_gridspec(2, 2, wspace=0.1, hspace=0.3)
    
    cmap_atp = LinearSegmentedColormap.from_list('atp_map', ['#d62728', '#ff7f0e', '#1f77b4']) 
    norm = Normalize(vmin=0.0, vmax=3.0)
    
    node_sizes = [250 if i == 0 else 40 for i in range(len(coords))]
    
    cohorts = [
        ('EUR Cohort (High mt-CN Buffer)', ATP_EUR, gs[0, 0]),
        ('EAS Cohort (Moderate-High Buffer)', ATP_EAS, gs[0, 1]),
        ('AMR Cohort (Moderate-Low Buffer)', ATP_AMR, gs[1, 0]),
        ('AFR Cohort (Depleted mt-CN Buffer)', ATP_AFR, gs[1, 1])
    ]
    
    for title, data, pos in cohorts:
        ax = fig2.add_subplot(pos)
        snapshot = data[:, TIME_IDX]
        
        node_colors_rgba = cmap_atp(norm(snapshot))
        node_colors_hex = [mcolors.to_hex(c) for c in node_colors_rgba]
        
        plotting.plot_connectome(
            L2_DTI_Mask, coords, display_mode='z', 
            edge_threshold='99.5%', edge_cmap='Greys', 
            node_color=node_colors_hex, node_size=node_sizes, 
            colorbar=False, axes=ax, title='' 
        )
        
        ax.set_title(f'{title}\nBiological Day 1000', fontsize=16, fontweight='bold', pad=20)
        print(f"     -> Topographical mapping resolved for {title[:3]}")

    plt.suptitle(r'Figure 2: Topological Heatmap of Focal Bioenergetic Depletion ($t=1000$)', fontweight='bold', fontsize=22, y=1.05)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_2_Network_Comparison.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print("   -> [VALIDATED] Figure 2 archived.")
except Exception as e:
    import traceback
    print(f"   -> [PIPELINE ERROR] Figure 2 rendering failed: {e}")
    traceback.print_exc()

# ---------------------------------------------------------
# [FIGURE 3 & 4] KAPLAN-MEIER ESTIMATE & THERMODYNAMIC CONTOUR
# ---------------------------------------------------------
print("\n[MODULE 3] Synthesizing Figures 3 & 4: Survival and heatmap contours...")
try:
    df_sim = pd.read_csv(f"{BASE_DIR}/Output_TrueDynamic_MegaSimulation_Final.csv")
    extreme_load = df_sim[df_sim['Allostatic_Load_OR'] > 3.0]
    
    # FIGURE 3 - Kaplan Meier
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    colors = {'AFR': '#d62728', 'AMR': '#ff7f0e', 'EAS': '#2ca02c', 'EUR': '#1f77b4'}
    for ancestry in ['AFR', 'AMR', 'EAS', 'EUR']:
        subset = extreme_load[extreme_load['Ancestry'] == ancestry]
        times = np.sort(subset['Time_To_Bifurcation_Days'].values)
        survival_prob = 1.0 - np.arange(1, len(times) + 1) / len(times)
        times = np.insert(times, 0, 0)
        survival_prob = np.insert(survival_prob, 0, 1.0)
        ax3.step(times, survival_prob * 100, where='post', label=ancestry, color=colors[ancestry], linewidth=2.5)
        
    ax3.set_xlim(0, 1000)
    ax3.set_ylim(-5, 105) 
    ax3.set_xlabel(r'Biological Days ($t$)', fontweight='bold')
    ax3.set_ylabel(r'Systemic Bioenergetic Integrity Probability $P(\theta > 0.5) \ (\%)$', fontweight='bold')
    ax3.set_title(r'Figure 3: Kaplan-Meier Survival Estimate Under $OR > 3.0$', fontweight='bold', pad=15)
    ax3.legend(title='Ancestry', loc='lower left')
    fig3.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_3_Kaplan_Meier.png"), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # FIGURE 4 - Thermodynamic Contour
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    collapsed_patients = df_sim[df_sim['Survived_1000_Days'] == False]
    
    sns.kdeplot(data=collapsed_patients, x='Age', y='Allostatic_Load_OR', fill=True, cmap="mako", thresh=0.05, levels=12, ax=ax4, alpha=0.9)
    sns.scatterplot(data=collapsed_patients, x='Age', y='Allostatic_Load_OR', color='white', s=20, alpha=0.6, marker='x', ax=ax4)
    ax4.set_xlim(20, 80)
    ax4.set_ylim(1.0, 4.0)
    ax4.set_xlabel(r'Senescence Entropy $\Delta S_{mito}$ (Age in Years)', fontweight='bold')
    ax4.set_ylabel(r'Cumulative Allostatic Load ($\Omega_{ext}$ / Odds Ratio)', fontweight='bold')
    ax4.set_title(r'Figure 4: Thermodynamic Vulnerability Contour (Bifurcation Zone Density)', fontweight='bold', pad=15)
    fig4.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_4_Thermodynamic_Heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("   -> [VALIDATED] Figures 3 & 4 archived.")
except Exception as e:
    import traceback
    print(f"   -> [PIPELINE ERROR] Figures 3 or 4 rendering failed: {e}")
    traceback.print_exc()

print("\n=============================================================================")
print("[STATUS] SCRIPT 04 COMPLETE. PUBLICATION-READY FIGURES ARCHIVED.")
print(f"Directory: {OUTPUT_DIR}")
print("=============================================================================\n")
