# Copyright (c) 2026 University of Bari.
# Licensed under GNU AGPLv3 OR Commercial License. See LICENSE file for details.


import MDAnalysis as mda
from MDAnalysis.analysis import align, rms, pca
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings('ignore')

def run_analysis():
    # --- CONFIGURATION ---
    pdb_ref_file = "protein_run1.pdb" # insert the name of your pdb template
    dcd_files = ["protein_run1.dcd"] # multiple files for the same run must be arranged in a list: dcd_files = ["your_eq.dcd", "your_prod1.dcd", "your_prod2.dcd"]
    start_frame = 500  # Frames to exclude (minimization/balancing)
    output_prefix = "MD_run1_out" # insert your output prefix

    # Verifica file
    for f in [pdb_ref_file] + dcd_files:
        if not os.path.exists(f):
            print(f"ERROR: File {f} is missing.")
            return

    # --- 1. LOADING AND ALIGNMENT ---
    u = mda.Universe(pdb_ref_file, dcd_files)
    ref = mda.Universe(pdb_ref_file)
    ca_selection = "name CA"
    ca_atoms = u.select_atoms(ca_selection)
    
    print(f"Total frames loaded: {len(u.trajectory)}")
    print("Global alignment in progress...")
    alignment = align.AlignTraj(u, ref, select=ca_selection, in_memory=True).run()

    # --- 2. DATA EXTRACTION (PRODUCTION ONLY) ---
    print(f"Extracting production frames (from frame {start_frame} to end)...")
    
    # Extract float32 coordinates (optimal for JAX)
    # Let's skip the first ones 'start_frame'
    prod_coords = []
    for ts in u.trajectory[start_frame:]:
        prod_coords.append(ca_atoms.positions.copy())
    
    prod_coords = np.array(prod_coords, dtype=np.float32) # Forziamo float32
    n_frames_prod = prod_coords.shape[0]

    # Saving in compressed .npz format
    # The saved format will be a dictionary with key 'coords'
    # Shape: (frame, atoms, 3) -> we flatten it into (frame, atoms*3) as required
    coords_flat = prod_coords.reshape(n_frames_prod, -1)
    np.savez_compressed(f"{output_prefix}_coords_ca.npz", coords=coords_flat)
    print(f"Coordinates saved in {output_prefix}_coords_ca.npz (Float32, Compressed)")

    # --- 3. CORRELATION MATRIX (ON PRODUCTION) ---
    # uncomment if needed
    print("DCCM calculation on production...")
    mean_pos = prod_coords.mean(axis=0)
    displacements = prod_coords - mean_pos
    
    num = np.einsum('tik, tjk -> ij', displacements, displacements)
    norm = np.sqrt(np.einsum('tik, tik -> i', displacements, displacements))
    dccm = (num / np.outer(norm, norm)).astype(np.float32)
    #np.savez_compressed(f"{output_prefix}_dccm.npz", dccm=dccm)

    # --- 4. PCA (ON PRODUCTION) ---
    print("PCA calculation on production...")
    # We create a trajectory subset for PCA using MDAnalysis slicing
    pc_analysis = pca.PCA(u, select=ca_selection, align=False)
    # Let's run it by specifying the time slice
    pc_analysis.run(start=start_frame)
    transformed = pc_analysis.transform(ca_atoms, n_components=5).astype(np.float32)
    #np.savez_compressed(f"{output_prefix}_pca_top5.npz", pca=transformed)

    # --- 5. RMSD (OVER THE ENTIRE TRAJECTORY) ---
    print("RMSD calculation (full simulation to see the balancing)...")
    R = rms.RMSD(u, ref, select=ca_selection).run()
    rmsd_data = R.results.rmsd.astype(np.float32)
    
     # --- 6. RMSF (OVER THE ENTIRE TRAJECTORY) ---
    print("RMSF calculation (full simulation)...")
    R1 = rms.RMSF(ca_atoms).run()
    rmsf_data = R1.results.rmsf.astype(np.float32)

    # --- 6. GRAPHICS ---
    # comment if not needed
    print("Generating graphs...")    
      
    # RMSD with vertical line on the cutoff
    plt.figure(figsize=(10, 4))
    plt.plot(0.01*rmsd_data[:, 0], rmsd_data[:, 2], color='black', lw=0.5)
    plt.axvline(x=0.01*start_frame, color='red', linestyle='--')
    plt.title("Simulation stability (RMSD)")
    plt.xlabel("nanoseconds")
    plt.ylabel("RMSD (Å)")
    #plt.legend()
    plt.savefig(f"{output_prefix}_plot_rmsd.png", dpi=300)
    plt.close()
    
    # RMSF with vertical line on the high attention
    plt.figure(figsize=(10, 4))
    plt.plot(ca_atoms.resids, rmsf_data, color='black', lw=0.5)
    plt.axvline(x=65, color='red', linestyle='--', label='High attention')
    plt.axvline(x=145, color='orange', linestyle='--', label='High attention')
    plt.title("Root mean square fluctuation")
    plt.xlabel("residue number")
    plt.ylabel("RMSF (Å)")
    #plt.legend()
    plt.savefig(f"{output_prefix}_plot_rmsf.png", dpi=300)
    plt.close()

    # PCA
    plt.figure(figsize=(6, 5))
    plt.scatter(transformed[:, 0], transformed[:, 1], c=range(len(transformed)), cmap='viridis', s=2, marker='.')
    plt.title("PCA (PC1 vs PC2)")
    plt.xlabel("PC #1")
    plt.ylabel("PC #2")
    plt.colorbar(label='Production frame')
    plt.savefig(f"{output_prefix}_plot_pca.png", dpi=300)
    plt.close()

    # DCCM
    plt.figure(figsize=(7, 6))
    plt.imshow(dccm, cmap='bwr', vmin=-1, vmax=1)
    plt.title("DCCM")
    plt.colorbar()
    plt.savefig(f"{output_prefix}_plot_dccm.png", dpi=300)
    plt.close()

    print(f"\nAnalysis completed. All the figures have been worked out.")

if __name__ == "__main__":
    run_analysis()
