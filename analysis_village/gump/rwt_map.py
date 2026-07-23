import pandas as pd
import os
import sys
from cycler import cycler
import argparse
from functools import reduce
from tqdm.auto import tqdm

workspace_root = os.getcwd()
sys.path.insert(0, workspace_root + "/../../")

import pyanalib.pandas_helpers as ph
import warnings
from pyanalib.split_df_helpers import *
from makedf.util import *
from analysis_village.gump.gump_cuts import *
import analysis_village.gump.PID as PID 
import loaddf
import syst 
import gump_cuts as gc

class FileHistogramFunction:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            line1 = f.readline().strip('# ').split(',')[:-1]
            line2 = f.readline().strip('# ').split(',')[:-1]

            # Extract x metadata
            self.x_edges = np.array([float(l) for l in line1])
            self.y_edges = np.array([float(l) for l in line2]) 

        # 2. Load the actual data grid (skipping the header lines)
        self.grid = np.loadtxt(filename, delimiter=",")

    def __call__(self, x_arr, y_arr):
        # x_arr and y_arr are now numpy arrays (e.g., df.nu_E_calo.values)
        
        # Use digitize to find bin indices for all points at once
        ix = np.digitize(x_arr, self.x_edges) - 1
        iy = np.digitize(y_arr, self.y_edges) - 1
        
        # Handle out-of-bounds (set to a default or clip)
        mask = (ix >= 0) & (ix < self.grid.shape[0]) & \
               (iy >= 0) & (iy < self.grid.shape[1])
        
        # Pre-fill result with 1.0 (your default)
        result = np.ones_like(x_arr, dtype=float)
        
        # Apply grid values where mask is True
        # We use indexing with arrays here
        result[mask] = self.grid[ix[mask], iy[mask]]
        
        return np.nan_to_num(result, nan=1.0)

def save_histogram(filename, hist_values, x_edges, y_edges):
    # Extract metadata to store in the header
    nx, ny = hist_values.shape
    
    # Create a header string
    header=""
    for x in x_edges:
        header += f"{x},"
    header +="\n"
    for y in y_edges:
        header += f"{y},"

    # Save the 2D grid
    print(f"Saving: {filename}")
    np.savetxt(filename, hist_values, header=header, delimiter=",")

def apply_map(df, map_file, col_name):
    if isinstance(map_file, (str, bytes)):
        map_files = [map_file]
    else:
        map_files = map_file

    weights = [[1]*len(df)] 
    for mf in map_files:
        func = FileHistogramFunction(mf)
        weights.append(func(df.nu_E_calo.values, df.del_p.values))
    return pd.DataFrame({col_name: [[row[i] for row in weights] for i in range(len(weights[0]))]}, index=df.index)

def plot_2d_hist_from_file(filename, plot_title, output_tag):
    x_edges = []
    y_edges = []
    data_rows = []

    with open(filename, 'r') as f:
        lines = f.readlines()
        x_edges = [float(x) for x in lines[0].strip('# ').split(',') if x.strip()]
        y_edges = [float(y) for y in lines[1].strip('# ').split(',') if y.strip()]
        
        for line in lines[2:]:
            if line.strip():
                row = [float(val) for val in line.strip().split(',') if val.strip()]
                data_rows.append(row)

    z_values = np.array(data_rows)

    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(x_edges, y_edges)
    mesh = plt.pcolormesh(x_edges, y_edges, z_values.T, cmap='seismic', linewidth=0.1, vmin=0.5, vmax=1.5)
    
    plt.colorbar(mesh, label='Value')
    plt.title(plot_title)
    plt.xlabel(r'Reconstructed Energy $E_{calo}$ [GeV]')
    plt.ylabel(r'$\delta p$ [GeV/c]')
    
    mesh.get_cmap().set_bad(color='gray')
    plt.savefig('/exp/sbnd/app/users/nrowe/cafpyana/analysis_village/gump/rwt_outputs/2d_ratio_'+output_tag+'.png', dpi=300)
    plt.clf() 

def remake_detvar_maps(detector, DF_DIR="/exp/sbnd/data/users/gputnam/GUMP/sbn-rewgted-10/"):
    if detector == "ICARUS Run2":
        GOAL_POT = 2e20
        DETVAR_FILES = [[DF_DIR + "ICARUSRun2_SpringMCOverlay_rewgt.df"], [DF_DIR + "ICARUSRun2_Spring_Overlay_WMXThXW.df"], [DF_DIR + "ICARUSRun2_Spring_Overlay_SCE.df"]]
        DETVAR_NAMES = ["Nominal", "WMXThetaXW", "SCE"]
    elif detector == "ICARUS Run4":
        GOAL_POT = 3e20
        DETVAR_FILES = [[DF_DIR + "ICARUSRun4_SpringMCOverlay_rewgt_%i.df" % i for i in range(2)], [DF_DIR + "ICARUSRun4_Spring_Overlay_SCE.df"]]
        DETVAR_NAMES = ["Nominal", "SCE"]
    elif detector == "SBND": 
        GOAL_POT = 1e20
        DETVAR_FILES = [[DF_DIR + "SBNDMCCV_%i.df" % i for i in range(3)], 
                        [DF_DIR + "SBND_SpringMC_WMXThetaXW.df"], 
                        [DF_DIR + "SBND_SpringMC_WMYZ.df"], 
                       ]

        DETVAR_NAMES = ["Nominal", "WMXThetaXW", "WMYZ"]


        DETVAR_FILES_SMALL = [DF_DIR + "SBND_SpringMC_Nom.df", 
                              DF_DIR + "SBND_SpringMC_2xSCE.df", 
                              DF_DIR + "SBND_SpringMC_0xSCE.df"]

        DETVAR_NAMES_SMALL = ["Nominal", "2xSCE", "0xSCE"]
   
    cols_to_drop = ['is_clear_cosmic', 'crlongtrkdiry', 'p_len', 'mu_E', 'mu_T', 'p_E', 'p_T', 'del_Tp', 'del_phi', 'has_stub',
                    'true_pcand_pdg', 'true_p_dir_x', 'true_p_dir_y', 'true_p_dir_z', 'true_pcand_dir_x', 'true_pcand_dir_y', 
                    'true_pcand_dir_z', 'true_pcand_end_x', 'true_pcand_end_y', 'true_pcand_end_z', 'true_mucand_pdg', 'true_mu_dir_x', 
                    'true_mu_dir_y', 'true_mu_dir_z', 'true_mucand_dir_x', 'true_mucand_dir_y', 'true_mucand_dir_z', 
                    'true_mucand_end_x', 'true_mucand_end_y', 'true_mucand_end_z', 'stub_l0_5cm_dedx','stub_l0_5cm_charge',
                    'stub_l1cm_dedx','stub_l1cm_charge','stub_l2cm_dedx','stub_l2cm_charge','stub_l3cm_dedx','stub_l3cm_charge',
                    'stub_l4cm_dedx','stub_l4cm_charge', 'prot_chi2smear5_of_prot_cand', 'prot_chi2smear5_of_mu_cand', 
                    'mu_chi2smear5_of_mu_cand', 'mu_chi2smear5_of_prot_cand', 'tmatch_pur', 'tmatch_eff', 'true_baseline', 
                    'true_nu_pdg_x', 'true_nu_pdg_y', 'true_nmu_27MeV', 'true_np_20MeV', 'true_np_50MeV', 'true_npi_30MeV', 
                    'is_cosmic', 'flash_sumpe', 'true_mucand_p', 'true_pcand_p', 'mu_true_p', 'p_true_p', 'true_mu_end_x', 
                    'true_p_end_x', 'true_mu_end_y', 'true_p_end_y', 'true_mu_end_z', 'true_p_end_z','crthit', 'true_nu_E', 
                    'p_true_pdg', 'mu_true_pdg', 'mu_chi22lo_of_mu_cand', 'mu_chi22hi_of_mu_cand', 
                    'prot_chi22lo_of_mu_cand', 'prot_chi22hi_of_mu_cand', 'mu_chi22lo_of_prot_cand', 'mu_chi22hi_of_prot_cand', 
                    'prot_chi22lo_of_prot_cand', 'prot_chi22hi_of_prot_cand', 'true_mu_p', 'true_p_p', 'pot_univ']

    detvars, detvarsmatch, detvar_pots = zip(*tqdm([loaddf.loadl(f, preselection=gc.slcfv_cut, include_syst=False, detector=detector, lightmem=True, drops=cols_to_drop) for f in DETVAR_FILES]))
    detvars, detvar_pots = loaddf.match_common_evts(detvarsmatch, detvars, detvar_pots)

    for i in range(len(detvars)):
        loaddf.scale_pot(detvars[i], detvar_pots[i], GOAL_POT)
    
    df = detvars[0]
    detvars.extend([syst.v_chi2smear(df), syst.v_chi2hi(df), syst.v_chi2alpha(df), syst.v_chi2beta(df), syst.v_chi2R(df), syst.v_flashscale(df, 1), syst.v_flashscale(df, -1)])
    DETVAR_NAMES.extend(["Smeared dE/dx", "Gain Hi", "EMB Alpha", "EMB Beta", "EMB R", "TrigEffPls", "TrigEffMin"]) 

    b = [np.array([0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.25, 1.5]), [0.0, 0.2, 0.4, 0.6]]
    hists = []
    for d in detvars:
        d['selected'] = gc.all_cuts(d)
        hists.append(np.histogram2d(*d.loc[d['selected'], ['nu_E_calo', 'del_p']].to_numpy().T, bins=b)[0])

    for name, h in zip(DETVAR_NAMES[1:], hists[1:]):
        save_histogram(f"rwt_outputs/{detector.replace(' ','')}_{name.replace('/', '').replace(' ','')}.txt", h/hists[0], b[0], b[1])

    ## SBND SCE now uses a different CV file than the WM samples, this is really cool and not annoying at all
    if detector == "SBND":
        detvars, detvarsmatch, detvar_pots = zip(*tqdm([loaddf.load(f, preselection=gc.slcfv_cut, include_syst=False, detector=detector) for f in DETVAR_FILES_SMALL]))
        detvars, detvar_pots = loaddf.match_common_evts(detvarsmatch, detvars, detvar_pots)

        for i in range(len(detvars)):
            loaddf.scale_pot(detvars[i], detvar_pots[i], GOAL_POT)
        
        b = [np.array([0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.25, 1.5]), [0.0, 0.2, 0.4, 0.6]]
        hists = []
        for d in detvars:
            d['selected'] = gc.all_cuts(d)
            hists.append(np.histogram2d(*d.loc[d['selected'], ['nu_E_calo', 'del_p']].to_numpy().T, bins=b)[0])

        for name, h in zip(DETVAR_NAMES_SMALL[1:], hists[1:]):
            save_histogram(f"rwt_outputs/{detector.replace(' ','')}_{name.replace('/', '').replace(' ','')}.txt", h/hists[0], b[0], b[1])

if __name__ == "__main__":
    remake_detvar_maps("SBND")
    remake_detvar_maps("ICARUS Run2")
    remake_detvar_maps("ICARUS Run4")
