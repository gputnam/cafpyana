import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from cycler import cycler
import argparse
from functools import reduce
from tqdm.auto import tqdm

import importlib

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
    #mesh = plt.pcolormesh(x_edges, y_edges, z_values.T, cmap='seismic', linewidth=0.1, vmin=0.5, vmax=1.5)
    mesh = plt.pcolormesh(x_edges, y_edges, z_values.T, cmap='seismic', linewidth=0.1)
    
    plt.colorbar(mesh, label='Value')
    plt.title(plot_title)
    plt.xlabel(r'Reconstructed Energy $E_{calo}$ [GeV]')
    plt.ylabel(r'$\delta p$ [GeV/c]')
    
    mesh.get_cmap().set_bad(color='gray')
    plt.savefig('/exp/sbnd/app/users/nrowe/cafpyana/analysis_village/gump/rwt_outputs/2d_ratio_'+output_tag+'.png', dpi=300)
    plt.clf() 

def remake_detvar_maps(detector, DF_DIR, selection=gc.all_cuts, outdir="rwt_outputs"):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if detector == "ICARUS Run2":
        GOAL_POT = 2e20
        DETVAR_FILES = [[DF_DIR + "ICARUSRun2_SpringMCOverlay_rewgt.df"], [DF_DIR + "ICARUSRun2_Spring_Overlay_WMXThXW.df"], [DF_DIR + "ICARUSRun2_Spring_Overlay_WMYZ.df"], [DF_DIR + "ICARUSRun2_Spring_Overlay_SCE.df"]]
        DETVAR_NAMES = ["Nominal", "WMXThetaXW", "WMYZ", "SCE"]
    elif detector == "ICARUS Run4":
        GOAL_POT = 3e20
        DETVAR_FILES = [[DF_DIR + "ICARUSRun4_SpringMCOverlay_rewgt_%i.df" % i for i in range(2)], [DF_DIR + "ICARUSRun4_Spring_Overlay_WMXThXW.df"], [DF_DIR + "ICARUSRun4_Spring_Overlay_WMYZ.df"], [DF_DIR + "ICARUSRun4_Spring_Overlay_SCE.df"]]
        DETVAR_NAMES = ["Nominal", "WMXThetaXW", "WMYZ", "SCE"]
    elif detector == "SBND": 
        GOAL_POT = 1e20
        DETVAR_FILES = [[DF_DIR + "SBNDMCCV_%i.df" % i for i in range(3)], 
                        [DF_DIR + "SBND_SpringMC_WMXThetaXW.df"], 
                        [DF_DIR + "SBND_SpringMC_WMYZ.df"], 
                       ]

        DETVAR_NAMES = [
                        "Nominal", 
                        "WMXThetaXW", 
                        "WMYZ", 
                        ]


        DETVAR_FILES_SMALL = [DF_DIR + "SBND_SpringMC_Nom.df", 
                              DF_DIR + "SBND_SpringMC_2xSCE.df", 
                              DF_DIR + "SBND_SpringMC_0xSCE.df",
                              DF_DIR + "SBND_SpringMC_DENT.df"]

        DETVAR_NAMES_SMALL = ["Nominal", "2xSCE", "0xSCE", "DENT"]
  

    b = [np.array([0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.25, 1.5]), [0.0, 0.2, 0.4, 0.6]]

    detvars, detvarsmatch, detvar_pots = zip(*tqdm([loaddf.loadl(f, preselection=gc.slcfv_cut, include_syst=False, detector=detector, lightmem=True, drops=loaddf.get_std_drops()) for f in DETVAR_FILES]))
    ## Binding E, track splitting req separate loads 
    bind_df, _, bind_pot = loaddf.loadl(DETVAR_FILES[0], preselection=gc.slcfv_cut, include_syst=False, detector=detector, lightmem=True, shift_binding_E=True, drops=loaddf.get_std_drops())

    cv_df = detvars[0].copy()
    cv_pot = detvar_pots[0].copy()

    loaddf.scale_pot(cv_df, cv_pot, GOAL_POT)
    loaddf.scale_pot(bind_df, bind_pot, GOAL_POT)

    cv_df['selected'] = selection(cv_df)
    bind_df['selected'] = selection(bind_df)

    cv_hist = np.histogram2d(*cv_df.loc[cv_df['selected'], ['nu_E_calo', 'del_p']].to_numpy().T, bins=b, weights=cv_df.loc[cv_df['selected'], 'glob_scale'].to_numpy())[0]
    bind_hist = np.histogram2d(*bind_df.loc[bind_df['selected'], ['nu_E_calo', 'del_p']].to_numpy().T, bins=b, weights=bind_df.loc[bind_df['selected'], 'glob_scale'].to_numpy())[0]
    save_histogram(f"{outdir}/{detector.replace(' ','')}_BIND.txt", bind_hist/cv_hist, b[0], b[1])
    del bind_df

    ### track splitting
    if "ICARUS" in detector:
        for split_region in ["Z=0","East Cathode","West Cathode"]:
            trksplt_df, _, trksplt_pot = loaddf.loadl(DETVAR_FILES[0], preselection=gc.slcfv_cut, include_syst=False, detector=detector, lightmem=True, split_tracks=split_region, drops=loaddf.get_std_drops())
            loaddf.scale_pot(trksplt_df, trksplt_pot, GOAL_POT)
            trksplt_df['selected'] = selection(trksplt_df)
            trksplt_hist = np.histogram2d(*trksplt_df.loc[trksplt_df['selected'], ['nu_E_calo', 'del_p']].to_numpy().T, bins=b, weights=trksplt_df.loc[trksplt_df['selected'], 'glob_scale'].to_numpy())[0]
            save_histogram(f"{outdir}/{detector.replace(' ','')}_{split_region.replace(' ','')}_TRKSPLT.txt", trksplt_hist/cv_hist, b[0], b[1])
            del trksplt_df

    ### Other big stuff
    detvars, detvar_pots = loaddf.match_common_evts(detvarsmatch, detvars, detvar_pots)

    for i in range(len(detvars)):
        loaddf.scale_pot(detvars[i], detvar_pots[i], GOAL_POT)
    
    df = detvars[0]
    detvars.extend([syst.v_chi2smear(df), syst.v_chi2hi(df), syst.v_chi2alpha(df), syst.v_chi2beta(df), syst.v_chi2R(df), syst.v_flashscale(df, 1), syst.v_flashscale(df, -1)])
    DETVAR_NAMES.extend(["Smeared dE/dx", "Gain Hi", "EMB Alpha", "EMB Beta", "EMB R", "TrigEffPls", "TrigEffMin"]) 


    hists = []

    for d in detvars:
        print("detvars loop", d)
        d['selected'] = selection(d)
        hists.append(np.histogram2d(*d.loc[d['selected'], ['nu_E_calo', 'del_p']].to_numpy().T, bins=b, weights=d.loc[d['selected'], 'glob_scale'].to_numpy())[0])

    for name, h in zip(DETVAR_NAMES[1:], hists[1:]):
        cv = hists[0]
        if name == "Smeared dE/dx" and detector == "SBND":
            save_histogram(f"{outdir}/{detector.replace(' ','')}_{name.replace('/', '').replace(' ','')}.txt", (2*(h-cv)+cv)/cv, b[0], b[1])
        else:
            save_histogram(f"{outdir}/{detector.replace(' ','')}_{name.replace('/', '').replace(' ','')}.txt", h/cv, b[0], b[1])

    ## SBND SCE now uses a different CV file than the WM samples, this is really cool and not annoying at all
    if detector == "SBND":
        detvars, detvarsmatch, detvar_pots = zip(*tqdm([loaddf.load(f, preselection=gc.slcfv_cut, include_syst=False, detector=detector) for f in DETVAR_FILES_SMALL]))
        detvars, detvar_pots = loaddf.match_common_evts(detvarsmatch, detvars, detvar_pots)

        for i in range(len(detvars)):
            loaddf.scale_pot(detvars[i], detvar_pots[i], GOAL_POT)
        
        b = [np.array([0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.25, 1.5]), [0.0, 0.2, 0.4, 0.6]]
        hists = []
        for d in detvars:
            d['selected'] = selection(d)
            hists.append(np.histogram2d(*d.loc[d['selected'], ['nu_E_calo', 'del_p']].to_numpy().T, bins=b, weights=d.loc[d['selected'], 'glob_scale'].to_numpy())[0])

        for name, h in zip(DETVAR_NAMES_SMALL[1:], hists[1:]):
            save_histogram(f"{outdir}/{detector.replace(' ','')}_{name.replace('/', '').replace(' ','')}.txt", h/hists[0], b[0], b[1])

def resolve_function(func_string):
    """
    Resolves strings like 'gc.all_cuts', 'gc.coworker_cuts', or 
    'my_cuts_module.custom_cut' into callable Python functions.
    """
    if "." not in func_string:
        # Fall back to checking inside the local gump_cuts module if no module prefix given
        if hasattr(gc, func_string):
            return getattr(gc, func_string)
        raise ValueError(f"Function name must be 'module.function' (e.g. 'gc.all_cuts'), got '{func_string}'")

    module_name, func_name = func_string.rsplit(".", 1)

    # Handle common alias 'gc' automatically
    if module_name == "gc":
        module_name = "analysis_village.gump.gump_cuts"  # Or your actual module import path

    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, func_name)
    except (ImportError, AttributeError) as e:
        raise argparse.ArgumentTypeError(f"Could not import '{func_string}': {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run reweighting maps with selection cuts.")
    parser.add_argument(
        "-s", "--selection",
        type=resolve_function,
        default=gc.all_cuts,
        help="Selection function to run (e.g., 'gc.all_cuts' or 'gc.coworker_cuts')"
    )
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        default="rwt_outputs",
        help="Output directory for reweight maps"
    )
    parser.add_argument(
        "-d", "--dfdir",
        type=str,
        default="/exp/sbnd/data/users/gputnam/GUMP/sbn-rewgted-14/",
        help="Output directory for reweight maps"
    )

   
    args = parser.parse_args()

    remake_detvar_maps("SBND", args.dfdir, selection=args.selection, outdir=args.outdir)
    remake_detvar_maps("ICARUS Run2", args.dfdir, selection=args.selection, outdir=args.outdir)
    remake_detvar_maps("ICARUS Run4", args.dfdir,  selection=args.selection, outdir=args.outdir)
