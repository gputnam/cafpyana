#!/usr/bin/env python3
"""Variante di run_gumple_pipeline.py: tutti i sistematici tranne detvar_spline."""
import os
import sys

workspace_root = os.getcwd()
sys.path.insert(0, workspace_root + "/../gump/")
from sbruce import *
sys.path.insert(0, workspace_root + "/../maple/")
import loaddf
sys.path.insert(0, workspace_root + "/../gumple/")
import custom_post_selection as custom_post_selection

#INPUT = "/exp/sbnd/data/users/gputnam/GUMPLE/sbn-rewgted-19/ICARUS_SpringRun2BNBOff_unblind.df"
#OUTPUT = "/exp/icarus/app/users/marterop/cafpyana/analysis_village/outputs/Icarus_Run2_offbeam_nocosmic_MAPLEonly_REDO.root"

#INPUT = "/exp/sbnd/data/users/gputnam/GUMPLE/sbn-rewgted-19/ICARUS_SpringRun2BNB_unblind.df"
#OUTPUT = "/exp/icarus/app/users/marterop/cafpyana/analysis_village/outputs/ICARUS_Run2_unblind_nocosmic_MAPLEonly.root"

#INPUT = "/exp/sbnd/data/users/gputnam/GUMPLE/sbn-rewgted-19/SBND_SpringLowEMC.df"
#OUTPUT = "/exp/icarus/app/users/marterop/cafpyana/analysis_village/outputs/SBND_Run1_dirt_nocosmic_MAPLEonly_REDO.root"


INPUT = "/exp/sbnd/data/users/gputnam/GUMPLE/sbn-rewgted-19/SBND_SpringBNBOffData.df"
OUTPUT = "/exp/icarus/app/users/marterop/cafpyana/analysis_village/outputs/SBND_Run1_offbeam_nocosmic_MAPLEonly_REDO.root" 

#INPUT = "/exp/sbnd/data/users/gputnam/GUMPLE/sbn-rewgted-19/SBND_SpringMC_Nom.df"
#OUTPUT = "/exp/icarus/app/users/marterop/cafpyana/analysis_village/outputs/SBND_Run1_Nom_nocosmic_MAPLEonly_REDO.root"

TEMP_STAGE1 = OUTPUT.replace(".root", "_stage1_tmp.root")

df, _, _ = loaddf.loadl(
    [INPUT],
    njob=1,
    xsec_univ=False,
    flux_univ=False,
    sep_flux_univ=False,     # flusso: acceso
    sep_g4_univ=False,       # Geant4: acceso
    xsec_spline=False,       # cross-section GENIE: acceso
    pot_spline=False,        # POT systematic: acceso
    match_Enu=False,         # is_mc=True
    load_truth=False,        # is_mc=True
    detvar_spline=True,    # <-- QUESTO è quello che vuoi spento
    spline_dir="rwt_outputs",
    include_syst=False,      # deve restare True, altrimenti scatta l'early return e salta TUTTO
    reweight_aFF=False,
    preselection=custom_post_selection.maple_sel_only,
#    preselection=custom_post_selection.pre_sel_only,
    detector="SBND",
    #detector="ICARUS Run2",
    lightmem=True,
)

export_dataframe_to_uproot(df, TEMP_STAGE1)
run_makesbruce_macro(TEMP_STAGE1, OUTPUT)
os.remove(TEMP_STAGE1)
print(f"Completato: {OUTPUT}")