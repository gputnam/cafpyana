import os
import hashlib
import json

import pandas as pd
import numpy as np
import h5py
from scipy.interpolate import CubicSpline

from tqdm.auto import tqdm
import pyanalib.pandas_helpers as ph
from multiprocess import Pool
from functools import partial
import syst

import gump_cuts as gc
import rwt_map as rw


def tmatch(reco, mc):
    for c in mc.columns:
        if c in reco.columns:
            print(f'duplicate column found! {c}, setting right col to *_true.')
            if(isinstance(c, tuple)):
                mc.rename(columns={c:(c[0]+'_true', '')}, inplace=True)
            else:
                mc.rename(columns={c:c+'_true'}, inplace=True)

    df = ph.multicol_merge(reco.reset_index(), mc.reset_index(),
                           left_on=[("__ntuple", ""), ("entry", ""), ("tmatch_idx", "")],
                           right_on=[("__ntuple", ""), ("entry", ""), ("rec.mc.nu..index", "")],
                           how="left") # start with keeping everything...
    return df

# Dataframe names
EVT = "evt_%i"
WGT = "wgt_%i"
HDR = "hdr_%i"
MC  = "mcnu_%i"
CRT = "crt_%i"
FLASH = "flash_%i"

pot_syst = {'ms3': 0.982714, 'ms2': 0.9887274, 'ms1': 0.99474195, 'cv': 1.0, 'ps1': 1.005, 'ps2': 1.01, 'ps3': 1.015}

xsec_syst = [
    # CCQE
    "GENIEReWeight_SBN_v1_multisigma_VecFFCCQEshape",
    'GENIEReWeight_SBN_v1_multisigma_CoulombCCQE',

    # MEC
    'GENIEReWeight_SBN_v1_multisigma_NormCCMEC',
    'GENIEReWeight_SBN_v1_multisigma_NormNCMEC',
    "GENIEReWeight_SBN_v1_multisigma_DecayAngMEC",

    # RES
    "GENIEReWeight_SBN_v1_multisigma_Theta_Delta2Npi",
    "GENIEReWeight_SBN_v1_multisigma_ThetaDelta2NRad",
    "GENIEReWeight_SBN_v1_multisigma_MaCCRES",
    "GENIEReWeight_SBN_v1_multisigma_MaNCRES",
    "GENIEReWeight_SBN_v1_multisigma_MvCCRES",
    "GENIEReWeight_SBN_v1_multisigma_MvNCRES",
    "GENIEReWeight_SBN_v1_multisigma_RDecBR1gamma",
    "GENIEReWeight_SBN_v1_multisigma_RDecBR1eta",

    # Non-Res
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvpCC1pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvpCC2pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvpNC1pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvpNC2pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvnCC1pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvnCC2pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvnNC1pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvnNC2pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvbarpCC1pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvbarpCC2pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvbarpNC1pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvbarpNC2pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvbarnCC1pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvbarnCC2pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvbarnNC1pi',
    'GENIEReWeight_SBN_v1_multisim_NonRESBGvbarnNC2pi',

    # DIS
    # "GENIEReWeight_SBN_v1_multisim_DISBYVariationResponse",
    'GENIEReWeight_SBN_v1_multisigma_AhtBY',
    'GENIEReWeight_SBN_v1_multisigma_BhtBY',
    'GENIEReWeight_SBN_v1_multisigma_CV1uBY',
    'GENIEReWeight_SBN_v1_multisigma_CV2uBY',

    # COH
    "GENIEReWeight_SBN_v1_multisigma_NormCCCOH",
    "GENIEReWeight_SBN_v1_multisigma_NormNCCOH",

    # FSI
    'GENIEReWeight_SBN_v1_multisigma_MFP_pi',
    'GENIEReWeight_SBN_v1_multisigma_FrCEx_pi',
    'GENIEReWeight_SBN_v1_multisigma_FrInel_pi',
    'GENIEReWeight_SBN_v1_multisigma_FrAbs_pi',
    'GENIEReWeight_SBN_v1_multisigma_FrPiProd_pi',

    # NCEL
    'GENIEReWeight_SBN_v1_multisigma_MaNCEL',
    'GENIEReWeight_SBN_v1_multisigma_EtaNCEL',

    # Systematics introduced by Ar23+
    "CCQETemplateReweight_SBN_v3_LFGToSF_q0bin0",
    "CCQETemplateReweight_SBN_v3_LFGToSF_q0bin1",
    "CCQETemplateReweight_SBN_v3_LFGToSF_q0bin2",
    "CCQETemplateReweight_SBN_v3_LFGToSF_q0bin3",
    "CCQETemplateReweight_SBN_v3_LFGToSF_q0bin4",

    "CCQETemplateReweight_SBN_v3_LFGToHF_q0bin0",
    "CCQETemplateReweight_SBN_v3_LFGToHF_q0bin1",
    "CCQETemplateReweight_SBN_v3_LFGToHF_q0bin2",
    "CCQETemplateReweight_SBN_v3_LFGToHF_q0bin3",
    "CCQETemplateReweight_SBN_v3_LFGToHF_q0bin4",

    "CCQETemplateReweight_SBN_v3_HFToCRPA_q0bin0",
    "CCQETemplateReweight_SBN_v3_HFToCRPA_q0bin1",
    "CCQETemplateReweight_SBN_v3_HFToCRPA_q0bin2",
    "CCQETemplateReweight_SBN_v3_HFToCRPA_q0bin3",
    "CCQETemplateReweight_SBN_v3_HFToCRPA_q0bin4",

    "QEInterference_SBN_v3_QEIntf_dial_0",
    "QEInterference_SBN_v3_QEIntf_dial_1",
    "QEInterference_SBN_v3_QEIntf_dial_2",
    "QEInterference_SBN_v3_QEIntf_dial_3",
    "QEInterference_SBN_v3_QEIntf_dial_4",
    "QEInterference_SBN_v3_QEIntf_dial_5",

    "GENIEReWeight_SBN_v3_FrG4LoE_N",
    "GENIEReWeight_SBN_v3_FrG4M1E_N",
    "GENIEReWeight_SBN_v3_FrG4M2E_N",
    "GENIEReWeight_SBN_v3_FrG4HiE_N",
    "GENIEReWeight_SBN_v3_FrINCLLoE_N",
    "GENIEReWeight_SBN_v3_FrINCLM1E_N",
    "GENIEReWeight_SBN_v3_FrINCLM2E_N",
    "GENIEReWeight_SBN_v3_FrINCLHiE_N",
    "GENIEReWeight_SBN_v3_MFPLoE_N",
    "GENIEReWeight_SBN_v3_MFPM1E_N",
    "GENIEReWeight_SBN_v3_MFPM2E_N",
    "GENIEReWeight_SBN_v3_MFPHiE_N",

    "ZExpPCAWeighter_SBN_v3_MvA_b1",
    "ZExpPCAWeighter_SBN_v3_MvA_b2",
    "ZExpPCAWeighter_SBN_v3_MvA_b3",
    "ZExpPCAWeighter_SBN_v3_MvA_b4",

    "MECq0q3InterpWeighting_SBN_v3_SuSAToVal_MECResponse_q0bin0",
    "MECq0q3InterpWeighting_SBN_v3_SuSAToVal_MECResponse_q0bin1",
    "MECq0q3InterpWeighting_SBN_v3_SuSAToVal_MECResponse_q0bin2",
    "MECq0q3InterpWeighting_SBN_v3_SuSAToVal_MECResponse_q0bin3",
    "MECq0q3InterpWeighting_SBN_v3_SuSAToMar_MECResponse_q0bin0",
    "MECq0q3InterpWeighting_SBN_v3_SuSAToMar_MECResponse_q0bin1",
    "MECq0q3InterpWeighting_SBN_v3_SuSAToMar_MECResponse_q0bin2",
    "MECq0q3InterpWeighting_SBN_v3_SuSAToMar_MECResponse_q0bin3",
]

xsec_cv_rwgt = [
    "ZExpPCAWeighter_SBN_v3_MvA_b1", 
    "CCQEXSecCorr_SBN_v3_CCQEXSecCorr"
]

flux_syst = [
 'expskin_Flux',
 'horncurrent_Flux',
 'kminus_Flux',
 'kplus_Flux',
 'kzero_Flux',
 'nucleoninexsec_Flux',
 'nucleonqexsec_Flux',
 'nucleontotxsec_Flux',
 'piminus_Flux',
 'pioninexsec_Flux',
 'pionqexsec_Flux',
 'piontotxsec_Flux',
 'piplus_Flux'
]

truthvars = {
  "true_E": ("nu_E", ""),
  "true_nu_pdg": ("pdg", ""),
  "true_issig": ("is_sig", ""),
  "true_isothernumucc": ("is_other_numucc", ""),
  "true_isfv": ("is_fv", ""),
  "true_isnc": ("is_nc", ""),
  "genie_mode": ("genie_mode", ""),
  "true_vtx_x": ("pos_x", ""),
  "true_vtx_y": ("pos_y", ""),
  "true_vtx_z": ("pos_z", ""),
  "true_nmu": ("nmu", ""),
  "true_np": ("np", ""),
  "true_nn": ("nn", ""),
  "true_npi": ("npi", ""),
  "true_npi0": ("npi0", ""),
}

detvar_rwt_files = [
  'rwt_outputs/SBND_WMXThetaXW.txt', 
  'rwt_outputs/SBND_WMYZ.txt', 
  ['rwt_outputs/SBND_0xSCE.txt', 'rwt_outputs/SBND_2xSCE.txt'],
  'rwt_outputs/SBND_SmeareddEdx.txt', 
  'rwt_outputs/ICARUSRun2_SmeareddEdx.txt', 
  'rwt_outputs/ICARUSRun2_WMXThetaXW.txt', 
  'rwt_outputs/ICARUSRun4_SmeareddEdx.txt', 
  'rwt_outputs/SBND_GainHi.txt',
  'rwt_outputs/ICARUSRun2_GainHi.txt',
  'rwt_outputs/ICARUSRun4_GainHi.txt',
]

detvar_rwt_lbls = [
  'WireMod_SBND_multisigma_WMXThetaXW', 
  'WireMod_SBND_multisigma_WMYZ', 
  'SCE_SBND_multisigma_SCE', 
  'SBND_PID_Smear',
  'ICARUSRun2_PID_Smear',
  'WireMod_ICARUSRun2_multisigma_WMXThetaXW', 
  'ICARUSRun4_PID_Smear',
  'SBND_PID_Gain',
  'ICARUSRun2_PID_Gain',
  'ICARUSRun4_PID_Gain'
]

std_drops = ['is_clear_cosmic', 'crlongtrkdiry', 'p_len', 'mu_E', 'mu_T', 
             'p_E', 'p_T', 'del_Tp', 'del_phi', 'has_stub',
             'true_pcand_pdg', 'true_p_dir_x', 'true_p_dir_y', 'true_p_dir_z', 
             'true_pcand_dir_x', 'true_pcand_dir_y', 'true_pcand_dir_z', 
             'true_pcand_end_x', 'true_pcand_end_y', 'true_pcand_end_z',
             'true_mucand_pdg', 'true_mu_dir_x', 'true_mu_dir_y', 
             'true_mu_dir_z', 'true_mucand_dir_x', 'true_mucand_dir_y', 
             'true_mucand_dir_z', 'true_mucand_end_x', 'true_mucand_end_y', 
             'true_mucand_end_z', 'stub_l0_5cm_dedx','stub_l0_5cm_charge',
             'stub_l1cm_dedx','stub_l1cm_charge','stub_l2cm_dedx',
             'stub_l2cm_charge','stub_l3cm_dedx','stub_l3cm_charge',
             'stub_l4cm_dedx','stub_l4cm_charge','prot_chi2smear5_of_prot_cand', 
             'prot_chi2smear5_of_mu_cand', 'mu_chi2smear5_of_mu_cand', 
             'mu_chi2smear5_of_prot_cand', 'tmatch_pur', 'tmatch_eff', 
             'true_baseline', 'true_nu_pdg_x', 'true_nu_pdg_y',
             'true_nmu_27MeV', 'true_np_20MeV', 'true_np_50MeV', 
             'true_npi_30MeV', 'is_cosmic', 'flash_sumpe', 'true_mucand_p', 
             'true_pcand_p', 'mu_true_p', 'p_true_p', 'true_mu_end_x', 
             'true_p_end_x', 'true_mu_end_y', 'true_p_end_y', 'true_mu_end_z', 
             'true_p_end_z','crthit', 'true_nu_E', 'p_true_pdg', 'mu_true_pdg', 
             'mu_chi22lo_of_mu_cand', 'mu_chi22hi_of_mu_cand', 
             'prot_chi22lo_of_mu_cand', 'prot_chi22hi_of_mu_cand',
             'mu_chi22lo_of_prot_cand', 'mu_chi22hi_of_prot_cand', 
             'prot_chi22lo_of_prot_cand', 'prot_chi22hi_of_prot_cand', 
             'true_mu_p', 'true_p_p', 'pot_univ']

def get_std_drops():
    return std_drops

def scale_pot(df, pot, desired_pot):
    """Scale DataFrame by desired POT."""
    scale = desired_pot / pot
    print(scale)
    df['glob_scale'] = scale * df.cvwgt
    return pot, scale

def _cache_key(fname, idf, **kwargs):
    """Build a deterministic hash from the input file path, split index, and all keyword args."""
    key_dict = {"fname": os.path.abspath(fname), "idf": idf}
    # Only include serializable kwargs (skip preselection function)
    for k, v in sorted(kwargs.items()):
        if callable(v):
            # Use the function's qualified name so different preselections bust the cache
            key_dict[k] = v.__module__ + "." + v.__qualname__
        else:
            key_dict[k] = v
    raw = json.dumps(key_dict, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _write_cache(cache_file, df, match, pot):
    """Write load_one output to an HDF5 cache file."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    df.to_hdf(cache_file, "df", mode="w")
    match.to_hdf(cache_file, "match", mode="a")
    with h5py.File(cache_file, "a") as cf:
        cf.attrs["pot"] = pot

def load_one(fname, idf,
    detector=None, # One of SBND, ICARUS, ICARUS Run4
    include_syst=True, nuniv=100, spline=False, xsec_univ=False, xsec_spline=False,# systematic handling
    reweight_aFF=False, pot_univ=False, pot_spline=False,
    flux_univ=True, sep_flux_univ=False, detvar_spline=False,
    load_flashes=True, load_truth=True, load_crt=False, match_Enu=True, # load extra information
    offbeampot=False, # POT handling
    preselection=None, # apply preselection cut
    cache_dir=None, # directory to cache output; None disables caching
    flashname=FLASH, hdrname=HDR, evtname=EVT, wgtname=WGT, mcname=MC, crtname=CRT, drops=None, lightmem=False): # override default table names

    assert(detector == "SBND" or detector == "ICARUS Run2" or detector == "ICARUS Run4")

    # Check cache
    if cache_dir is not None:
        cache_hash = _cache_key(fname, idf, detector=detector, include_syst=include_syst,
            nuniv=nuniv, spline=spline, xsec_univ=xsec_univ, xsec_spline=xsec_spline, reweight_aFF=reweight_aFF, pot_univ=pot_univ,
            load_flashes=load_flashes, load_truth=load_truth, load_crt=load_crt,
            match_Enu=match_Enu, offbeampot=offbeampot, preselection=preselection)
        cache_file = os.path.join(cache_dir, cache_hash + ".h5")
        if os.path.exists(cache_file):
            try:
                df = pd.read_hdf(cache_file, "df")
                match = pd.read_hdf(cache_file, "match")
            except Exception as err:
                print(fname, cache_file)
                raise err 
            with h5py.File(cache_file, "r") as cf:
                pot = float(cf.attrs["pot"])
            return df, match, pot

    df =  pd.read_hdf(fname, evtname % idf)
    hdr = pd.read_hdf(fname, hdrname % idf)
    ismc = hdr.ismc.iloc[0] == 1

    # set run 
    if "SBND" in fname:
        df["Run"] = 1
        Run = 1
        det = "SBND"
    elif "ICARUS" in fname and "Run4" in fname:
        df["Run"] = 4
        Run = 4
        det = "ICARUS Run4"
    elif "ICARUS" in fname:
        df["Run"] = 2
        Run = 2
        det = "ICARUS Run2"
    else: assert(False)

    # LOAD FLASHES
    if load_flashes:
        if "flash_maxpe" in df.columns:
          del df["flash_maxpe"]

        flashes = pd.read_hdf(fname, flashname % idf)

        time_name = "firsttime" if detector == "SBND" else "time"
        if ismc: # Scale PE for MC-only
            if detector == "SBND": pe_scale = 0.66
            elif detector == "ICARUS Run2": pe_scale = 0.6
            elif detector == "ICARUS Run4": pe_scale = 0.4
        else:
            pe_scale = 1.0

        intime = (flashes[time_name] > -5) & (flashes[time_name] < 5)
        maxpe = (flashes.totalpe*intime).groupby(level=[0, 1]).max().rename("flash_maxpe")*pe_scale
        df = df.join(maxpe)

    # Apply preselection
    if preselection is not None:
        df = df[preselection(df)]

    match = hdr[["run", "evt"]]
    match_ind = list(match.columns)
    # if needed, include neutrino energy in matching information
    if match_Enu:
        mcdf = pd.read_hdf(fname, mcname % idf)
        match = match.merge(mcdf.nu_E.groupby(level=[0,1]).max().rename("nu_E0"), on=["__ntuple", "entry"], how="left")
        match_ind = list(match.columns)

        # Add in other meta-data to match.
        vtx = pd.DataFrame({
          "detector": detector,
          "Run": Run,
          "x": mcdf.pos_x,
          "y": mcdf.pos_y,
          "z": mcdf.pos_z,
        })
        any_in_AV = gc._fv_cut(vtx, 0, 0, 0, 0).groupby(level=[0,1]).any().rename("AVnu")
        match = match.merge(any_in_AV, on=["__ntuple", "entry"], how="left")

    df = df.merge(match, on=["__ntuple", "entry"], how="left")

    # DROP DUPLICATED EVENTS
    # A "duplicate" is the same physical event appearing in more than one
    # (__ntuple, entry) row of the header — i.e. the same event reconstructed twice.
    # SBND MC legitimately reuses (run, evt) across distinct MC events, so when
    # match_Enu is True we include nu_E0 in the dedup key as a tie-breaker.
    # Drop ALL occurrences (keep=False), not just the extras, from both match and df.
    # Use a MultiIndex.isin mask on df rather than df.merge, so df's existing
    # MultiIndex (__ntuple, entry, rec.slc..index) is preserved.
    dedup_cols = ["run", "evt", "nu_E0"] if match_Enu else ["run", "evt"]
    dup_mask_match = match.duplicated(subset=dedup_cols, keep=False)
    n_dup_pairs = int(match.loc[dup_mask_match, dedup_cols].drop_duplicates().shape[0])
    n_dup_rows  = int(dup_mask_match.sum())
    if n_dup_rows > 0:
        bad_pairs = pd.MultiIndex.from_frame(
            match.loc[dup_mask_match, dedup_cols].drop_duplicates())
        df_pairs = pd.MultiIndex.from_arrays([df[c] for c in dedup_cols])
        df = df[~df_pairs.isin(bad_pairs)]
        match = match[~dup_mask_match]
    print(f"[{os.path.basename(fname)} idf={idf}] dedup: dropped "
          f"{n_dup_pairs} duplicated {tuple(dedup_cols)} keys ({n_dup_rows} hdr rows)")

    match = match.set_index(list(match.columns), append=True).droplevel([0,1]).sort_index()

    # LOAD POT
    if offbeampot:
        if detector == "SBND":
            N_GATES_ON_PER_5e12POT = 1.05104
            pot = hdr.noffbeambnb.sum()/N_GATES_ON_PER_5e12POT*5e12
        elif detector == "ICARUS Run4":
            trig = pd.read_hdf(fname, "trig_%i" % idf)
            N_GATES_ON_PER_5e12POT = 1.0631936867739828
            pot = trig.gate_delta.sum()*(1-1/20.)/N_GATES_ON_PER_5e12POT*5e12
        elif detector == "ICARUS Run2":
            trig = pd.read_hdf(fname, "trig_%i" % idf)
            N_GATES_ON_PER_5e12POT = 1.3886218026202426
            pot = trig.gate_delta.sum()*(1-1/20.)/N_GATES_ON_PER_5e12POT*5e12
    else:
        pot = hdr.pot.sum()
    # LOAD TRUTH
    if load_truth:
        mcdf = pd.read_hdf(fname, mcname % idf)
        mc_tosave = {}
        for setv, load in truthvars.items():
            mc_tosave[setv] = mcdf[load]
        mcdf = pd.DataFrame(mc_tosave, mcdf.index)
        df = df.merge(mcdf, left_on=["__ntuple", "entry", "tmatch_idx"], right_index=True, how="left") 

    # LOAD CRT
    if load_crt:
        if "crthit" in df.columns: del df["crthit"]

        crtdf = pd.read_hdf(fname, crtname % idf)
        crthit = ((crt.time > -1) & (crt.time < 1.8) & (crt.plane != 50)).groupby(level=[0, 1]).any()
        crthit.name = "crthit"
        df = df.join(crthit, on=["__ntuple", "entry"])
        
    df["crthit"] = df.crthit.fillna(False).astype(bool)
    # LOAD WEIGHTS
    if reweight_aFF or include_syst:
        wgt = pd.read_hdf(fname, wgtname % idf) 

    # LOAD AXIAL FORM FACTOR REWEIGHT
    if reweight_aFF:
        rewgt = wgt[xsec_cv_rwgt].copy()
        rewgt["cvwgt"] = 1.
        for w in xsec_cv_rwgt:
            cvcol = "cv" if "cv" in rewgt[w].columns else "morph"
            rewgt["cvwgt"] = rewgt.cvwgt * rewgt[w][cvcol]
        df = df.merge(rewgt.cvwgt.rename("cvwgt"), left_on=["__ntuple", "entry", "tmatch_idx"], right_index=True, how="left")
        df.cvwgt = df.cvwgt.fillna(1.)
    else:
        df["cvwgt"] = 1.

    if drops is not None:
        df.drop(columns=drops, inplace=True, errors='ignore')

    if lightmem:
        type_map = {
            'detector': 'category',
            'Run': 'category',
            'true_isfv': 'Int8',
            'true_isothernumucc': 'Int8',
            'true_issig': 'Int8',
            'true_isnc': 'Int8'
        }
        
        valid_type_map = {col: dtype for col, dtype in type_map.items() if col in df.columns}
        
        df = df.astype(valid_type_map)
        df[df.select_dtypes(include=['float64']).columns] = df.select_dtypes(include=['float64']).astype('float32')

    # EARLY RETURN IF NOT LOADING WEIGHTS
    if not include_syst:
        if cache_dir is not None:
            _write_cache(cache_file, df, match, pot)
        return df, match, pot

    # APPLY WEIGHTS
    skim = {}
    if flux_univ:
        num_to_process = min(100, nuniv)
        
        # Pre-cache the system lookups to avoid doing it inside the inner loops
        system_data = [wgt[s] for s in flux_syst]
        
        new_columns_dict = {}
        for i in range(num_to_process):
            univ_key = "univ_%i" % i
            # np.prod over the pre-cached systems list
            new_columns_dict["flux_univ%i" % i] = np.prod([sys[univ_key] for sys in system_data], axis=0)
            
        # --- FIX HERE: Merging two dictionaries ---
        skim.update(new_columns_dict)

    #if flux_univ:
    #    for i in range(min(100, nuniv)):
    #        skim["flux_univ%i" % i] = np.prod([wgt[s]["univ_%i" % i] for s in flux_syst], axis=0)

    if pot_univ:
        rng = np.random.default_rng(seed=24601) # repeatable random numbers
        rnd = np.clip(rng.normal(size=nuniv), -3, 3)
        for i in range(nuniv):
            wgt_vs = []
            r = rnd[i]
        
            if "ps1" in pot_syst:
                if spline:
                    w = pot_syst
                    spline_ = CubicSpline([-3, -2, -1, 0, 1, 2, 3], 
                            [w["ms3"]/w["cv"], w["ms2"]/w["cv"], w["ms1"]/w["cv"], pd.Series(1, w.index), w["ps1"]/w["cv"], w["ps2"]/w["cv"], w["ps3"]/w["cv"]])
                    s = spline_(r)
                else:
                    s = 1 + (pot_syst["ps1"]/pot_syst["cv"] - 1)*r
            else:
                assert(False)

            wgt_vs.append(s)
            
            skim["pot_univ%i" % i] = np.prod(wgt_vs, axis=0)
    else:
        if "ps1" in pot_syst:
            skim["pot_univ"] = pot_syst["ps1"]/pot_syst["cv"]
        else:
            assert(False)

    multisim_cols = []
    multisigma_cols = []
    if pot_spline:
        for d in ["SBND", "ICARUS Run2", "ICARUS Run4"]:
            col_str = f"multisigma_{d.replace(' ', '')}_POT"
            multisigma_cols.append(col_str)
            if det == d:
                skim[f"{col_str}"] = [list(pot_syst.values()) for _ in range(len(wgt))]
            else:
                skim[f"{col_str}"] = [[1.0]*7 for _ in range(len(wgt))]

    if sep_flux_univ:
        for j, s in enumerate(flux_syst):
            multisim_cols.append(s)
            w = wgt[s]#.fillna(1).replace([np.inf, -np.inf], 1)
            if lightmem:
                w[w.select_dtypes(include=["float64"]).columns] = w.select_dtypes(include=["float64"]).astype("float32")
            stacked_variants = np.vstack([np.nan_to_num(w["univ_%i" % i].to_numpy(), nan=1.0, posinf=1.0, neginf=1.0) for i in range(min(100, nuniv))])
            skim[s] = stacked_variants.T.tolist()

    if xsec_univ:
        rng = np.random.default_rng(seed=24601) # repeatable random numbers
        rnd = np.clip(rng.normal(size=(len(xsec_syst), nuniv)), -3, 3)
        for i in range(nuniv):
            wgt_vs = []
            for j, s in enumerate(xsec_syst):
                r = rnd[j][i]
        
                if "ps1" in wgt[s]:
                    if spline:
                        w = wgt[s].fillna(1).replace([np.inf, -np.inf], 1)
                        spline_ = CubicSpline([-3, -2, -1, 0, 1, 2, 3], 
                                [w["ms3"]/w["cv"], w["ms2"]/w["cv"], w["ms1"]/w["cv"], pd.Series(1, w.index), w["ps1"]/w["cv"], w["ps2"]/w["cv"], w["ps3"]/w["cv"]])
                        s = spline_(r)
                    else:
                        s = 1 + (wgt[s]["ps1"]/wgt[s]["cv"] - 1)*r
                elif "morph" in wgt[s]:
                    s = 1 + (wgt[s]["morph"] - 1)*np.abs(np.clip(r, -1, 1))
                else:
                    assert(False)

                s = np.clip(s, 0, 10)
                wgt_vs.append(s)
            
            skim["xsec_univ%i" % i] = np.clip(np.prod(wgt_vs, axis=0), 0, 30).fillna(1.)

    if xsec_spline:
        for j, s in enumerate(xsec_syst):
            if "ps1" in wgt[s]:
                w = wgt[s].fillna(1).replace([np.inf, -np.inf], 1)
                stacked_variants = np.vstack([
                    np.clip((w["ms3"] / w["cv"]).to_numpy(), 0, 10),
                    np.clip((w["ms2"] / w["cv"]).to_numpy(), 0, 10),
                    np.clip((w["ms1"] / w["cv"]).to_numpy(), 0, 10),
                    np.ones(len(w)),  # Central value ratio is exactly 1.0
                    np.clip((w["ps1"] / w["cv"]).to_numpy(), 0, 10),
                    np.clip((w["ps2"] / w["cv"]).to_numpy(), 0, 10),
                    np.clip((w["ps3"] / w["cv"]).to_numpy(), 0, 10)
                ])

            elif "morph" in wgt[s]:
                w = wgt[s].fillna(1).replace([np.inf, -np.inf], 1)
                if lightmem:
                    w[w.select_dtypes(include=["float64"]).columns] = w.select_dtypes(include=["float64"]).astype("float32")

                stacked_variants = np.vstack([
                    np.ones(len(w)),  # Central value ratio is exactly 1.0
                    np.clip((w["morph"]).to_numpy(), 0, 10)
                ])

            if not 'multisigma' in s:
                col_str = 'multisigma_'+s
            else:
                col_str = s
            skim[col_str] = stacked_variants.T.tolist()
            multisigma_cols.append(col_str)

    else:
        for i, s in enumerate(xsec_syst):
            if "ps1" in wgt[s]:
                skim["%s_univ" % s] = np.clip(wgt[s]["ps1"]/wgt[s]["cv"], 0, 10).fillna(1.)
            elif "morph" in wgt[s]:
                skim["%s_univ" % s] = np.clip(wgt[s]["morph"], 0, 10).fillna(1.)
            else:
                assert(False)

    skim = pd.DataFrame(skim, index=wgt.index)


    mrg = df.merge(skim,
            left_on=["__ntuple", "entry", "tmatch_idx"],
            right_index=True,
            how="left") ## -- save all sllices


    if detvar_spline:
        for s, f in zip(detvar_rwt_lbls, detvar_rwt_files):
            if isinstance(f, (str, bytes)):
                fs = [f]
            else:
                fs = f
            
            allowed_substrings = ["ICARUSRun4", "ICARUSRun2", "SBND"]

            if not all(any(sub in s for sub in allowed_substrings) for s in fs):
                # Find the specific offender to make the error message helpful
                invalid_string = next(s for s in fs if not any(sub in s for sub in allowed_substrings))
                raise ValueError(f"Validation failed: '{invalid_string}' is invalid. Check that your reweight files are all for the same detector.")

            if not 'multisigma' in s:
                col_str = 'multisigma_' + s
            else:
                col_str = s

            # allow for f 
            if det.replace(' ', '') in fs[0]:
                s_df = rw.apply_map(mrg, fs, s)
                mrg[col_str] = s_df
            else:
                mrg[col_str] = [[1.0]*(len(fs)+1) for _ in range(len(mrg))]

            multisigma_cols.append(col_str)

    univ_cols = [col for col in skim.columns if "univ" in col]
    if len(multisigma_cols) > 0:
        nan_mask = mrg[multisigma_cols[0]].isna()
        n_missing = nan_mask.sum()
        for col in multisigma_cols:
            valid_rows = mrg.loc[~nan_mask, col]
            if len(valid_rows) > 0:
                col_len = len(valid_rows.iloc[0])
            else:
                col_len = 7  # Fallback to standard 7-knot default if the whole block is NaN
            
            # 2. Vectorized assignment: Create the block of lists all at once
            default_val = [1.0] * col_len
            mrg.loc[nan_mask, col] = pd.Series([default_val] * n_missing, index=mrg.index[nan_mask])
            #mrg.loc[nan_mask, col] = mrg.loc[nan_mask, col].apply(lambda x: [1.0] * len(mrg.loc[~nan_mask, col].iloc[0]))

    if len(multisim_cols) > 0:
        nan_mask = mrg[multisim_cols[0]].isna()
        n_missing = nan_mask.sum()
        for col in multisim_cols:
            valid_rows = mrg.loc[~nan_mask, col]
            col_len = 100 

            # 2. Vectorized assignment: Create the block of lists all at once
            default_val = [1.0] * col_len
            mrg.loc[nan_mask, col] = pd.Series([default_val] * n_missing, index=mrg.index[nan_mask])
            #mrg.loc[nan_mask, col] = mrg.loc[nan_mask, col].apply(lambda x: [1.0] * 100)

    if len(univ_cols) > 0:
        mrg.loc[np.isnan(mrg[univ_cols[0]]), univ_cols] = 1.0 

    if drops is not None:
        mrg.drop(columns=drops, inplace=True, errors='ignore')
    if cache_dir is not None:
        _write_cache(cache_file, mrg, match, pot)

    return mrg, match, pot


def load(fname, maxdf=None, **kwargs):
    with h5py.File(fname, "r") as f:
        ndf = len([k for k in f.keys() if k.startswith("hdr")])

    if maxdf is None:
        maxdf = ndf

    pots = 0
    dfs = []
    matches = []
    for idf in range(min(ndf, maxdf)):
        df, match, pot = load_one(fname, idf, **kwargs)
        pots += pot
        dfs.append(df)
        matches.append(match)
    df = pd.concat(dfs).reset_index(drop=True)
    match = pd.concat(matches)

    # CROSS-IDF DEDUP
    # `load_one` only sees one idf (split) at a time. The same physical event can
    # show up in more than one idf — `__ntuple` is a per-idf ordinal, not globally
    # unique, so the within-idf check can't catch this. Drop every occurrence of
    # any duplicate after the concat across idfs. Match nu_E0 in the key when it's
    # present (match_Enu=True), since SBND MC reuses (run, evt) across distinct
    # MC events and would over-drop on (run, evt) alone.
    dedup_levels = ["run", "evt"]
    if "nu_E0" in match.index.names:
        dedup_levels.append("nu_E0")
    key = pd.MultiIndex.from_arrays([
        match.index.get_level_values(name) for name in dedup_levels
    ])
    dup_mask = key.duplicated(keep=False)
    if dup_mask.any():
        bad_pairs = pd.MultiIndex.from_arrays([
            key.get_level_values(i)[dup_mask] for i in range(len(dedup_levels))
        ]).unique()
        n_dup_pairs = len(bad_pairs)
        n_dup_rows  = int(dup_mask.sum())
        match = match[~dup_mask]
        df_pairs = pd.MultiIndex.from_arrays([df[name] for name in dedup_levels])
        df = df[~df_pairs.isin(bad_pairs)]
        print(f"[{os.path.basename(fname)}] cross-idf dedup: dropped "
              f"{n_dup_pairs} duplicated {tuple(dedup_levels)} keys "
              f"({n_dup_rows} match rows)")

    return df, match, pots
    
def loadl(flist, progress=True, njob=None, **kwargs):
    if njob is not None:
        pool = Pool(njob)
        m = pool.imap_unordered
    else:
        m = map

    # define function w/ kwargs since multiproc doesn't allow for lambdas
    doload_ = partial(load, **kwargs)

    it = m(doload_, flist)

    if progress:
        it = tqdm(it, total=len(flist))

    dfs = []
    matches = []
    pots = 0
    for df, match, pot in it:
        pots += pot
        dfs.append(df)
        matches.append(match)
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    matches = pd.concat(matches)

    if njob is not None:
        pool.close()

    ###################################
    # ADDED BY NATE Jun 16th '26
    # CROSS-FILE DEDUP
    # `load` only sees one file at a time. The same physical event can
    # show up in more than one file — `__ntuple` is a per-idf ordinal, not globally
    # unique, so the within-file check can't catch this. Drop every occurrence of
    # any duplicate after the concat across idfs. Match nu_E0 in the key when it's
    # present (match_Enu=True), since SBND MC reuses (run, evt) across distinct
    # MC events and would over-drop on (run, evt) alone.

    remaining_columns = ['nu_E_calo', 'slc_vtx_x', 'slc_vtx_y', 'slc_vtx_z']
    dup_mask = df.duplicated(subset=remaining_columns)
    df = df[~dup_mask]
    frac = len(df)/len(dup_mask)
    frac = 1.0
    print(f"dedup: dropped {len(df)} duplicated rows {frac} of the POT remaining. Before POT: {pots}, after POT {pots*frac}.")
    ###################################
    return df, matches, pots*frac

def match_common_evts(mrgs, dfs, pots):
    common_ind = mrgs[0].index
    for m in mrgs[1:]:
        common_ind = common_ind.intersection(m.index)
    common_df = pd.DataFrame({"common": 1}, index=common_ind)

    outdfs = []
    outpots = []
    for m, df, p in zip(mrgs, dfs, pots):
        common_frac = common_ind.size / m.index.size
        outpots.append(common_frac*p)
        outdf = df.merge(common_df, left_on=common_ind.names, right_index=True, how="left")
        outdf["common"] = outdf["common"].fillna(0)
        outdf = outdf[outdf.common == 1]
        outdfs.append(outdf)

    return outdfs, outpots

# Systematic class helpers for what is in these files
class FluxSystematic(syst.WeightSystematic):
    def __init__(self, df, scale="glob_scale"):
        wgts = ["flux_univ%i" % i for i in range(100)]
        super().__init__(df, wgts, scale=scale)

class XSecSystematic(syst.WeightSystematic):
    def __init__(self, df, scale="glob_scale"):
        super().__init__(df, ["%s_univ" % s for s in xsec_syst], avg=False, scale=scale)

class POTSystematic(syst.WeightSystematic):
    def __init__(self, df, scale="glob_scale"):
        super().__init__(df, ["pot_univ"], avg=False, scale=scale)


