"""MAPLE (1mu + N>1 p) dataframe builders for cafpyana.

Port of the CAFANA selection in
NicolaICARUS/MAPLE_GUMP/icarus/helper_eff_cf_FINAL_..._Trigger.h
and the SpillMultiVar output variables in helper_variables.h.

Detector support:
  ICARUS -- the original selection, bit-compatible with the CAFANA port.
  SBND   -- generalization: SBND geometry for FV (high-YZ volume dropped,
            same 10 cm insets / 50 cm zback) and containment; the
            ICARUS-only cosmic rejection (CRT top veto, cryo-light,
            bar-flash) passes trivially; the GUMP cathode-crossing veto
            (no track in the slice may cross the x=0 cathode) joins the
            containment stage; PID uses SBND-calibrated chi2 with the
            same MAPLE thresholds as ICARUS.

PID modes:
  "cafpyana" -- chi2 recomputed from hit dQ/dx via makedf.chi2pid with ICARUS
                gains + calibration (the gump-cafpyana way).  Physics default.
  "cafana"   -- chi2 replicating chi2_ALG on the CAF-stored dedx
                (chi2pid_cafana).  Used for validation against CAFANA-MAPLE.

Both flavors of chi2 are always stored; `pid_mode` picks which one drives
the muon/proton candidate identification, the cut booleans, and the sBruce
variables.  The cut chain for the *other* mode is also stored, with a
"_alt" suffix, so migration between the two PIDs can be studied.

Selection option:
  selection="none"   -- keep all slices, with all cut booleans (default)
  selection="presel" -- keep slices passing the PID-free MAPLE preselection
                        (sanity + FV + CRT veto + cryo-light + containment)
  selection="full"   -- keep slices passing the full MAPLE selection
"""
import numpy as np
import pandas as pd

from pyanalib.pandas_helpers import *
from makedf.util import *
from makedf.makedf import (
    loadbranches, make_slcdf, make_trkdf, make_trkhitdf, make_crthitdf,
    make_hdrdf, make_triggerdf, make_potdf_bnb, make_mcnudf,
    make_genie_evtrec_df, _build_genie_evtrec_df,
)
from makedf import chi2pid

from analysis_village.maple.maple_cuts import *
from analysis_village.maple import chi2pid_cafana
from makedf.branches import (
    crtpmtbranches, shwbranches,
    mcbranches, mcprimbranches, mcprimvisEbranches, trueparticlebranches,
)

# PFPID enum from the helper header
PID_UNKNOWN, PID_PROTON, PID_PION, PID_SHOWER, PID_OTHER = 0, 1, 2, 3, 4
# TruthClass: 0=1mu1p, 1=1muNp, 2=Other, 3=Cosmic, 4=Invalid
CLS_1MU1P, CLS_1MUNP, CLS_OTHER, CLS_COSMIC, CLS_INVALID = 0, 1, 2, 3, 4

CALO_VARIATIONS = ["cv", "alpha_p", "alpha_m", "beta_p", "beta_m", "R_p", "R_m"]
SCALE_SMEAR_VARIATIONS = ["lo", "hi", "2lo", "2hi", "smear5", "smear13", "sqsmear15"]


def _flatcols(df):
    """Flatten a multiindex-column dataframe to underscore-joined names."""
    df = df.copy()
    df.columns = ["_".join([str(c) for c in (col if isinstance(col, tuple) else (col,)) if str(c) != ""])
                  for col in df.columns]
    return df


def _reindex(series, index, fill):
    return series.reindex(index).fillna(fill)


# =====================================================================
# Truth classification (classification_type_MC port)
# =====================================================================
def maple_truth_classdf(f, det="ICARUS"):
    """Per-(entry, mc.nu index) MAPLE truth classification.

    Returns a DataFrame with columns:
      maple_class (0=1mu1p, 1=1muNp, 2=Other, 4=Invalid -- Cosmic is a
      slice-level concept, handled in the evt df), n_mu, mu_length,
      n_p_above40, veto, uncontained, true_visible_Enu
    """
    mc = _flatcols(loadbranches(f["recTree"], mcbranches).rec.mc.nu)
    if mc.empty:
        out = pd.DataFrame(columns=["maple_class", "n_mu", "mu_length", "n_p_above40",
                                    "veto", "uncontained", "true_visible_Enu"])
        return out

    nuidx = mc.index

    prim = _flatcols(loadbranches(f["recTree"], mcprimbranches + mcprimvisEbranches).rec.mc.nu.prim)
    prim.index.names = ["entry", "inu", "iprim"]
    tp = _flatcols(loadbranches(f["recTree"], trueparticlebranches).rec.true_particles)
    tp.index.names = ["entry", "itp"]

    # Only primaries with G4ID >= 0 and cryostat >= 0 participate (C++ `continue`)
    prim = prim[(prim.G4ID >= 0) & (prim.cryostat >= 0)]

    apdg = np.abs(prim.pdg)
    is_mu = apdg == 13
    is_p = apdg == 2212
    is_cpi = apdg == 211
    is_pi0 = apdg == 111
    is_gamma = apdg == 22
    is_e = apdg == 11
    is_charged = is_mu | is_p | is_cpi | is_e

    # own deposited energy on plane 2 in the primary's cryostat [MeV]
    visE_own = np.where(prim.cryostat == 0, prim.plane_I0_I2_visE, prim.plane_I1_I2_visE) * 1000.0
    prim = prim.assign(visE_own=visE_own)

    # ---- daughters: true_particles with parent == prim.G4ID (same entry) ----
    prim_r = prim.reset_index()
    tp_r = tp.reset_index()
    m = prim_r[["entry", "inu", "iprim", "G4ID", "cryostat"]].merge(
        tp_r, left_on=["entry", "G4ID"], right_on=["entry", "parent"],
        suffixes=("", "_d"))
    if len(m):
        # daughter visE evaluated at the PRIMARY's cryostat (as in C++)
        m["d_visE"] = np.where(m.cryostat == 0, m.plane_I0_I2_visE, m.plane_I1_I2_visE) * 1000.0
        m["d_apdg"] = np.abs(m.pdg)
        m["d_is_gamma"] = m.d_apdg == 22
        m["d_charged"] = m.d_apdg.isin([13, 2212, 211, 11])
        # daughter containment check: skip cryostat<0 or end == -9999
        d_valid_cont = (m.cryostat_d >= 0) & m.d_charged & \
            ~((m.end_x == -9999) | (m.end_y == -9999) | (m.end_z == -9999))
        m["d_uncont"] = d_valid_cont & ~maple_isInContained(m.end_x, m.end_y, m.end_z, det=det)
        # note: for the pi0-gamma veto and visE sums, C++ has no cryostat check
        # on the daughter itself
        g = m.groupby(["entry", "inu", "iprim"])
        dep_daughters = g.d_visE.sum()
        pi0_gamma_veto = g.apply(lambda x: bool(((x.d_is_gamma) & (x.d_visE > PION_KE_MIN)).any()))
        d_uncont_any = g.d_uncont.any()
    else:
        dep_daughters = pd.Series(dtype=float)
        pi0_gamma_veto = pd.Series(dtype=bool)
        d_uncont_any = pd.Series(dtype=bool)

    dep_daughters = _reindex(dep_daughters, prim.index, 0.0)
    pi0_gamma_veto = _reindex(pi0_gamma_veto, prim.index, False).astype(bool)
    d_uncont_any = _reindex(d_uncont_any, prim.index, False).astype(bool)

    # ---- per-primary vetoes ----
    cpi_veto = is_cpi & (prim.visE_own > PION_KE_MIN)
    pi0_veto = is_pi0 & pi0_gamma_veto
    gamma_veto = is_gamma & ((prim.visE_own + dep_daughters) > PION_KE_MIN)
    p_depE = prim.visE_own + dep_daughters
    p_above = is_p & (p_depE > PROTON_KE_MIN)
    p_below = is_p & ~(p_depE > PROTON_KE_MIN)
    prim_veto = cpi_veto | pi0_veto | gamma_veto | p_below

    # ---- containment (all_contained_mc): charged primaries, no -9999 skip ----
    prim_uncont = is_charged & ~maple_isInContained(prim.end_x, prim.end_y, prim.end_z, det=det)

    # ---- per-nu aggregation ----
    grp = prim.groupby(level=["entry", "inu"])

    def agg(series, fill=False):
        s = series.groupby(level=["entry", "inu"]).any() if series.dtype == bool else \
            series.groupby(level=["entry", "inu"]).sum()
        s.index.names = nuidx.names
        return _reindex(s, nuidx, fill)

    n_mu = agg(is_mu.astype(int), 0)
    n_p_above = agg(p_above.astype(int), 0)
    veto_any = agg(prim_veto, False).astype(bool)
    uncont_any = (agg(prim_uncont, False).astype(bool) |
                  agg(d_uncont_any, False).astype(bool))

    # last muon in loop order sets muon_length / E_mu (C++ overwrites per muon)
    mu_rows = prim[is_mu]
    mu_last = mu_rows.groupby(level=["entry", "inu"]).last()
    mu_last.index.names = nuidx.names
    mu_length = _reindex(mu_last.length, nuidx, np.nan)
    mu_p_GeV = np.sqrt(mu_last.genp_x**2 + mu_last.genp_y**2 + mu_last.genp_z**2)
    E_mu_vis = np.sqrt((mu_p_GeV * 1000.0)**2 + MUON_MASS**2)  # MeV
    E_mu_vis = _reindex(E_mu_vis, nuidx, 0.0)

    # visible energy: protons above threshold contribute KE(genp) + Eb
    p_rows = prim[p_above]
    p_ke = kinetic_energy(PROTON_MASS, np.sqrt(p_rows.genp_x**2 + p_rows.genp_y**2 + p_rows.genp_z**2))
    E_p_sum = (p_ke + PROTON_BINDING_ENERGY).groupby(level=["entry", "inu"]).sum()
    E_p_sum.index.names = nuidx.names
    E_p_sum = _reindex(E_p_sum, nuidx, 0.0)

    true_visible_Enu = (E_p_sum + E_mu_vis) / 1000.0  # GeV

    # ---- classification ----
    pos_nan = mc.position_x.isna() | mc.position_y.isna() | mc.position_z.isna()
    not_numucc = (np.abs(mc.pdg) != 14) | (mc.iscc == 0)
    not_av = ~maple_isInActive(mc.position_x, mc.position_y, mc.position_z, det=det)
    not_fv = ~maple_isInFV(mc.position_x, mc.position_y, mc.position_z, det=det)
    good_mu = (n_mu == 1) & (mu_length > MIN_MUON_LENGTH) & (mu_length < MAX_MUON_LENGTH)

    maple_class = np.select(
        [pos_nan,
         not_numucc | not_av | not_fv | veto_any | uncont_any,
         good_mu & (n_p_above == 1),
         good_mu & (n_p_above > 1)],
        [CLS_INVALID, CLS_OTHER, CLS_1MU1P, CLS_1MUNP],
        default=CLS_OTHER)

    return pd.DataFrame({
        "maple_class": maple_class,
        "n_mu": n_mu,
        "mu_length": mu_length,
        "n_p_above40": n_p_above,
        "veto": veto_any,
        "uncontained": uncont_any,
        "true_visible_Enu": true_visible_Enu,
    }, index=nuidx)


# =====================================================================
# Per-pfp selection machinery
# =====================================================================
def _run_selection(P, chi2mu, chi2p):
    """find_muon + id_pfp + count_particles for one chi2 flavor.

    P: flat per-pfp frame (index entry, slc, pfp).
    Returns (mu_ilocs [per-slice Index of muon rows], pid [Series],
             counts DataFrame per slice).
    All masks mirror the C++ skip conditions, including NaN behavior.
    """
    # ---- find_muon ----
    keep = P.start_x.notna() & P.len.notna() \
        & ~(P.trackScore < PRIMARY_TRACK_SCORE) \
        & ~(P.dist_start > 10.0) \
        & ~((P.len < MIN_MUON_LENGTH) | (P.len > MAX_MUON_LENGTH)) \
        & P.prim_pfp \
        & P.contained10 \
        & (P.end_x * P.vtx_x > 0) \
        & chi2mu.notna() & chi2p.notna() \
        & ~(chi2mu > MAX_CHI2_MUON) & ~(chi2p < MIN_CHI2_PROTON)

    cand = P[keep]
    if len(cand):
        mu_ilocs = cand.len.groupby(level=[0, 1]).idxmax()
    else:
        mu_ilocs = pd.Series(dtype=object)

    # ---- id_pfp (all pfps; the muon row is excluded from counts below) ----
    unknown0 = (~P.prim_pfp) | P.start_x.isna() | P.end_x.isna() | P.len.isna()
    unknown1 = P.min_dist > VTX_MAX_DIST
    no_calo = P.ncalo == 0
    is_pion = (chi2p >= CHI2_PROTON_PION) & (P.dist_start < 10.0) & (P.ke_pion >= PION_KE_MIN)
    is_proton = (chi2p < CHI2_PROTON_PION) & (P.dist_start < 10.0) & (P.ke_proton >= PROTON_KE_MIN)
    shw_unknown = P.shw_energy2.isna()
    is_shower = P.shw_energy2 * 1000.0 > PION_KE_MIN

    pid = pd.Series(np.select(
        [unknown0, unknown1, no_calo, is_pion, is_proton, shw_unknown, is_shower],
        [PID_UNKNOWN, PID_UNKNOWN, PID_UNKNOWN, PID_PION, PID_PROTON, PID_UNKNOWN, PID_SHOWER],
        default=PID_OTHER), index=P.index)

    # exclude the muon candidate from particle counting
    pid_nomu = pid.copy()
    if len(mu_ilocs):
        pid_nomu.loc[pd.Index(mu_ilocs.values)] = -1

    grp = lambda s: s.groupby(level=[0, 1]).sum()
    counts = pd.DataFrame({
        "n_proton": grp((pid_nomu == PID_PROTON).astype(int)),
        "n_pion": grp((pid_nomu == PID_PION).astype(int)),
        "n_shower": grp((pid_nomu == PID_SHOWER).astype(int)),
        "n_other": grp((pid_nomu == PID_OTHER).astype(int)),
    })

    return mu_ilocs, pid_nomu, counts


def _cut_chain(S, counts, has_mu):
    """MAPLE cut booleans for one PID flavor, given slice frame S."""
    cut_muon = has_mu
    cut_np = counts.n_proton > 1 if NP_MODE else counts.n_proton > 0
    cut_0pi = counts.n_pion == 0
    cut_0shwother = (counts.n_shower == 0) & (counts.n_other == 0)
    maple_sel = S.maple_presel & cut_muon & cut_np & cut_0pi & cut_0shwother

    # MaxCutPassed replication (cutflow order: sanity, FV, CRT veto,
    # cryo-light ["barycenter"], containment [+ SBND cathode veto], muon,
    # proton, pion, shower). cut_cathode is always True on ICARUS, so the
    # numbering there is unchanged from the CAFANA port.
    maxcut = np.select(
        [~S.cut_sanity, ~S.cut_fv, ~S.cut_crtveto, ~S.cut_cryo,
         ~(S.cut_contained & S.cut_cathode), ~cut_muon, ~cut_np, ~cut_0pi, ~cut_0shwother],
        [1, 2, 3, 4, 5, 6, 7, 8, 9], default=10)

    return pd.DataFrame({
        "cut_muon": cut_muon, "cut_np": cut_np, "cut_0pi": cut_0pi,
        "cut_0shwother": cut_0shwother, "maple_sel": maple_sel,
        "maxcut": maxcut,
    }, index=S.index)


# =====================================================================
# Main evt builder
# =====================================================================
def make_maple_evt_df(f, selection="none", pid_mode="cafpyana", do_calo_syst=True):
    det = loadbranches(f["recTree"], ["rec.hdr.det"]).rec.hdr.det
    if det.empty:
        return pd.DataFrame()
    if 1 == det.unique():
        DETECTOR = "SBND"
    elif 2 == det.unique():
        DETECTOR = "ICARUS"
    else:
        raise ValueError("MAPLE needs rec.hdr.det == 1 (SBND) or 2 (ICARUS); got %s" % det.unique())
    run = loadbranches(f["recTree"], ["rec.hdr.run"]).rec.hdr.run
    RUN = 1 if DETECTOR == "SBND" else (2 if run.iloc[0] < 12960 else 4)
    ismc = bool(loadbranches(f["recTree"], ["rec.hdr.ismc"]).rec.hdr.ismc.iloc[0])

    # ------------------------------------------------------------------
    # slice frame
    # ------------------------------------------------------------------
    slcdf = make_slcdf(f)

    S = pd.DataFrame({
        "vtx_x": slcdf.slc.vertex.x,
        "vtx_y": slcdf.slc.vertex.y,
        "vtx_z": slcdf.slc.vertex.z,
        "charge_center_z": slcdf.slc.charge_center.z,
        "nu_score": slcdf.slc.nu_score,
        "tmatch_idx": slcdf.slc.tmatch.idx,
        "tmatch_eff": slcdf.slc.tmatch.eff,
        "tmatch_pur": slcdf.slc.tmatch.pur,
        "E_nu_true": slcdf.slc.truth.E,
        "true_pdg": slcdf.slc.truth.pdg,
        "true_iscc": slcdf.slc.truth.iscc,
        "true_genie_mode": slcdf.slc.truth.genie_mode,
        "true_vtx_x": slcdf.slc.truth.position.x,
        "true_vtx_y": slcdf.slc.truth.position.y,
        "true_vtx_z": slcdf.slc.truth.position.z,
    })
    S["slice_index"] = S.index.get_level_values(1)

    # ------------------------------------------------------------------
    # per-pfp frame
    # ------------------------------------------------------------------
    trkdf = make_trkdf(f, False)  # NO vertex-distance pre-filter: MAPLE needs all pfps
    # core shwbranches, restricted to what the file carries (ICARUS flat CAFs
    # lack the SBND-only bestplane_for_* branches)
    keys = set(f["recTree"].keys())
    shwdf = loadbranches(f["recTree"], [b for b in shwbranches if b in keys]).rec.slc.reco.pfp.shw
    shwdf.index.names = trkdf.index.names

    P = pd.DataFrame({
        "start_x": trkdf.pfp.trk.start.x,
        "start_y": trkdf.pfp.trk.start.y,
        "start_z": trkdf.pfp.trk.start.z,
        "end_x": trkdf.pfp.trk.end.x,
        "end_y": trkdf.pfp.trk.end.y,
        "end_z": trkdf.pfp.trk.end.z,
        "len": trkdf.pfp.trk.len,
        "dir_x": trkdf.pfp.trk.dir.x,
        "dir_y": trkdf.pfp.trk.dir.y,
        "dir_z": trkdf.pfp.trk.dir.z,
        "p_muon": trkdf.pfp.trk.rangeP.p_muon,
        "p_pion": trkdf.pfp.trk.rangeP.p_pion,
        "p_proton": trkdf.pfp.trk.rangeP.p_proton,
        "trackScore": trkdf.pfp.trackScore,
        "true_genp_x": trkdf.pfp.trk.truth.p.genp.x,
        "shw_energy2": shwdf.plane.I2.energy,
    })
    P["prim_pfp"] = trkdf.pfp.parent_is_primary.fillna(False).astype(bool)

    # broadcast slice vertex onto pfps
    P = P.join(S[["vtx_x", "vtx_y", "vtx_z"]])

    P["dist_start"] = np.sqrt((P.start_x - P.vtx_x)**2 + (P.start_y - P.vtx_y)**2 + (P.start_z - P.vtx_z)**2)
    dist_end = np.sqrt((P.end_x - P.vtx_x)**2 + (P.end_y - P.vtx_y)**2 + (P.end_z - P.vtx_z)**2)
    # std::min(a, b) semantics: b if b < a else a  (NaN b -> a; NaN a -> NaN)
    P["min_dist"] = np.where(np.isnan(dist_end), P.dist_start, np.minimum(P.dist_start, dist_end))
    P["contained10"] = maple_isInContained(P.end_x, P.end_y, P.end_z, det=DETECTOR)
    P["ke_pion"] = kinetic_energy(PION_MASS, np.sqrt((P.dir_x * P.p_pion)**2 + (P.dir_y * P.p_pion)**2 + (P.dir_z * P.p_pion)**2))
    P["ke_proton"] = kinetic_energy(PROTON_MASS, np.sqrt((P.dir_x * P.p_proton)**2 + (P.dir_y * P.p_proton)**2 + (P.dir_z * P.p_proton)**2))

    # ------------------------------------------------------------------
    # PID: both flavors
    # ------------------------------------------------------------------
    trkhitdf = make_trkhitdf(f)

    # number of plane-2 calo points (compute_chi2 returns {} when empty)
    ncalo = trkhitdf.groupby(level=[0, 1, 2]).size()
    P["ncalo"] = _reindex(ncalo, P.index, 0).astype(int)

    # CAFANA-compat chi2 on stored dedx
    cafana = chi2pid_cafana.chi2_cafana(trkhitdf)
    P["chi2u_cafana"] = cafana.chi2_mu
    P["chi2p_cafana"] = cafana.chi2_pro

    # gump-style chi2 on recomputed dE/dx (detector gains + calibration)
    trkhitdf["dedx_redo"] = chi2pid.dedx(trkhitdf, gain=DETECTOR, calibrate=DETECTOR, isMC=ismc)
    P["chi2u_cafpyana"] = chi2pid.chi2u(trkhitdf, dedxname="dedx_redo")[0]
    P["chi2p_cafpyana"] = chi2pid.chi2p(trkhitdf, dedxname="dedx_redo")[0]

    # calorimetric variations (ported from gump make_pandora_no_cuts_df,
    # including the gump detector-specific scale sizes)
    if do_calo_syst:
        if DETECTOR == "ICARUS":
            scale_lo, scale_hi, scale_2lo, scale_2hi = 0.99, 1.01, 0.98, 1.02
            calo_var_params = chi2pid.ICARUS_CALO_VARIATIONS
        else:
            scale_lo, scale_hi, scale_2lo, scale_2hi = 0.98, 1.02, 0.96, 1.04
            calo_var_params = chi2pid.SBND_CALO_VARIATIONS
        trkhitdf["dedx_lo"] = chi2pid.dedx(trkhitdf, gain=DETECTOR, calibrate=DETECTOR, isMC=ismc, scale=scale_lo)
        trkhitdf["dedx_hi"] = chi2pid.dedx(trkhitdf, gain=DETECTOR, calibrate=DETECTOR, isMC=ismc, scale=scale_hi)
        trkhitdf["dedx_2lo"] = chi2pid.dedx(trkhitdf, gain=DETECTOR, calibrate=DETECTOR, isMC=ismc, scale=scale_2lo)
        trkhitdf["dedx_2hi"] = chi2pid.dedx(trkhitdf, gain=DETECTOR, calibrate=DETECTOR, isMC=ismc, scale=scale_2hi)
        trkhitdf["dedx_smear5"] = chi2pid.dedx(trkhitdf, gain=DETECTOR, calibrate=DETECTOR, isMC=ismc, smear=0.05)
        trkhitdf["dedx_smear13"] = chi2pid.dedx(trkhitdf, gain=DETECTOR, calibrate=DETECTOR, isMC=ismc, smear=0.13)
        trkhitdf["dedx_sqsmear15"] = chi2pid.dedx(trkhitdf, gain=DETECTOR, calibrate=DETECTOR, isMC=ismc, sqrt_smear=0.15)
        for c_var in CALO_VARIATIONS:
            trkhitdf["dedx_%s" % c_var] = chi2pid.dedx(
                trkhitdf, gain=DETECTOR, calibrate=DETECTOR, isMC=ismc,
                new_calo_params=calo_var_params[c_var])

        for var in SCALE_SMEAR_VARIATIONS + CALO_VARIATIONS:
            P["chi2u_%s" % var] = chi2pid.chi2u(trkhitdf, dedxname="dedx_%s" % var)[0]
            P["chi2p_%s" % var] = chi2pid.chi2p(trkhitdf, dedxname="dedx_%s" % var)[0]

        # Don't apply variations to (Overlay) cosmics
        cosmic = P.true_genp_x.isna()
        for var in SCALE_SMEAR_VARIATIONS + CALO_VARIATIONS:
            P.loc[cosmic, "chi2u_%s" % var] = P.loc[cosmic, "chi2u_cafpyana"]
            P.loc[cosmic, "chi2p_%s" % var] = P.loc[cosmic, "chi2p_cafpyana"]

    # ------------------------------------------------------------------
    # spill-level inputs
    # ------------------------------------------------------------------
    entries = S.index.get_level_values(0)

    if DETECTOR == "ICARUS":
        crtpmt = loadbranches(f["recTree"], crtpmtbranches).rec.crtpmt_matches

        # cryo_selection_from_light
        inwin = (crtpmt.flashGateTime > CRYO_LIGHT_TMIN) & (crtpmt.flashGateTime < CRYO_LIGHT_TMAX)
        haspe = ~(crtpmt.flashPE < CRYO_LIGHT_PE_THRESHOLD)
        west = (inwin & haspe & (crtpmt.flashPosition.x > 0)).groupby(level=0).any()
        east = (inwin & haspe & (crtpmt.flashPosition.x < 0)).groupby(level=0).any()
        perentry = pd.DataFrame(index=pd.Index(entries.unique(), name="entry"))
        perentry["west"] = _reindex(west, perentry.index, False).astype(bool)
        perentry["east"] = _reindex(east, perentry.index, False).astype(bool)
        perentry["cryo_light"] = np.select(
            [perentry.west & perentry.east, perentry.west, perentry.east],
            [2, 1, 0], default=-1)

        # k_pe: max flash PE in the cryo-light window, no PE threshold, default 0
        maxpe = crtpmt.flashPE[inwin].groupby(level=0).max()
        perentry["flash_maxpe"] = _reindex(maxpe, perentry.index, 0.0)

        # bar_flash: first in-window match on each side (MC/data window differs)
        btmin, btmax = (BAR_FLASH_TMIN_MC, BAR_FLASH_TMAX_MC) if ismc else (BAR_FLASH_TMIN_DATA, BAR_FLASH_TMAX_DATA)
        barwin = (crtpmt.flashGateTime > btmin) & (crtpmt.flashGateTime < btmax)
        for side, cond in [("west", crtpmt.flashPosition.x > 0), ("east", crtpmt.flashPosition.x < 0)]:
            first = crtpmt[barwin & cond].groupby(level=0).first()
            perentry["bar_z_" + side] = first.flashPosition.z.reindex(perentry.index)
            perentry["bar_x_" + side] = first.flashPosition.x.reindex(perentry.index)

        # kCRTNeutrino: top-CRT veto
        crt = make_crthitdf(f)
        vetohit = ((crt.time > CRT_VETO_TMIN) & (crt.time < CRT_VETO_TMAX) &
                   (crt.plane > CRT_VETO_PLANE_MIN) & (crt.plane < CRT_VETO_PLANE_MAX)).groupby(level=0).any()
        perentry["crtveto"] = _reindex(vetohit, perentry.index, False).astype(bool)
    else:
        # SBND: the ICARUS-only cosmic rejection is dropped; fill sentinels
        # (cryo_light=-1, no bar flash, no CRT veto) so the evt schema is
        # identical across detectors. The cut booleans are forced True below.
        perentry = pd.DataFrame(index=pd.Index(entries.unique(), name="entry"))
        perentry["cryo_light"] = -1
        perentry["flash_maxpe"] = np.nan
        for side in ("west", "east"):
            perentry["bar_z_" + side] = np.nan
            perentry["bar_x_" + side] = np.nan
        perentry["crtveto"] = False

    S = S.join(perentry[["cryo_light", "flash_maxpe", "bar_z_west", "bar_x_west",
                         "bar_z_east", "bar_x_east", "crtveto"]])

    # ------------------------------------------------------------------
    # PID-free cut chain
    # ------------------------------------------------------------------
    S["cut_sanity"] = S.vtx_x.notna() & S.vtx_y.notna() & S.vtx_z.notna() & S.charge_center_z.notna()
    S["cut_fv"] = maple_isInFV(S.vtx_x, S.vtx_y, S.vtx_z, det=DETECTOR)
    if DETECTOR == "ICARUS":
        S["cut_crtveto"] = ~S.crtveto
        slice_cryo = np.select([S.vtx_x < 0, S.vtx_x > 0], [0, 1], default=-1)
        S["slice_cryo"] = slice_cryo
        S["cut_cryo"] = (S.cryo_light != -1) & ((S.cryo_light == 2) | (slice_cryo == S.cryo_light))
    else:
        # SBND: CRT veto and cryo-light cuts pass trivially (see above)
        S["cut_crtveto"] = True
        S["slice_cryo"] = -1
        S["cut_cryo"] = True

    valid_c = P.start_x.notna() & P.end_x.notna() & P.len.notna()
    bad_contain = valid_c & ((P.end_x * P.vtx_x < 0) | ~P.contained10)
    any_bad = bad_contain.groupby(level=[0, 1]).any()
    S["cut_contained"] = ~_reindex(any_bad, S.index, False).astype(bool)

    # SBND cathode-crossing veto (GUMP cathode_cut, extended to all tracks in
    # the slice); always True on ICARUS so the cut chain is unchanged there
    if DETECTOR == "SBND":
        cross = maple_sbnd_cathode_crossing(
            P.vtx_x[valid_c], P.vtx_y[valid_c], P.vtx_z[valid_c],
            P.end_x[valid_c], P.end_y[valid_c], P.end_z[valid_c])
        any_cross = pd.Series(cross, index=P.index[valid_c]).groupby(level=[0, 1]).any()
        S["cut_cathode"] = ~_reindex(any_cross, S.index, False).astype(bool)
    else:
        S["cut_cathode"] = True

    S["maple_presel"] = S.cut_sanity & S.cut_fv & S.cut_crtveto & S.cut_cryo & \
        S.cut_contained & S.cut_cathode

    # ------------------------------------------------------------------
    # PID-dependent selection: primary + alternate flavor
    # ------------------------------------------------------------------
    if pid_mode == "cafpyana":
        chi_pri = ("chi2u_cafpyana", "chi2p_cafpyana")
        chi_alt = ("chi2u_cafana", "chi2p_cafana")
    elif pid_mode == "cafana":
        chi_pri = ("chi2u_cafana", "chi2p_cafana")
        chi_alt = ("chi2u_cafpyana", "chi2p_cafpyana")
    else:
        raise ValueError("pid_mode must be 'cafpyana' or 'cafana'")

    mu_ilocs, pid_s, counts = _run_selection(P, P[chi_pri[0]], P[chi_pri[1]])
    counts = counts.reindex(S.index).fillna(0).astype(int)
    has_mu = pd.Series(False, index=S.index)
    if len(mu_ilocs):
        has_mu.loc[mu_ilocs.index] = True

    for c in counts.columns:
        S[c] = counts[c]
    chain = _cut_chain(S, counts, has_mu)
    for c in chain.columns:
        S[c] = chain[c]

    mu_ilocs_alt, _, counts_alt = _run_selection(P, P[chi_alt[0]], P[chi_alt[1]])
    counts_alt = counts_alt.reindex(S.index).fillna(0).astype(int)
    has_mu_alt = pd.Series(False, index=S.index)
    if len(mu_ilocs_alt):
        has_mu_alt.loc[mu_ilocs_alt.index] = True
    chain_alt = _cut_chain(S, counts_alt, has_mu_alt)
    for c in chain_alt.columns:
        S[c + "_alt"] = chain_alt[c]
    S["n_proton_alt"] = counts_alt.n_proton

    # ------------------------------------------------------------------
    # sBruce variables (from the primary-flavor candidates)
    # ------------------------------------------------------------------
    mucols = ["len", "end_x", "end_y", "end_z", "dir_x", "dir_y", "dir_z",
              "p_muon", "trackScore", "chi2u_cafpyana", "chi2p_cafpyana",
              "chi2u_cafana", "chi2p_cafana"]
    if do_calo_syst:
        mucols += ["chi2u_%s" % v for v in SCALE_SMEAR_VARIATIONS + CALO_VARIATIONS]
        mucols += ["chi2p_%s" % v for v in SCALE_SMEAR_VARIATIONS + CALO_VARIATIONS]

    if len(mu_ilocs):
        mu = P.loc[pd.Index(mu_ilocs.values), mucols].copy()
        mu.index = mu_ilocs.index
    else:
        mu = pd.DataFrame(columns=mucols, dtype=float)
    mu = mu.reindex(S.index)

    # leading proton: longest pid==Proton pfp with len > 0 (find_longest_proton)
    prodf = P[(pid_s == PID_PROTON) & (P.len > 0)]
    pcols = ["len", "end_x", "end_y", "end_z", "dir_x", "dir_y", "dir_z",
             "p_proton", "ke_proton", "trackScore", "chi2u_cafpyana", "chi2p_cafpyana",
             "chi2u_cafana", "chi2p_cafana"]
    if do_calo_syst:
        pcols += ["chi2u_%s" % v for v in SCALE_SMEAR_VARIATIONS + CALO_VARIATIONS]
        pcols += ["chi2p_%s" % v for v in SCALE_SMEAR_VARIATIONS + CALO_VARIATIONS]
    if len(prodf):
        p_ilocs = prodf.len.groupby(level=[0, 1]).idxmax()
        pro = P.loc[pd.Index(p_ilocs.values), pcols].copy()
        pro.index = p_ilocs.index
    else:
        pro = pd.DataFrame(columns=pcols, dtype=float)
    pro = pro.reindex(S.index)

    # muon 4-momentum (MeV) and proton KE sum for recoE
    p_mu_x = mu.p_muon * mu.dir_x
    p_mu_y = mu.p_muon * mu.dir_y
    p_mu_z = mu.p_muon * mu.dir_z
    p_mu_mag = np.sqrt(p_mu_x**2 + p_mu_y**2 + p_mu_z**2)
    E_mu = np.sqrt((p_mu_mag * 1000.0)**2 + MUON_MASS**2)

    proton_ke_sum = (P.ke_proton[pid_s == PID_PROTON] + PROTON_BINDING_ENERGY).groupby(level=[0, 1]).sum()
    proton_ke_sum = _reindex(proton_ke_sum, S.index, np.nan)
    found_proton = _reindex((pid_s == PID_PROTON).groupby(level=[0, 1]).any(), S.index, False).astype(bool)

    recoE = np.where(has_mu & found_proton, (E_mu + proton_ke_sum) / 1000.0, -999.0)

    # transverse / angular variables (leading proton)
    p_p_x = pro.p_proton * pro.dir_x
    p_p_y = pro.p_proton * pro.dir_y
    p_p_z = pro.p_proton * pro.dir_z
    norm_mu_T = np.sqrt(p_mu_x**2 + p_mu_y**2)
    norm_p_T = np.sqrt(p_p_x**2 + p_p_y**2)
    transverse_angle = (p_mu_x * p_p_x + p_mu_y * p_p_y) / (norm_mu_T * norm_p_T)
    p_p_mag = np.sqrt(p_p_x**2 + p_p_y**2 + p_p_z**2)
    t3d_angle = (p_mu_x * p_p_x + p_mu_y * p_p_y + p_mu_z * p_p_z) / (p_mu_mag * p_p_mag)
    deltaPt = np.sqrt((p_mu_x + p_p_x)**2 + (p_mu_y + p_p_y)**2)

    # Barycenter_delta (bar_flash logic + sentinels)
    bar_z = np.select([S.vtx_x > 0, S.vtx_x < 0], [S.bar_z_west, S.bar_z_east], default=np.nan)
    bar_z = np.where(np.isnan(bar_z), -10000.0, bar_z)
    bar_x = np.select([S.vtx_x > 0, S.vtx_x < 0], [S.bar_x_west, S.bar_x_east], default=np.nan)
    bar_x = np.where(np.isnan(bar_x), 0.0, bar_x)
    delta = np.abs(bar_z - S.charge_center_z)
    S["Barycenter_delta"] = np.where((bar_z < -9999) & (np.abs(bar_x) < 1), -10.0, delta)
    S["bar_flash_z"] = bar_z
    S["bar_flash_x"] = bar_x

    # CryoSel
    light_east = (S.cryo_light == 2) | (S.cryo_light == 0)
    light_west = (S.cryo_light == 2) | (S.cryo_light == 1)
    S["CryoSel"] = np.select(
        [S.cryo_light < 0, (S.vtx_x > 0) & light_west, (S.vtx_x < 0) & light_east],
        [-1, 1, 0], default=-1)

    # ------------------------------------------------------------------
    # Reco_class (classification_type_debug port, via mcnu-level classification)
    # ------------------------------------------------------------------
    clsdf = maple_truth_classdf(f, det=DETECTOR)
    cls_lookup = clsdf.maple_class if len(clsdf) else pd.Series(dtype=float)
    key = pd.MultiIndex.from_arrays([S.index.get_level_values(0), S.tmatch_idx.fillna(-1).astype(int)])
    nu_class = pd.Series(cls_lookup.reindex(key).values, index=S.index)

    S["Reco_class"] = np.select(
        [S.tmatch_idx < 0,
         nu_class == CLS_1MU1P,
         nu_class == CLS_1MUNP,
         ~maple_isInFV(S.true_vtx_x, S.true_vtx_y, S.true_vtx_z, det=DETECTOR),
         np.abs(S.true_pdg) == 12,
         S.true_iscc == 0,
         (S.true_iscc == 1) & (S.true_genie_mode == 0),
         (S.true_iscc == 1) & (S.true_genie_mode == 1),
         (S.true_iscc == 1) & (S.true_genie_mode == 2),
         (S.true_iscc == 1) & ((S.true_genie_mode == 3) | (S.true_genie_mode == 4)),
         (S.true_iscc == 1) & (S.true_genie_mode == 10)],
        [3, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], default=12)

    # ------------------------------------------------------------------
    # assemble sBruce columns
    # ------------------------------------------------------------------
    S["recoE"] = recoE
    S["Muon_length"] = mu.len
    S["Muon_endx"] = mu.end_x
    S["Muon_endy"] = mu.end_y
    S["Muon_endz"] = mu.end_z
    S["Muon_trackScore"] = mu.trackScore
    S["Muon_chi2mu"] = mu[chi_pri[0]]
    S["Muon_chi2pro"] = mu[chi_pri[1]]
    S["Muon_chi2mu_cafpyana"] = mu.chi2u_cafpyana
    S["Muon_chi2pro_cafpyana"] = mu.chi2p_cafpyana
    S["Muon_chi2mu_cafana"] = mu.chi2u_cafana
    S["Muon_chi2pro_cafana"] = mu.chi2p_cafana
    S["Proton_length_leading"] = pro.len
    S["Proton_kinetic_leading"] = pro.ke_proton / 1000.0
    S["Proton_endx"] = pro.end_x
    S["Proton_endy"] = pro.end_y
    S["Proton_endz"] = pro.end_z
    S["Proton_trackScore"] = pro.trackScore
    S["Proton_chi2mu"] = pro[chi_pri[0]]
    S["Proton_chi2pro"] = pro[chi_pri[1]]
    S["Proton_chi2mu_cafpyana"] = pro.chi2u_cafpyana
    S["Proton_chi2pro_cafpyana"] = pro.chi2p_cafpyana
    S["Proton_chi2mu_cafana"] = pro.chi2u_cafana
    S["Proton_chi2pro_cafana"] = pro.chi2p_cafana
    S["Transverse_angle"] = transverse_angle
    S["T3D_angle_mup"] = t3d_angle
    S["deltaPt"] = deltaPt
    S["Transverse_mom_reco_mu"] = norm_mu_T
    S["Transverse_mom_reco_pro"] = norm_p_T
    S["Number_protons"] = S.n_proton
    S["FlashPE"] = S.flash_maxpe

    if do_calo_syst:
        for v in SCALE_SMEAR_VARIATIONS + CALO_VARIATIONS:
            S["Muon_chi2mu_%s" % v] = mu["chi2u_%s" % v]
            S["Muon_chi2pro_%s" % v] = mu["chi2p_%s" % v]
            S["Proton_chi2mu_%s" % v] = pro["chi2u_%s" % v]
            S["Proton_chi2pro_%s" % v] = pro["chi2p_%s" % v]

    S["detector"] = DETECTOR
    S["Run"] = RUN
    S["ismc"] = ismc
    S["pid_mode"] = pid_mode

    # ------------------------------------------------------------------
    # selection option
    # ------------------------------------------------------------------
    if selection == "none":
        pass
    elif selection == "presel":
        S = S[S.maple_presel]
    elif selection == "full":
        S = S[S.maple_sel]
    else:
        raise ValueError("selection must be 'none', 'presel', or 'full'")

    return S


# thin wrappers for configs -----------------------------------------------
def make_maple_evt_nosel_df(f):
    return make_maple_evt_df(f, selection="none", pid_mode="cafpyana", do_calo_syst=True)

def make_maple_evt_presel_df(f):
    return make_maple_evt_df(f, selection="presel", pid_mode="cafpyana", do_calo_syst=True)

def make_maple_evt_fullsel_df(f):
    return make_maple_evt_df(f, selection="full", pid_mode="cafpyana", do_calo_syst=True)

def make_maple_evt_nosel_cafanapid_df(f):
    return make_maple_evt_df(f, selection="none", pid_mode="cafana", do_calo_syst=False)

def make_maple_evt_fullsel_data_df(f):
    return make_maple_evt_df(f, selection="full", pid_mode="cafpyana", do_calo_syst=False)


# =====================================================================
# mcnu builder
# =====================================================================
def make_maple_nudf(f):
    mc = _flatcols(loadbranches(f["recTree"], mcbranches).rec.mc.nu)
    if mc.empty:
        return pd.DataFrame()

    det = loadbranches(f["recTree"], ["rec.hdr.det"]).rec.hdr.det
    DETECTOR = "SBND" if 1 == det.unique() else "ICARUS"
    run = loadbranches(f["recTree"], ["rec.hdr.run"]).rec.hdr.run
    RUN = 1 if DETECTOR == "SBND" else (2 if run.iloc[0] < 12960 else 4)

    cls = maple_truth_classdf(f, det=DETECTOR)

    nudf = pd.DataFrame({
        "nu_E": mc.E,
        "pdg": mc.pdg,
        "iscc": mc.iscc,
        "genie_mode": mc.genie_mode,
        "pos_x": mc.position_x,
        "pos_y": mc.position_y,
        "pos_z": mc.position_z,
        "baseline": mc.baseline,
        "time": mc.time,
        # link to the GENIE event record (evtrec table entry index)
        "genie_evtrec_idx": mc.genie_evtrec_idx,
        "maple_class": cls.maple_class,
        "is_1mu1p_maple": cls.maple_class == CLS_1MU1P,
        "is_1muNp_maple": cls.maple_class == CLS_1MUNP,
        "n_mu": cls.n_mu,
        "mu_length": cls.mu_length,
        "n_p_above40": cls.n_p_above40,
        "veto_particles": cls.veto,
        "uncontained_truth": cls.uncontained,
        "true_visible_Enu": cls.true_visible_Enu,
    })
    nudf["is_fv"] = maple_isInFV(nudf.pos_x, nudf.pos_y, nudf.pos_z, det=DETECTOR)
    nudf["is_av"] = maple_isInActive(nudf.pos_x, nudf.pos_y, nudf.pos_z, det=DETECTOR)
    nudf["ind"] = nudf.index.get_level_values(1)
    nudf["detector"] = DETECTOR
    nudf["Run"] = RUN

    return nudf


# =====================================================================
# wgt builder
# =====================================================================
def make_maple_wgtdf(f):
    """Systematic-weight dataframe (mcnu + multisim/multisigma weights).

    Requests the standard GENIE reweight set, restricted to the psets
    actually present in the input file (samples like ReCAF2026 do not carry
    every pset in the default list).
    """
    from makedf import geniesyst
    if "globalTree" in f:
        avail = list(f["globalTree"]["global/wgts/wgts.name"].arrays(library="np")["wgts.name"][0])
    else:
        avail = []
    systs = [s for s in geniesyst.regen_systematics if s in avail]
    missing = len(geniesyst.regen_systematics) - len(systs)
    if missing:
        print("make_maple_wgtdf: %d requested GENIE systematics absent in file, using %d" % (missing, len(systs)))
    return make_mcnudf(f, include_weights=True, multisim_nuniv=100, genie_systematics=systs)


def make_maple_rewgtdf(f):
    """Systematic-weight dataframe with the GUMP CV reweight knob set.

    Same weight request as gump make_gump_nurewgtdf, so the wgt table is
    column-compatible with the GUMP sbn-rewgted CV productions.
    """
    from analysis_village.gump.makedf import gump_genie_reknob_systematics
    return make_mcnudf(f, include_weights=True, slim=False,
                       genie_systematics=gump_genie_reknob_systematics)


# =====================================================================
# GENIE event record (evtrec) builder
# =====================================================================
def _read_genie_evtrec_subprocess(path, timeout=900):
    """Run genie_evtrec.read_genie_evtrec in a fresh python interpreter.

    pyROOT deadlocks when first used inside a forked multiprocessing Pool
    worker (as spawned by NTupleGlob.dataframes): the worker either hangs at
    recycling (maxtasksperchild=1) or dies without delivering its result,
    stalling the pool forever.  A fresh exec'd interpreter has none of the
    inherited fork state and exits cleanly, so the raw-object read is done
    there and the numpy arrays are shipped back via pickle.

    Raises on subprocess failure or if the read exceeds `timeout` seconds
    (a stuck read must fail loudly rather than hang the production).
    """
    import os
    import pickle
    import subprocess
    import sys
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pkl") as tf:
        code = (
            "import pickle\n"
            "from makedf import genie_evtrec\n"
            "d = genie_evtrec.read_genie_evtrec(%r)\n"
            "with open(%r, 'wb') as f:\n"
            "    pickle.dump(d, f)\n" % (str(path), tf.name)
        )
        # repo root on sys.path so `makedf` is importable regardless of cwd
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = dict(os.environ)
        env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
        res = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            timeout=timeout, env=env)
        if res.returncode != 0:
            raise IOError(
                "GENIE event-record subprocess failed (exit %d) for %s:\n%s"
                % (res.returncode, path, res.stderr.decode(errors="replace")[-2000:]))
        with open(tf.name, "rb") as f:
            return pickle.load(f)


def make_maple_evtrec_df(f):
    """GENIE event record (evtrec) table; same schema as make_genie_evtrec_df.

    The flat-StdHep path is pure uproot and pool-safe, so it goes through the
    core builder unchanged.  The raw genie::NtpMCEventRecord path (used e.g.
    by the ICARUS ReCAF2026 files) needs pyROOT + the GENIE libraries, which
    deadlock inside forked Pool workers -- that read is isolated in a fresh
    interpreter via _read_genie_evtrec_subprocess.
    """
    if "GenieEvtRecTree" not in f:
        return pd.DataFrame([])
    if "GenieEvtRec.StdHepPdg" in f["GenieEvtRecTree"].keys():
        return make_genie_evtrec_df(f)
    path = getattr(f, "file_path", None)
    if path is None:
        path = f._file.file_path
    d = _read_genie_evtrec_subprocess(path)
    return _build_genie_evtrec_df(d)
