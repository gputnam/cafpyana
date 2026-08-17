# maple_cuts.py

SBND_CUTS = {
    "nu_score_th": 0.35,
    "max_opening_angle": 160,
    "musel_track_score_min": 0.5,
    "musel_muscore_th": 38,
    "musel_pscore_th": 82,
    "musel_len_th_min": 40,
    "musel_len_th_max": 400,
    "psel_muscore_th": 0,
    "psel_pscore_th": 141,
}

ICARUS_CUTS = {
    "nu_score_th": 0.35,
    "max_opening_angle": 160,
    "musel_track_score_min": 0.5,
    "musel_muscore_th": 111,
    "musel_pscore_th": 74,
    "musel_len_th_min": 40,
    "musel_len_th_max": 400,
    "psel_muscore_th": 0,
    "psel_pscore_th": 92,
}

# Registry map for easy lookup
CUTS_BY_DETECTOR = {
    "SBND": SBND_CUTS,
    "ICARUS": ICARUS_CUTS,
}

def get_base_muon_mask(df, cuts=CUTS_BY_DETECTOR["SBND"], level="slc"):
    if level == "trk":
        pref = ""
    elif level == "slc":
        pref = "mu_"

    for c in df.columns:
        print(c)

    base_mask = (
            df[pref+"start_x"].notna()
            & df[pref+"len"].notna()
            & (df[pref+"trackScore"] >= cuts["musel_track_score_min"])
            & (df[pref+"dist_start"] <= 10.0)
            & df[pref+"len"].between(cuts["musel_len_th_min"], cuts["musel_len_th_max"])
            & df[pref+"prim_pfp"]
            & df[pref+"contained10"]
            & (df[pref+"end_x"] * df["slc_vtx_x"] > 0)
        )
    return base_mask

def get_muon_mask(df, detector="SBND", variation=None, **override_cuts):
    """Generates a boolean mask filtering candidate muon tracks.

    Applies the variation suffix 'v' strictly to chi2 variables.
    """
    if detector not in CUTS_BY_DETECTOR:
        raise ValueError(
            f"Unknown detector '{detector}'. Valid options: {list(CUTS_BY_DETECTOR.keys())}"
        )

    # Format variation suffix (e.g., None -> "", "_sys" -> "_sys")
    v = "" if variation is None else variation
    cuts = {**CUTS_BY_DETECTOR[detector], **override_cuts}

    # Standard kinematic/topological selection
    base_mask = get_base_muon_mask(df, cuts=cuts, level="slc")

    # Apply muon/proton chi2 cuts using the variation suffix on chi2 columns
    chi2_muon_col = f"muon_chi2{v}_of_muon_cand"
    chi2_prot_col = f"prot_chi2{v}_of_muon_cand"

    chi2_mask = True
    if chi2_muon_col in df.columns:
        chi2_mask &= df[chi2_muon_col] < cuts["musel_muscore_th"]
    if chi2_prot_col in df.columns:
        chi2_mask &= df[chi2_prot_col] > cuts["musel_pscore_th"]

    return base_mask & chi2_mask

def get_proton_mask(df, detector="SBND", variation=None, **override_cuts):
    """Generates a boolean mask filtering candidate proton tracks on chi2 score."""
    if detector not in CUTS_BY_DETECTOR:
        raise ValueError(
            f"Unknown detector '{detector}'. Valid options: {list(CUTS_BY_DETECTOR.keys())}"
        )

    # Format variation suffix (e.g., None -> "", "_sys" -> "_sys")
    v = "" if variation is None else variation
    cuts = {**CUTS_BY_DETECTOR[detector], **override_cuts}

    # Dynamic column name targeting only prot_chi2
    chi2_prot_col = f"prot_chi2{v}_of_prot_cand"

    return df[chi2_prot_col] < cuts["psel_pscore_th"]
