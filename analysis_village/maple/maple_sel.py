"""Standalone MAPLE selection on evt dataframes.

Applies the MAPLE PID-dependent cuts post-hoc from the columns stored by
make_maple_evt_df, so the same dataframe can be selected under the nominal
chi2 and under any calorimetric chi2 variation (the GUMP
SignalBoxSystematics pattern: swap the chi2 columns, re-apply the cuts,
treat each variation as a SampleSystematic).

The evt df stores, per slice:
  - the chi2-free preselection booleans (cut_sanity, cut_fv, cut_crthit,
    cut_cryo, cut_contained, cut_cathode, maple_presel) and the chi2-free
    candidate booleans/counts (has_muon, n_proton, cut_np, cut_0shwother);
  - the muon-candidate chi2 (mu_chi2{v}_of_mu_cand, prot_chi2{v}_of_mu_cand);
  - worst-case aggregates over the candidate protons:
    mu_chi2{v}_of_prot_cand = min over protons (cut direction >),
    prot_chi2{v}_of_prot_cand = max over protons (cut direction <),
    min_proton_ke, min_proton_track_score;
where {v} is "" (nominal cafpyana), one of the calorimetric variations, or
"cafana".  Aggregates use skipna=False semantics: any NaN chi2 among the
candidates makes the aggregate NaN, so the cut fails (the old per-pfp
selection required chi2.notna()).

This module only needs numpy/pandas + the maple_cuts constants; it does not
pull in the df-making machinery.
"""
import numpy as np
import pandas as pd
from analysis_village.maple.maple_cuts import * 

# All chi2 variation suffixes stored in the evt df (when do_calo_syst=True),
# plus the CAFANA-compat chi2 stored under the same scheme.
SCALE_SMEAR_VARIATIONS = ["lo", "hi", "2lo", "2hi", "smear5", "smear13", "sqsmear15"]
CALO_VARIATIONS = ["cv", "alpha_p", "alpha_m", "beta_p", "beta_m", "R_p", "R_m", "R_p25", "dedxbias"]
CHI2_VARIATIONS = SCALE_SMEAR_VARIATIONS + CALO_VARIATIONS + ["cafana"]

# The four per-slice chi2 candidate columns, GUMP naming ("%s" is the
# variation suffix; "" is nominal).
CHI2_CAND_COLS = [
    "mu_chi2%s_of_mu_cand",
    "prot_chi2%s_of_mu_cand",
    "mu_chi2%s_of_prot_cand",
    "prot_chi2%s_of_prot_cand",
]


def maple_v_calovariation(df, variation):
    """Copy of df with the nominal chi2 candidate columns replaced by the
    {variation}-suffixed ones (GUMP nb v_calovariation semantics)."""
    df = df.copy()
    for c in CHI2_CAND_COLS:
        df[c % ""] = df[c % variation]
    return df


def maple_selection(df, variation=None):
    """MAPLE PID-dependent cut chain evaluated from evt-df columns.

    variation: None for the nominal chi2, or a suffix from CHI2_VARIATIONS.

    Returns a DataFrame (index = df.index) with columns:
      cut_muon      -- muon candidate exists and passes the chi2 cuts
      cut_np        -- proton-multiplicity cut (chi2-free, from the df)
      cut_protons   -- every candidate proton is a proton: max chi2p over
                       candidates < CHI2_PROTON_PION (the old pion veto)
      cut_0shwother -- no shower / other pfps (chi2-free, from the df)
      maple_sel     -- maple_presel & all of the above
      maxcut        -- cutflow stage (1-10): sanity, FV, CRT veto,
                       cryo-light, containment+cathode, muon, Np, protons
                       (old pion stage), shower/other; 10 = selected

    NaN comparisons are False, so a NaN chi2 or aggregate fails the cut
    (matching the old explicit notna() requirements).
    """
    cut_muon = get_muon_mask(df, detector=df.detector.iloc[0], variation=variation)
    cut_protons = get_proton_mask(df, detector=df.detector.iloc[0], variation=variation)

    maple_sel = df.maple_presel & cut_muon & df.cut_np & cut_protons & df.cut_0shwother

    maxcut = np.select(
        [~df.cut_sanity, ~df.cut_fv, ~df.cut_crthit, ~df.cut_cryo,
         ~(df.cut_contained & df.cut_cathode), ~cut_muon, ~df.cut_np,
         ~cut_protons, ~df.cut_0shwother],
        [1, 2, 3, 4, 5, 6, 7, 8, 9], default=10)

    return pd.DataFrame({
        "cut_muon": cut_muon,
        "cut_np": df.cut_np,
        "cut_protons": cut_protons,
        "cut_0shwother": df.cut_0shwother,
        "maple_sel": maple_sel,
        "maxcut": maxcut,
    }, index=df.index)

def all_maple_cuts(df):
    return maple_selection(df)['maple_sel']
