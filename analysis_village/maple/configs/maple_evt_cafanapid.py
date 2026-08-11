# MAPLE evt production for VALIDATION against CAFANA-MAPLE: candidates and
# cut booleans driven by the CAFANA-compat chi2 (stored dedx + chi2_ALG).
from analysis_village.maple.makedf import *

DFS = [make_maple_evt_nosel_cafanapid_df, make_maple_nudf, make_hdrdf, make_triggerdf, make_potdf_bnb]
NAMES = ["evt", "mcnu", "hdr", "trig", "bnb"]
