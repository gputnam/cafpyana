"""Exact port of the MAPLE (CAFANA) geometry and cut constants.

Source of truth:
  NicolaICARUS/MAPLE_GUMP/icarus/
    helper_eff_cf_FINAL_Lmu_EKpro_CHI2var_TRKSCOREvar_mup_only_CRTveto_Lmu_FV_Trigger.h

All comparisons here mirror the C++ semantics, including how NaN behaves
(in C++ any comparison with NaN is false; the same is true for pandas
comparison operators, so masks written in "C++ direction" carry over).
"""
import numpy as np
import pandas as pd
from analysis_village.gump.kinematics import *
import analysis_village.gump.gump_cuts as gc

# ----------------- Tunable cuts (GUMP-tuned values) -----------------
DEFAULT_PLANE = 2
PRIMARY_TRACK_SCORE = 0.5
SECONDARY_TRACK_LOW = 0.4
VTX_MAX_DIST = 50.0        # cm
MIN_MUON_LENGTH = 40.0     # cm
MAX_MUON_LENGTH = 400.0    # cm
MAX_CHI2_MUON = 111.0
MIN_CHI2_PROTON = 74.0
CHI2_PROTON_PION = 92.0
CONTAINMENT_CUT = 10.0     # cm
CALO_RR_MIN = 0.0
CALO_RR_MAX = 25.0
PION_KE_MIN = 0.0          # GeV
PROTON_KE_MIN = 0.04       # GeV
PION_MASS = 0.139570        # GeV
PROTON_MASS = 0.9383        # GeV
MUON_MASS = 0.105658        # GeV
PROTON_BINDING_ENERGY = 0.0309  # GeV (argon effective)
NP_MODE = True             # True -> 1muNp (protons > 1)

# cryo_selection_from_light
CRYO_LIGHT_TMIN = -0.6
CRYO_LIGHT_TMAX = 1.8
CRYO_LIGHT_PE_THRESHOLD = 3000. / 0.341  # valid for Run 4

# kCRTNeutrino (CRT top veto)
CRT_VETO_TMIN = -1.0
CRT_VETO_TMAX = 1.8
CRT_VETO_PLANE_MIN = 29  # exclusive
CRT_VETO_PLANE_MAX = 50  # exclusive

# bar_flash time windows
BAR_FLASH_TMIN_MC, BAR_FLASH_TMAX_MC = 0.0, 1.6
BAR_FLASH_TMIN_DATA, BAR_FLASH_TMAX_DATA = -0.4, 1.5

# ICARUS active-volume boundaries used by the MAPLE helpers
_XE_LO, _XE_HI = -358.49, -61.94   # east cryostat
_XW_LO, _XW_HI = 61.94, 358.49     # west cryostat
_Y_LO, _Y_HI = -181.86, 134.96
_Z_LO, _Z_HI = -894.95, 894.95

# SBND active-volume boundaries
_S_X_LO, _S_X_HI = -200.0, 200.0
_S_Y_LO, _S_Y_HI = -200.0, 200.0
_S_Z_LO, _S_Z_HI = 0.0, 500.0

# SBND cathode prism for the cathode-crossing veto (GUMP gump_cuts.cathode_cut)
SBND_CATHODE_PRISM_MIN = (-5.0, -200.0, 0.0)
SBND_CATHODE_PRISM_MAX = (5.0, 200.0, 500.0)
