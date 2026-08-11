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
PION_KE_MIN = 0.0          # MeV
PROTON_KE_MIN = 40.0       # MeV
PION_MASS = 139.570        # MeV
PROTON_MASS = 938.3        # MeV
MUON_MASS = 105.658        # MeV
PROTON_BINDING_ENERGY = 30.9  # MeV (argon effective)
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


def maple_isInFV(x, y, z):
    """isInFV: 10 cm insets (50 cm downstream z), dangling-cable exclusion."""
    ok = (((x < _XE_HI - 10) & (x > _XE_LO + 10)) |
          ((x > _XW_LO + 10) & (x < _XW_HI - 10))) & \
         ((y > _Y_LO + 10) & (y < _Y_HI - 10)) & \
         ((z > _Z_LO + 10) & (z < _Z_HI - 50))
    cable = (x > 210.0) & (y > 60.0) & (z > 290.0) & (z < 390.0)
    return ok & ~cable


def maple_isInContained(x, y, z, dist=CONTAINMENT_CUT):
    """isInContained: `dist` insets plus removal of Y-Z corner triangles."""
    tri = (y < (1.732007 * z - 1687.5114)) | \
          (y > (-1.732007 * z + 1640.6114)) | \
          (y > (1.732007 * z + 1640.6114)) | \
          (y < (-1.732007 * z - 1687.5114))
    ok = (((x < _XE_HI - dist) & (x > _XE_LO + dist)) |
          ((x > _XW_LO + dist) & (x < _XW_HI - dist))) & \
         ((y > _Y_LO + dist) & (y < _Y_HI - dist)) & \
         ((z > _Z_LO + dist) & (z < _Z_HI - dist))
    return ok & ~tri


def maple_isInActive(x, y, z):
    return (((x < _XE_HI) & (x > _XE_LO)) |
            ((x > _XW_LO) & (x < _XW_HI))) & \
           ((y > _Y_LO) & (y < _Y_HI)) & \
           ((z > _Z_LO) & (z < _Z_HI))


def kinetic_energy(mass_MeV, momentum_GeV):
    """KE in MeV given momentum in GeV (matches the CAFANA helper)."""
    p = momentum_GeV * 1000.0
    return np.sqrt(mass_MeV * mass_MeV + p * p) - mass_MeV
