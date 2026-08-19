import os
import sys
import pandas as pd

# Add the head directly to sys.path
workspace_root = os.getcwd()
sys.path.insert(0, workspace_root + "/../../")

from makedf.util import *

PROTON_MASS = 0.938272
NEUTRON_MASS = 0.939565
MUON_MASS = 0.105658
PION_MASS = 0.139570
MASS_A = 22*NEUTRON_MASS + 18*PROTON_MASS - 0.34381
BE = 0.0295 # note: maple had a difference value for this... 0.0309
MASS_Ap = MASS_A - NEUTRON_MASS + BE

def neutrino_energy(mu_p, mu_dir, p_p, p_dir, p_E, BE=BE):
    # p_p, p_dir, p_E are the proton momentum magnitude, direction and energy.
    # For a single proton p_E = mag2d(p_p, PROTON_MASS); for a summed multi-proton
    # system p_p = |sum_i p_i| and p_E = sum_i E_i.
    mass_Ap = MASS_A - NEUTRON_MASS + BE

    mu_E = mag2d(mu_p, MUON_MASS)
    # invariant mass of the proton system (PROTON_MASS for a single on-shell
    # proton); p_E - p_M is the proton system's kinetic energy.
    p_M = np.sqrt(np.maximum(p_E**2 - p_p**2, 0.))

    dpT = transverse_kinematics(mu_p, mu_dir, p_p, p_dir, p_E, BE=BE)['del_Tp']
    ET = np.sqrt(dpT**2 + mass_Ap**2) - mass_Ap

    return mu_E + p_E - p_M + ET + BE

def neutrino_energy_ccqe(mu_p, mu_costh, BE=BE):
    """Reconstructed neutrino energy under the CCQE assumption, from the muon
    kinematics alone (nu_mu n -> mu- p on a nucleon at rest):

        E_nu^QE = [2(M_n - E_B)E_mu - ((M_n - E_B)^2 + m_mu^2 - M_p^2)]
                  / [2(M_n - E_B - E_mu + p_mu cos(theta_mu))]

    This is the MiniBooNE form (Phys. Rev. D 81, 092005); the numerator is
    often written as E_B^2 - 2 M_n E_B + m_mu^2 + (M_n^2 - M_p^2), which is
    algebraically the same thing.

    NB: E_B defaults to this module's BE (29.5 MeV) rather than MiniBooNE's
    34 MeV, so that E_nu^QE and neutrino_energy() share one binding energy and
    the binding-energy systematic moves both together.

    Unlike neutrino_energy(), this uses no proton information at all -- that is
    the point of it: it is the estimator that is biased by any non-QE (MEC,
    RES, FSI) contribution, so comparing it against the calorimetric energy
    probes that modelling.

    The denominator vanishes as the muon approaches the kinematic edge; rows
    where it is non-positive are unphysical under the QE assumption and are
    returned as NaN rather than as a large negative energy.
    """
    Mn = NEUTRON_MASS - BE
    mu_E = mag2d(mu_p, MUON_MASS)

    num = 2*Mn*mu_E - (Mn**2 + MUON_MASS**2 - PROTON_MASS**2)
    den = 2*(Mn - mu_E + mu_p*mu_costh)

    # Series / ndarray keeps the Series index, so this preserves whatever
    # container came in. NB: do NOT re-wrap in pd.Series(..., index=...) --
    # the CV frames carry duplicate index labels and that would try to align.
    return num/np.where(den > 0, den, np.nan)

def transverse_kinematics(mu_p, mu_dir, p_p, p_dir, p_E=None, BE=BE):
    # p_p, p_dir, p_E are the proton momentum magnitude, direction and energy.
    # For a single proton p_E = mag2d(p_p, PROTON_MASS); for a summed multi-proton
    # system p_p = |sum_i p_i| and p_E = sum_i E_i.
    mass_Ap = MASS_A - NEUTRON_MASS + BE
    mu_E = mag2d(mu_p, MUON_MASS)

    # if no sep proton energy input, assume 1mu1p
    if p_E is None:
        p_E = mag2d(p_p, PROTON_MASS)

    mu_p_x = mu_p * mu_dir.x
    mu_p_y = mu_p * mu_dir.y
    mu_p_z = mu_p * mu_dir.z
    mu_phi_x = mu_p_x/mag2d(mu_p_x, mu_p_y)
    mu_phi_y = mu_p_y/mag2d(mu_p_x, mu_p_y)

    p_p_x = p_p * p_dir.x
    p_p_y = p_p * p_dir.y
    p_p_z = p_p * p_dir.z
    p_phi_x = p_p_x/mag2d(p_p_x, p_p_y)
    p_phi_y = p_p_y/mag2d(p_p_x, p_p_y)

    mu_Tp_x = mu_phi_y*mu_p_x - mu_phi_x*mu_p_y
    mu_Tp_y = mu_phi_x*mu_p_x + mu_phi_y*mu_p_y
    mu_Tp = mag2d(mu_Tp_x, mu_Tp_y)


    p_Tp_x = mu_phi_y*p_p_x - mu_phi_x*p_p_y
    p_Tp_y = mu_phi_x*p_p_x + mu_phi_y*p_p_y
    p_Tp = mag2d(p_Tp_x, p_Tp_y)



    del_Tp_x = mu_Tp_x + p_Tp_x
    del_Tp_y = mu_Tp_y + p_Tp_y
    del_Tp = mag2d(del_Tp_x, del_Tp_y)


    del_alpha = np.arccos(-(mu_Tp_x*del_Tp_x + mu_Tp_y*del_Tp_y)/(mu_Tp*del_Tp))
    del_phi = np.arccos(-(mu_Tp_x*p_Tp_x + mu_Tp_y*p_Tp_y)/(mu_Tp*p_Tp))

    R = MASS_A + mu_p_z + p_p_z - mu_E - p_E
    del_Lp = 0.5*R - mag2d(mass_Ap, del_Tp)**2/(2*R)

    del_p = mag2d(del_Tp, del_Lp)


    return pd.Series({'del_p' : del_p, 
                      'del_Tp' : del_Tp, 
                      'del_phi' : del_phi, 
                      'del_alpha' : del_alpha, 
                      'mu_E' : mu_E, 
                      'p_E' : p_E})

def kinetic_energy(mass_GeV, momentum_GeV):
    """KE in GeV given momentum in GeV (matches the CAFANA helper)."""
    return np.sqrt(mass_GeV * mass_GeV + momentum_GeV * momentum_GeV) - mass_GeV
