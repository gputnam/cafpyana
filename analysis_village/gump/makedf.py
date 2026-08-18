from analysis_village.maple.makedf import *

def make_gump_evt_df(f, selection="none", do_calo_syst=True):
    S = make_maple_evt_df(f, selection=selection, do_calo_syst=do_calo_syst)
    ## no selection is sam btwn maple and gump
    if selection == "none":
        pass
    ## presel is same btwn maple and gump
    elif selection == "presel":
        pass
    ## full gump selection is just maple, but with added cut 
    elif selection == "full":
        S = S[(S['n_proton'] == 1)]
    return S

def make_gump_evt_df(f):
    return make_maple_nudf(f)

def make_gump_evtrec_df(f):
    return make_maple_nudf(f)

def make_gump_wgtdf(f):
    return make_maple_wgtdf(f)

def make_gump_rewgtdf(f):
    return make_maple_rewgtdf(f)

def make_gump_evt_nosel_df(f):
    return make_maple_evt_df(f, selection="none", do_calo_syst=True)

def make_gump_evt_presel_df(f):
    return make_maple_evt_df(f, selection="presel", do_calo_syst=True)

def make_gump_evt_fullsel_df(f):
    S = make_maple_evt_df(f, selection="full", do_calo_syst=True)
    S = S[(S['n_proton'] == 1)]
    return S

def make_gump_evt_fullsel_data_df(f):
    S = make_maple_evt_df(f, selection="full", do_calo_syst=False)
    S = S[(S['n_proton'] == 1)]
    return S

def make_gump_nudf(f):
    return make_maple_nudf(f)
