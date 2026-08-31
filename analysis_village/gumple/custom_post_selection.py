
import gumple_cuts as gmpl

def maple_sel_only(df):
    return gmpl.maple_cut_chain(df)["m_maple_sel"]

def pre_sel_only(df):
    return gmpl.maple_cut_chain(df)["cut_presel"]

def pre_muon_only(df):
    return gmpl.maple_cut_chain(df)["m_maple_muon"]


def gump_sel_only(df):
    return gmpl.maple_cut_chain(df)["m_gump_sel"]