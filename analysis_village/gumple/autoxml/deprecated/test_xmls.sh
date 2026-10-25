#!/bin/bash
### PROfit tests of the XMLs written by rend_it_all.sh: CV plots, sensitivity
### surfaces and their comparison, and Data/MC plots for each reco variable.
### Runs up to JOBS PROfit jobs at once.
### Usage:  test_xmls.sh LABEL OUT        (or JOBS=16 test_xmls.sh LABEL OUT)
LABEL=$1
OUT=$2
JOBS=${JOBS:-8}
HERE=$(dirname "$(readlink -f "$0")")
PROFIT=/exp/sbnd/app/users/nrowe/xml_tests/Elephant_Vanishes/build/bin/PROfit

### Remember the venv's Python before spack puts its own Python first
PY=$(command -v python3); PY_PATH=$PYTHONPATH

### PROfit dependencies (AL9 setup from the PROfit README)
source /cvmfs/larsoft.opensciencegrid.org/spack-packages/setup-env.sh
spack load gcc@12.2.0
spack load cmake@3.27.7
spack load boost/iylabz2
spack load hdf5@1.14.3
spack load root@6.28.12
spack load ifdhc@2.7.2

### Wait until fewer than JOBS background jobs are running
throttle() { while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done; }

### PROfit writes its cache and plots into the current directory
mkdir -p $OUT/plots && cd $OUT/plots

### ---------------------------------------------------------------------------
### CV plots and sensitivity surfaces (sin^2 2theta_mumu vs dm^2) for the Gump,
### Gumple and MapleNP XMLs. The three selections run in the background; within
### each, surface reuses the cache that plot builds, so those two run in order.
### ---------------------------------------------------------------------------
cv_and_surface() {
    $PROFIT -x $1 -t $2 --log $2.log plot
    $PROFIT -x $1 -t $2 -o surf -n 8 --log $2_surf.log surface
}
cv_and_surface $OUT/GUMP/Gump$LABEL.xml       Gump$LABEL    > Gump${LABEL}_stdout.log    2>&1 &
cv_and_surface $OUT/MAPLEMP/Gumple$LABEL.xml  Gumple$LABEL  > Gumple${LABEL}_stdout.log  2>&1 &
cv_and_surface $OUT/MAPLENP/MapleNP$LABEL.xml MapleNP$LABEL > MapleNP${LABEL}_stdout.log 2>&1 &

### ---------------------------------------------------------------------------
### Data/MC plots: the DataMC XMLs have BRANCH and BINS placeholders, filled in
### here for each variable. The branch name is the variable name.
### ---------------------------------------------------------------------------
declare -A COMMON_BINS=(
    [del_Tp]='<bins unit ="Transverse momenta [GeV/c]" edges="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0"/>'
    [slc_vtx_y]='<bins unit ="Vertex y [cm]" min="-200" max="200" nbins="20"/>'
    [del_p]='<bins unit ="Missing momenta [GeV/c]" min="0" max="1.5" nbins="15"/>'
    [mu_end_y]='<bins unit ="Muon end y [cm]" min="-200" max="200" nbins="20"/>'
    [p_end_y]='<bins unit ="Proton candidate end y [cm]" min="-200" max="200" nbins="20"/>'
    [p_ke]='<bins unit ="Proton candidate KE [GeV]" min="0" max="1" nbins="20"/>'
    [n_pfp]='<bins unit ="Number of pfp" min="1.5" max="5.5" nbins="4"/>'
    [mu_chi2_of_mu_cand]='<bins unit ="muon #chi^{2}_{#mu}" edges="0 5 10 15 20 25 30 40"/>'
    [prot_chi2_of_prot_cand]='<bins unit ="proton #chi^{2}_{p}" min="0" max="90" nbins="18"/>'
    [prot_chi2_of_mu_cand]='<bins unit ="muon #chi^{2}_{p}" min="80" max="320" nbins="24"/>'
    [mu_chi2_of_prot_cand]='<bins unit ="proton #chi^{2}_{#mu}" min="0" max="90" nbins="18"/>'
)
### Detector geometry differs in x and z; SBND also plots muon length and reco
### energy (nu_E_calo, whose binning depends on the selection: see NU_E_EDGES)
declare -A ICARUS_BINS=(
    [slc_vtx_x]='<bins unit ="Vertex x [cm]" min="-400" max="400" nbins="20"/>'
    [slc_vtx_z]='<bins unit ="Vertex Z [cm]" min="-900" max="900" nbins="25"/>'
    [mu_end_x]='<bins unit ="Muon end x [cm]" min="-400" max="400" nbins="20"/>'
    [mu_end_z]='<bins unit ="Muon end z [cm]" min="-900" max="900" nbins="25"/>'
    [p_end_x]='<bins unit ="Proton candidate end x [cm]" min="-400" max="400" nbins="20"/>'
    [p_end_z]='<bins unit ="Proton candidate end z [cm]" min="-900" max="900" nbins="25"/>'
)
declare -A SBND_BINS=(
    [slc_vtx_x]='<bins unit ="Vertex x [cm]" min="-200" max="200" nbins="20"/>'
    [slc_vtx_z]='<bins unit ="Vertex Z [cm]" min="0" max="500" nbins="25"/>'
    [mu_end_x]='<bins unit ="Muon end x [cm]" min="-200" max="200" nbins="20"/>'
    [mu_end_z]='<bins unit ="Muon end z [cm]" min="0" max="500" nbins="25"/>'
    [p_end_x]='<bins unit ="Proton candidate end x [cm]" min="-200" max="200" nbins="20"/>'
    [p_end_z]='<bins unit ="Proton candidate end z [cm]" min="0" max="500" nbins="25"/>'
    [mu_len]='<bins unit ="Muon length [cm]" min="40" max="400" nbins="18"/>'
)
### SBND nu_E_calo bin edges per selection, matching each selection's analysis template
declare -A NU_E_EDGES=(
    [MAPLEMP]="0.3 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.05 1.10 1.25 1.5"
    [GUMP]="0.3 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.5"
    [MAPLENP]="0.3 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.5"
)
for v in "${!COMMON_BINS[@]}"; do
    ICARUS_BINS[$v]=${COMMON_BINS[$v]}
    SBND_BINS[$v]=${COMMON_BINS[$v]}
done

### datamc DET SEL MC_XML DATA_XML BINS_ARRAY SUFFIX [extra PROfit options]
### Starts one background PROfit job per variable, each with its own temporary
### XMLs, cache and log, so parallel jobs don't interfere.
datamc() {
    local DET=$1 SEL=$2 MC=$3 DATA=$4 SUFFIX=$6
    local -n BINS=$5
    [ -f "$MC" ] && [ -f "$DATA" ] || { echo "Missing $MC or $DATA, skipping" >&2; return; }
    for v in "${!BINS[@]}"; do
        local T="Pandora_${DET}_${SEL}_${v}_$SUFFIX"
        throttle
        (
            sed -e "s|BRANCH|$v|g" -e "s|BINS|${BINS[$v]}|g" "$MC"   > tmp_$T.xml
            sed -e "s|BRANCH|$v|g" -e "s|BINS|${BINS[$v]}|g" "$DATA" > tmp_${T}_data.xml
            $PROFIT -x tmp_$T.xml --force -t "$T" -o datamc --data tmp_${T}_data.xml "${@:7}" plot > $T.log 2>&1
            rm -f tmp_$T.xml tmp_${T}_data.xml "${T}"_*.bin
        ) &
    done
}

### Folder and file names match the Overleaf figures, e.g.
### Figures/MAPLE_new_DATA_MC_comp_DQ/new20Oct/Pandora_sbnd_MAPLEMP_del_Tp_abs_datamc_PROplot_Variable_0_ErrorBand.pdf
OVERLEAF_DIR=MAPLE_new_DATA_MC_comp_DQ/new20Oct
mkdir -p $OUT/plots/$OVERLEAF_DIR && cd $OUT/plots/$OVERLEAF_DIR
for SEL in GUMP MAPLEMP MAPLENP; do
    D=$OUT/$SEL
    SBND_BINS[nu_E_calo]="<bins unit =\"Reconstructed neutrino energy [GeV]\" edges=\"${NU_E_EDGES[$SEL]}\"/>"
    datamc sbnd    $SEL $D/sbnd_bash_rwt_$SEL.xml    $D/sbnd_data_bash_$SEL.xml    SBND_BINS   abs
    datamc sbnd    $SEL $D/sbnd_bash_rwt_$SEL.xml    $D/sbnd_data_bash_$SEL.xml    SBND_BINS   shapeonly --shapeonly
    datamc icarus2 $SEL $D/icarus2_bash_rwt_$SEL.xml $D/icarus_data2_bash_$SEL.xml ICARUS_BINS shapeonly --shapeonly
    datamc icarus4 $SEL $D/icarus4_bash_rwt_$SEL.xml $D/icarus4_data_bash_$SEL.xml ICARUS_BINS shapeonly --shapeonly
done

### Wait for everything, then compare the three surfaces
wait
cd $OUT/plots
PYTHONPATH=$PY_PATH $PY $HERE/../plotter.py \
    -f Gumple${LABEL}_surf_surface.txt Gump${LABEL}_surf_surface.txt MapleNP${LABEL}_surf_surface.txt \
    -l Gumple Gump MapleNP -o surface_comparison_$LABEL.pdf
