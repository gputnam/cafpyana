#!/bin/bash
### One invocation: every sample's POT is read once and reused for all 18 configs.
### -t, -d and -o are positionally paired, so the three lists must stay in step.

### Date label used in the ANL path, output names and plot tags
LABEL=${LABEL:-Oct21}

HDF=/exp/sbnd/data/users/gputnam/GUMPLE/sbn-rewgted-25/
EXP=/exp/sbnd/data/users/nrowe/sbn-rewgted-25-new
ANL=/flare/neutrinoGPU/SBN_PROfit/sBruce/$LABEL/sbn-rewgted-25-new
OUT=/exp/sbnd/data/users/nrowe/sbn-rewgted-25-new

### One render_config.py call, so each sample's POT is read once. -t, -d and -o are
### paired by position: keep the three lists in the same order, group by group.
python3 render_config.py --hdf-dir $HDF \
    -t `# Combined analysis, SBND + ICARUS Runs 2 and 4` \
       GumpleTemplate.xml.j2 \
       GumpTemplate.xml.j2 \
       MapleNPTemplate.xml.j2 \
       GumpleAr25_data.xml.j2 \
       GumpleNoCVWgtTemplate.xml.j2 \
       `# SBND + ICARUS Run 2 only` \
       GumpR2Only_mc.xml.j2 \
       GumpleR2Only_mc.xml.j2 \
       MapleNPR2Only_mc.xml.j2 \
       `# SBND + ICARUS Run 4 only` \
       GumpR4Only_mc.xml.j2 \
       GumpleR4Only_mc.xml.j2 \
       MapleNPR4Only_mc.xml.j2 \
       `# ANL copy of Gumple Run 4 only` \
       GumpleR4Only_mc.xml.j2 \
       `# Data/MC, GUMP: SBND, ICARUS Run 2, Run 4 - MC then data` \
       DataMC/sbnd_bash_rwt_GUMP.xml.j2 \
       DataMC/sbnd_data_bash_GUMP.xml.j2 \
       DataMC/icarus2_bash_rwt_GUMP.xml.j2 \
       DataMC/icarus_data2_bash_GUMP.xml.j2 \
       DataMC/icarus4_bash_rwt_GUMP.xml.j2 \
       DataMC/icarus4_data_bash_GUMP.xml.j2 \
       `# Data/MC, MAPLEMP` \
       DataMC/sbnd_bash_rwt_MAPLEMP.xml.j2 \
       DataMC/sbnd_data_bash_MAPLEMP.xml.j2 \
       DataMC/icarus2_bash_rwt_MAPLEMP.xml.j2 \
       DataMC/icarus_data2_bash_MAPLEMP.xml.j2 \
       DataMC/icarus4_bash_rwt_MAPLEMP.xml.j2 \
       DataMC/icarus4_data_bash_MAPLEMP.xml.j2 \
       `# Data/MC, MAPLENP` \
       DataMC/sbnd_bash_rwt_MAPLENP.xml.j2 \
       DataMC/sbnd_data_bash_MAPLENP.xml.j2 \
       DataMC/icarus2_bash_rwt_MAPLENP.xml.j2 \
       DataMC/icarus_data2_bash_MAPLENP.xml.j2 \
       DataMC/icarus4_bash_rwt_MAPLENP.xml.j2 \
       DataMC/icarus4_data_bash_MAPLENP.xml.j2 \
    -d `# Combined analysis` \
       $EXP \
       $EXP \
       $EXP \
       $EXP \
       $EXP \
       `# Run 2 only` \
       $EXP \
       $EXP \
       $EXP \
       `# Run 4 only` \
       $EXP \
       $EXP \
       $EXP \
       `# ANL copy of Gumple Run 4 only` \
       $ANL \
       `# Data/MC, GUMP` \
       $EXP \
       $EXP \
       $EXP \
       $EXP \
       $EXP \
       $EXP \
       `# Data/MC, MAPLEMP` \
       $EXP \
       $EXP \
       $EXP \
       $EXP \
       $EXP \
       $EXP \
       `# Data/MC, MAPLENP` \
       $EXP \
       $EXP \
       $EXP \
       $EXP \
       $EXP \
       $EXP \
    -o `# Combined analysis` \
       $OUT/MAPLEMP/Gumple$LABEL.xml \
       $OUT/GUMP/Gump$LABEL.xml \
       $OUT/MAPLENP/MapleNP$LABEL.xml \
       $OUT/MAPLEMP/GumpleAr25$LABEL.xml \
       $OUT/MAPLEMP/GumpleNoCVWgt$LABEL.xml \
       `# Run 2 only` \
       $OUT/GUMP/GumpR2Only$LABEL.xml \
       $OUT/MAPLEMP/GumpleR2Only$LABEL.xml \
       $OUT/MAPLENP/MapleNPR2Only$LABEL.xml \
       `# Run 4 only` \
       $OUT/GUMP/GumpR4Only$LABEL.xml \
       $OUT/MAPLEMP/GumpleR4Only$LABEL.xml \
       $OUT/MAPLENP/MapleNPR4Only$LABEL.xml \
       `# ANL copy of Gumple Run 4 only` \
       $OUT/MAPLEMP/GumpleANLR4Only$LABEL.xml \
       `# Data/MC, GUMP` \
       $OUT/GUMP/sbnd_bash_rwt_GUMP.xml \
       $OUT/GUMP/sbnd_data_bash_GUMP.xml \
       $OUT/GUMP/icarus2_bash_rwt_GUMP.xml \
       $OUT/GUMP/icarus_data2_bash_GUMP.xml \
       $OUT/GUMP/icarus4_bash_rwt_GUMP.xml \
       $OUT/GUMP/icarus4_data_bash_GUMP.xml \
       `# Data/MC, MAPLEMP` \
       $OUT/MAPLEMP/sbnd_bash_rwt_MAPLEMP.xml \
       $OUT/MAPLEMP/sbnd_data_bash_MAPLEMP.xml \
       $OUT/MAPLEMP/icarus2_bash_rwt_MAPLEMP.xml \
       $OUT/MAPLEMP/icarus_data2_bash_MAPLEMP.xml \
       $OUT/MAPLEMP/icarus4_bash_rwt_MAPLEMP.xml \
       $OUT/MAPLEMP/icarus4_data_bash_MAPLEMP.xml \
       `# Data/MC, MAPLENP` \
       $OUT/MAPLENP/sbnd_bash_rwt_MAPLENP.xml \
       $OUT/MAPLENP/sbnd_data_bash_MAPLENP.xml \
       $OUT/MAPLENP/icarus2_bash_rwt_MAPLENP.xml \
       $OUT/MAPLENP/icarus_data2_bash_MAPLENP.xml \
       $OUT/MAPLENP/icarus4_bash_rwt_MAPLENP.xml \
       $OUT/MAPLENP/icarus4_data_bash_MAPLENP.xml

RENDER_RC=$?
if [ $RENDER_RC -ne 0 ]; then
    echo "render_config.py failed (exit $RENDER_RC), skipping tests" >&2
    exit $RENDER_RC
fi

### PROfit tests, run side by side. Each posts its own plots to Slack when it
### finishes (test scripts and slack_upload.sh sit next to this script).
DATAMC=$OUT/plots/MAPLE_new_DATA_MC_comp_DQ/$LABEL

./test_cv_surfaces.sh $LABEL $OUT
./slack_upload.sh "PROfit CV plots and surfaces for $LABEL" \
    $OUT/plots/Gumple${LABEL}_v1_PROplot_Variable_0_ErrorBand.pdf \
    $OUT/plots/Gump${LABEL}_v1_PROplot_Variable_0_ErrorBand.pdf \
    $OUT/plots/MapleNP${LABEL}_v1_PROplot_Variable_0_ErrorBand.pdf \
    $OUT/plots/surface_comparison_$LABEL.pdf

./test_datamc.sh $LABEL $OUT
./slack_upload.sh "PROfit Data/MC comparisons for $LABEL" \
    $DATAMC/Pandora_sbnd_*_nu_E_calo_*_ErrorBand.pdf \
    $DATAMC/Pandora_*_del_p_*_ErrorBand.pdf
