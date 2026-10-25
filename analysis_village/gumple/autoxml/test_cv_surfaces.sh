#!/bin/bash
### CV plots and sensitivity surfaces (sin^2 2theta_mumu vs dm^2) for the Gump,
### Gumple and MapleNP XMLs, then a comparison of the three surfaces.
### Usage:  test_cv_surfaces.sh LABEL OUT       (or NTHREAD=4 test_cv_surfaces.sh ...)
LABEL=$1
OUT=$2
NTHREAD=${NTHREAD:-20}     # threads per surface; the three selections run at once
HERE=$(dirname "$(readlink -f "$0")")
source $HERE/profit_env.sh

### PROfit writes its cache and plots into the current directory
mkdir -p $OUT/plots && cd $OUT/plots

### The three selections run in the background; within each, surface reuses the
### cache that plot builds, so those two run in order

cv_only() {
    $PROFIT -x $1 -t $2 --log $2.log plot
}

cv_and_surface() {
    $PROFIT -x $1 -t $2 --log $2.log plot
    $PROFIT -x $1 -t $2 -o surf -n $NTHREAD --log $2_surf.log surface
}

cv_and_surface $OUT/GUMP/Gump$LABEL.xml       Gump$LABEL    > Gump${LABEL}_stdout.log    2>&1 &
cv_and_surface $OUT/MAPLEMP/Gumple$LABEL.xml  Gumple$LABEL  > Gumple${LABEL}_stdout.log  2>&1 &
cv_and_surface $OUT/MAPLENP/MapleNP$LABEL.xml MapleNP$LABEL > MapleNP${LABEL}_stdout.log 2>&1 &

cv_only $OUT/GUMP/GumpR2Only$LABEL.xml       GumpR2$LABEL    > GumpR2${LABEL}_stdout.log    2>&1 &
cv_only $OUT/GUMP/GumpR4Only$LABEL.xml       GumpR4$LABEL    > GumpR4${LABEL}_stdout.log    2>&1 &

cv_only $OUT/MAPLEMP/GumpleR2Only$LABEL.xml       GumpleR2$LABEL    > GumpleR2${LABEL}_stdout.log    2>&1 &
cv_only $OUT/MAPLEMP/GumpleR4Only$LABEL.xml       GumpleR4$LABEL    > GumpleR4${LABEL}_stdout.log    2>&1 &

cv_only $OUT/MAPLENP/MapleNPR2Only$LABEL.xml       MapleNPR2$LABEL    > MapleNPR2${LABEL}_stdout.log    2>&1 &
cv_only $OUT/MAPLENP/MapleNPR4Only$LABEL.xml       MapleNPR4$LABEL    > MapleNPR4${LABEL}_stdout.log    2>&1 &

wait


PYTHONPATH=$PY_PATH $PY $HERE/../plotter.py \
    -f Gumple${LABEL}_surf_surface.txt Gump${LABEL}_surf_surface.txt MapleNP${LABEL}_surf_surface.txt \
    -l Gumple Gump MapleNP -o surface_comparison_$LABEL.pdf
