### Shared PROfit setup, sourced by test_cv_surfaces.sh and test_datamc.sh
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
