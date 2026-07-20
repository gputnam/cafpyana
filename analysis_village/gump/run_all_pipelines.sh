#!/bin/bash

# Define the absolute input storage directories
gray_prefix='/exp/sbnd/data/users/gputnam/GUMP/sbn-rewgted-7/'
output='/exp/sbnd/data/users/nrowe/GUMP/sbn-rewgted-7/opt_cuts/'
MAX_JOBS=6

# Navigate to the working directory context
echo "========================================================"
echo " Starting GUMP TTree Processing Batch Run...            "
echo "========================================================"

echo "Remaking det var maps..."
python3 rwt_map.py

## 1. SBND MC (20 files, 0 to 19)
echo "--> Staging SBND Spring MC Files..."
for i in {0..19}
do
    while [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; do
        sleep 10 # Check every 2 seconds
    done

    echo "Launching SBND MC Step $i"
    python3 run_gump_pipeline.py \
        -c mc \
        -i ${gray_prefix}SBND_SpringMC_rewgt_E_${i}.df \
        -o ${output}SBND_SpringMC_rewgt_E_${i}_sbruce.root &
done

## 2. ICARUS Run 4 MC (10 files, 0 to 9)
echo "--> Staging ICARUS Run 4 MC Files..."
for i in {0..9}
do
    while [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; do
        sleep 10 # Check every 2 seconds
    done

    echo "Launching ICARUS Run 4 MC Step $i"
    python3 run_gump_pipeline.py \
        -c mc \
        -i ${gray_prefix}ICARUSRun4_SpringMCOverlay_rewgt_${i}.df \
        -o ${output}ICARUSRun4_SpringMCOverlay_rewgt_${i}_sbruce.root &
done

while [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; do
    sleep 10 # Check every 2 seconds
done

### 3. ICARUS Run 2 MC
echo "--> Launching ICARUS Run 2 MC..."
python3 run_gump_pipeline.py \
    -c mc \
    -i ${gray_prefix}ICARUS_SpringMCOverlay_rewgt.df \
    -o ${output}ICARUS_SpringMCOverlay_rewgt_sbruce.root &

while [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; do
    sleep 10 # Check every 2 seconds
done

### 4. ICARUS Run 2 OffBeam
echo "--> Launching ICARUS Run 2 OffBeam Data..."
python3 run_gump_pipeline.py \
    -c data \
    -i ${gray_prefix}ICARUS_SpringRun2BNBOff_unblind_prescaled.df \
    -o ${output}ICARUS_SpringRun2BNBOff_unblind_prescaled_sbruce.root &

while [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; do
    sleep 10 # Check every 2 seconds
done

### 5. ICARUS Run 4 OffBeam
echo "--> Launching ICARUS Run 4 OffBeam Data..."
python3 run_gump_pipeline.py \
    -c data \
    -i ${gray_prefix}ICARUS_SpringRun4BNBOff_unblind_prescaled.df \
    -o ${output}ICARUS_SpringRun4BNBOff_unblind_prescaled_sbruce.root &

while [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; do
    sleep 10 # Check every 2 seconds
done

### 6. SBND OffBeam
echo "--> Launching SBND OffBeam Data..."
python3 run_gump_pipeline.py \
    -c data \
    -i ${gray_prefix}SBND_SpringBNBOffData_5000.df \
    -o ${output}SBND_SpringBNBOffData_5000_sbruce.root &

while [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; do
    sleep 10 # Check every 2 seconds
done

### 7. ICARUS Run 2 Dirt
echo "--> Launching ICARUS Run 2 Dirt..."
python3 run_gump_pipeline.py \
    -c data \
    -i ${gray_prefix}ICARUS_Spring_Overlay_Dirt_lowE.df \
    -o ${output}ICARUS_Spring_Overlay_Dirt_lowE_sbruce.root &

while [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; do
    sleep 10 # Check every 2 seconds
done

### 8. ICARUS Run 4 Dirt
echo "--> Launching ICARUS Run 4 Dirt..."
python3 run_gump_pipeline.py \
    -c data \
    -i ${gray_prefix}ICARUSRun4_Spring_Overlay_Dirt_lowE.df \
    -o ${output}ICARUSRun4_Spring_Overlay_Dirt_lowE_sbruce.root &

while [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; do
    sleep 10 # Check every 2 seconds
done

### 9. SBND Dirt
echo "--> Launching SBND Dirt..."
python3 run_gump_pipeline.py \
    -c data \
    -i ${gray_prefix}SBND_SpringLowEMC.df \
    -o ${output}SBND_SpringLowEMC_sbruce.root &
