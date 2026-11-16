#!/bin/bash
### Copy the sbruce files the ANL XMLs read to Aurora, plus any other files given.
### For each selection that is the *_sbruce.root files at its top level and its
### sbruce_trees/ and FD/ directories. One MobilePASS+ login, asked for only after you say y.
### Usage:  send_to_aurora.sh SRC DEST OUT FILE [FILE ...]
###   SRC   local storage root holding the selection directories, e.g. $EXP
###   DEST  root on Aurora that the ANL XML uses as BASE_DIR, e.g. $ANL
###   OUT   local directory the FILEs sit under; each keeps its path relative to OUT,
###         so $OUT/GUMP/Gump.xml goes to $DEST/GUMP/Gump.xml
###   FILE  files to copy, e.g. the rendered XMLs
### Set AURORA_USER if your ALCF username differs from your Fermilab one.
### Job submission is off for now. To turn it on, set AURORA_JOB_SCRIPT to the
### PBS script's path on Aurora; it is submitted over the same login after the copy.
SRC=$1
DEST=$2
OUT=$3
shift 3
FILES=("$@")
AURORA_JOB_SCRIPT=${AURORA_JOB_SCRIPT:-}

AURORA_USER=${AURORA_USER:-nathanielerowe}
AURORA_HOST=aurora.alcf.anl.gov
REMOTE=$AURORA_USER@$AURORA_HOST

### Selection directories under SRC/DEST, and the subdirectories of each that hold
### sbruce files
SELECTIONS="GUMP MAPLEMP MAPLENP"
SUBDIRS="sbruce_trees FD"

### Shared SSH connection, so scp and ssh below reuse the one login
SOCKET=~/.ssh/cm-aurora-%r@%h:%p
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="$SOCKET" -o ControlPersist=10m)

### The prompt needs someone at a terminal (not cron or a background job)
if [ ! -t 0 ]; then
    echo "No interactive terminal: skipping the Aurora transfer. Run by hand:" >&2
    echo "  $0 $SRC $DEST $OUT ${FILES[*]}" >&2
    exit 1
fi

while true; do
    read -r -p "Are you ready to transfer to Aurora? (y/n) " ans
    case "$ans" in
        [yY]) break ;;
        [nN]) echo "Skipping the Aurora transfer."; exit 0 ;;
        *)    echo "Please enter y or n." ;;
    esac
done

cleanup() { ssh "${SSH_OPTS[@]}" -O exit "$REMOTE" 2>/dev/null; }
trap cleanup EXIT

### The only passcode prompt
echo "Logging in to $REMOTE: enter your MobilePASS+ passcode when prompted."
ssh "${SSH_OPTS[@]}" -fN "$REMOTE" || { echo "Login to Aurora failed" >&2; exit 1; }

### Make every destination directory in one call
DIRS=""
for sel in $SELECTIONS; do DIRS="$DIRS '$DEST/$sel'"; done
for f in "${FILES[@]}"; do DIRS="$DIRS '$DEST/$(dirname "${f#$OUT/}")'"; done
ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p $DIRS" || exit 1

RC=0
shopt -s nullglob
for sel in $SELECTIONS; do
    ### Top-level sbruce files, in one scp
    TOP=("$SRC/$sel"/*_sbruce.root)
    if [ ${#TOP[@]} -gt 0 ]; then
        echo "Copying ${#TOP[@]} $sel/*_sbruce.root files ..."
        scp "${SSH_OPTS[@]}" -q "${TOP[@]}" "$REMOTE:$DEST/$sel/" || { echo "Copy of $sel/*_sbruce.root failed" >&2; RC=1; }
    else
        echo "Warning: no $SRC/$sel/*_sbruce.root files found" >&2
        RC=1
    fi

    ### sbruce_trees/ and FD/
    for sub in $SUBDIRS; do
        if [ ! -d "$SRC/$sel/$sub" ]; then
            echo "Warning: $SRC/$sel/$sub not found, not copied" >&2
            RC=1; continue
        fi
        echo "Copying $sel/$sub ..."
        scp "${SSH_OPTS[@]}" -rq "$SRC/$sel/$sub" "$REMOTE:$DEST/$sel/" || { echo "Copy of $sel/$sub failed" >&2; RC=1; }
    done
done

echo "Copying ${#FILES[@]} files ..."
for f in "${FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "Warning: $f not found, not copied" >&2
        RC=1; continue
    fi
    rel=${f#$OUT/}
    scp "${SSH_OPTS[@]}" -q "$f" "$REMOTE:$DEST/$rel" || { echo "Copy of $rel failed" >&2; RC=1; }
done

[ $RC -eq 0 ] && echo "Aurora transfer done: $DEST" || echo "Aurora transfer finished with errors" >&2

### Job submission (off unless AURORA_JOB_SCRIPT is set). Only submits if every
### copy succeeded, so a job never runs on a partial set of files.
submit_job() {
    ### bash -lc loads the login environment, so qsub is on the PATH
    JOBID=$(ssh "${SSH_OPTS[@]}" "$REMOTE" "bash -lc 'cd $DEST && qsub $AURORA_JOB_SCRIPT'") ||
        { echo "qsub failed" >&2; return 1; }
    echo "Submitted Aurora job: $JOBID"
}

if [ -n "$AURORA_JOB_SCRIPT" ]; then
    if [ $RC -eq 0 ]; then
        submit_job || RC=1
    else
        echo "Not submitting $AURORA_JOB_SCRIPT because the transfer had errors" >&2
    fi
fi

exit $RC
