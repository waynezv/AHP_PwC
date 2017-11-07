#!/bin/bash

# Extract openSMILE features from audios.
# wzhao1 cs cmu edu
# 11/02/2017

if [[ $# -lt 5 ]]; then
    echo "USAGE: $0 [-c] openSMILE configure file
    [-f] ctl file containing list of wavs to be processed
    [-i] indir [-o] outdir [-t] tmpdir"

    exit 0
fi

# Parse arguments
while [[ $# -gt 1 ]]; do
    key="$1"

    case "$key" in
        -c)
            CONFIG="$2"  # openSMILE configure file
            shift
            ;;
        -f)
            FILELIST="$2"  # files to be processed
            shift
            ;;
        -i)
            INDIR="$2"  # input dir
            shift
            ;;
        -o)
            OUTDIR="$2"  # output dir
            shift
            ;;
        -t)
            TMPDIR="$2"  # temporary dir
            shift
            ;;
        *)
            echo "Unknown option!"
            ;;
    esac

    shift
done

# Check availability of necessary files
[[ -f "$CONFIG" ]] || ( echo "$CONFIG not found"; exit 1 )
[[ -f "$FILELIST" ]] || ( echo "$FILELIST not found"; exit 1 )
[[ -d "$INDIR" ]] || ( echo "$INDIR not found"; exit 1 )
[[ -d "$OUTDIR" ]] || ( echo "$OUTDIR not found"; exit 1 )
[[ -d "$TMPDIR" ]]|| ( echo "$TMPDIR not found"; exit 1 )

opensmile=./opensmile-2.3.0/bin/linux_x64_standalone_libstdc6/SMILExtract  # openSMILE executable

TMPFILE=$( mktemp --tmpdir=${TMPDIR} tmp.XXXX )  # make tmp file

function clean_up {  # clean up tmp file
    rm -rf "$TMPFILE"
    exit 0
}
trap clean_up 0 1 2 3  # clean up on exit

echo "Extracting openSMILE features for ${FILELIST} ..."

for f in $( cat "${FILELIST}" ); do
    infile="${INDIR}/${f}"  # input wav filename
    [ -f "${infile}" ] || ( echo "${infile} not found"; exit 1 )

    DIR=$( dirname "$f" )
    BASE=$( basename "$f" )
    OUTPATH="${OUTDIR}/${DIR}"
    [[ -e "$OUTPATH" ]] || mkdir -p "$OUTPATH"  # make output dir

    outfile="${OUTPATH}/${BASE%.*}.smile"  # output feature filename
    [ ! -f "${outfile}" ] || echo "${outfile} duplicated"

    # Execute openSMILE extractor
    ${opensmile} -C $CONFIG -I $infile -csvoutput $outfile -logfile $TMPFILE > /dev/null 2>&1
    echo "processed $infile to $outfile"
done

echo "Done."
exit 0
