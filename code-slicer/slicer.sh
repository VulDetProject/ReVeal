VERBOSE=1
if [[ $# -ge 4 ]]; then
    inpdir=$1;
    filename=$2;
    outdir=$4;
    lineno=$3;
    dfonly='';
    if [[ $VERBOSE = 1 ]]; then
      VERB='--verbose';
    fi
    if [[ $# = 5 ]]; then
      dfonly='--data_flow_only'
    fi
    mkdir -p 'tmp';
    cp $inpdir'/'$filename 'tmp/'$filename;
    if [[ -d parsed ]]; then
      rm -rf parsed;
    fi
    ./joern/joern-parse tmp/;
    python parse_joern_output.py --code $filename --line $lineno --output $outdir $dfonly $VERB;
#    rm -rf parsed;
#    rm -rf tmp;

else
  echo 'Wrong Argument!.'
  echo 'slicer.sh <Directory of the C File> <Name of the C File> <Line For Slice> <Output Directory> <DataFlowOnly(Optional, used if mentioned)>'
fi
