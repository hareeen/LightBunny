if [ $# -lt 2 ]; then
    echo "Region range is not given"
    exit 1
fi

batches=$((($2 - $1 + 15) / 16))
minutes=$(($batches*595/19))

echo "This script will generate a$(($1)) to a$(($2 - 1)) inclusively, divided into $batches batches."
echo "Estimated time is $(($minutes / 60)) hour(s) $(($minutes % 60)) minute(s)."
read -p "Confirm? [y/N] " REPLY
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    for (( start_region=$1; start_region<$2; start_region+=16 ))
    do
        end_region=$((start_region+16))

        if [ $end_region -gt $2 ]; then
            end_region=$2
        fi

        if [ -d "../source/raw_nmm" ]; then
            rm -r "../source/raw_nmm"
        fi

        if [ ! -d "../source/raw_nmm" ]; then
            mkdir ../source/raw_nmm
        fi

        python -W ignore generate_tvb_data.py --a_start=$start_region --a_end=$end_region
        matlab -batch "process_raw_nmm('regions', $(($start_region + 1)):$end_region)"
    done
fi
