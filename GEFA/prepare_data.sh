#!/bin/bash
source /home/archit3400/anaconda3/etc/profile.d/conda.sh # change this to match the location of your conda.sh file

cmd=`readlink -f $0`
cmdDir=`dirname $cmd`
repoDir=`dirname $cmdDir`
export ModelingHome="${repoDir}/RaptorX-3DModeling"

. $ModelingHome/raptorx-path.sh
. $ModelingHome/raptorx-external.sh

profile_dir="${cmdDir}/data/${1}/profile"
emb_dir="${cmdDir}/data/${1}/emb"
map_dir="${cmdDir}/data/${1}/map"
name=""

if [ ! -n "$1" ]
then
    echo "Usage: $0 dataset_name"
    exit 1
fi
conda activate RaptorX # a conda environment named RaptorX with all the required packages for running RaptorX is required to run this script
# Gener
for seq in ${cmdDir}/data/${1}/sequences/*.fasta
do
    name=`basename $seq .fasta`
    
    if [ ! -d ${map_dir}/${name} ]
    then
        mkdir ${map_dir}/${name}
    fi

    if [ ! -f ${map_dir}/${name}.npy ]
    then
        ${DistFeatureHome}/BuildMSAs.sh -d ${map_dir}/${name} $seq
        ${DistFeatureHome}/GenDistFeaturesFromMSA.sh -o ${map_dir}/${name} ${map_dir}/${name}/${name}_OUT/${name}_contact/${name}_uce5/${name}.a3m
        ${DL4DistancePredHome}/Scripts/PredictPairRelation4OneInput.sh -d ${map_dir}/${name} ${map_dir}/${name}/${name}.inputFeatures.pkl
        ${DL4DistancePredHome}/Scripts/PrintContactPrediction.sh ${map_dir}/${name}/${name}.predictedDistMatrix.pkl ${map_dir}/${name}
        python txt2npy.py ${map_dir}/${name}/${name}.CM.txt
        mv ${map_dir}/${name}/${name}.npy ${map_dir}/${name}.npy
        rm -r ${map_dir}/${name}
    fi
    
    if [ ! -d ${profile_dir}/${name}_PROP ]
    then
        mkdir ${profile_dir}/${name}_PROP
    fi

    cd ${repoDir}/Predict_Property
    ${repoDir}/Predict_Property/Predict_Property.sh -o ${profile_dir}/${name}_PROP -i $seq
    cd ${cmdDir}
done

conda activate tape # a conda environment named tape with tape proteins library installed is required
for seq in ${cmdDir}/data/${1}/sequences/*.fasta
do
    name=`basename $seq .fasta`
    if [ ! -f ${cmdDir}/data/${1}/emb/${name}.npz ]
    then
        tape-embed transformer $seq ${cmdDir}/data/${1}/emb/${name}.npz bert-base --tokenizer unirep --full_sequence_embed
    fi
done

conda activate geometric # a conda environment named geometric with libraries required to run GEFA model
python prepare_data.py $1