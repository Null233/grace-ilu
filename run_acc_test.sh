#!/bin/bash

GRA_PATH=/home/simon/grace-ilu
models=(ResNet18 ResNet50)
compressions=(ternallreduce)

function main(){
    for model in ${models[*]}; do
        for comp in ${compressions[*]}; do
            echo "Using ${model} model with ${comp} method"
            horovodrun --gloo -np 2 python3 ${GRA_PATH}/supervised_classificatin/main.py \
                        --compression ${comp} --model ${model}
        done
    done
}

main
