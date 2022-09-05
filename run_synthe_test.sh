#!/bin/bash

GRA_PATH=/home/simon/grace-ilu
models=(resnet18 resnet50 resnet101 resnet152 vgg16 vgg19 densenet121 densenet201 SqueezeNet)
compressions=(none terngrad)

function main(){
    for model in ${models[*]}; do
        for comp in ${compressions[*]}; do
            echo "Using ${model} model with ${comp} method"
            horovodrun --network-interface eno1 --gloo -np 4 -H localhost:2,10.212.67.20:2 HOROVOD_CACHE_CAPACITY=0  \
		    python3 ./pytorch_synthetic_benchmark.py --optimizer grc --batch-size 64 --compression ${comp} --model ${model}
        done
    done
}

main
