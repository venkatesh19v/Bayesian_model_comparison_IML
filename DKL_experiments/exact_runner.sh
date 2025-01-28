#!/bin/bash

for dset in {boston,power,energy,concrete,winered,winewhite}
do
    for m in {0.75,0.9}
    # for m in {0.1,0.25,0.5,0.75,0.9}
    do
        for ntrain in {200,400,600}
        do
            python exact_runner.py --ntrain=${ntrain} --m=${m} --losstype=cmll --dataset=${dset}
        done
    done

    for ntrain in {200,400,600}
    do
        python exact_runner.py --ntrain=${ntrain} --losstype=mll --dataset=${dset}
    done
done
echo All done