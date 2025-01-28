#!/bin/bash
for m in {0.1,0.25,0.5,0.75,0.9}
do
    for ntrain in {100,200,400,600,800,1000}
    do
        python exact_runner.py --ntrain=${ntrain} --m=${m} --losstype=cmll --dataset=boston

    done
done
    for ntrain in {100,200,400,600,800,1000}
do
    python exact_runner.py --ntrain=${ntrain} --losstype=mll --dataset=boston
done
echo All done
