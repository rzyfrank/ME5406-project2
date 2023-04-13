#!/usr/bin/env sh

set -e
echo script name: $0
echo $# arguments
if [ "$#" -ne 1 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for gpu devices and the model and the sigma"
  exit 1
fi
episode=$1

python main.py --episode ${episode} \
  --batch_size 32 --lr 5e-5 --save_freq 500 --algorithm DQN --type Train --track_h 1300 --track_w 2100 --gamma 0.9 --resume_checkpoint ./checkpoint/model.pt