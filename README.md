# ME5406-project2
This is the code for NUS ME5406 Project2. Our team members are: Zhiyao Ren, Jiawei Chen and Boxuan Men.

The goal of our project is to train the carts to move forward on the straights, 45 degree bend, 90 degree bend and 135 degree bend using deep reinforcement learning algorithms. And can complete our randomly generated complete track. The example of the track is shown below:

## Installation
Clone this repository and navigate to it in your terminal. Then run:

```
pip install -r requirements.txt
```

## Design track
You need to change the `turning_point` in `main.py` before training or testing. The track should start in `(400, 500)` and the width and height of the track image should be at least 500 larger than the maximum values of x and y in `turning_point`. You can set `--track_w` and `--track_h` to change the width and height of the track image.

You can use `lib/create_track` to help you visualize the track and try to manually move forward on it.

## Running the code
You can running the training or testing code by:

```
bash main.sh x
```

x means how many episodes you want in the training process. If you are running the testing process, you can choose any x.

You can set ip this run in `main.sh`. you should set `--type Train` for training or set `--type Test` for testing. You can select to algorithm '--algorithm DQN'. We support DQN, DoubleDQN and DuelingDQN in our code. You can use `--resume_checkpoint path/to/model.pt` to load your model for testing or further training. You can also set some hyperparameters like `--batch_size 32`, `--lr 5e-5` and `--epsilon 0.95`.

## Test our code
You can download our trained model [[checkpoint](https://drive.google.com/drive/folders/19h5doLD4dR8IcrVsNhTK_q6gophP124X?usp=share_link)]. You can random choose a `turning_point` in `main.py`. This is a model for DuelingDQN, please set following in `main.sh`:

```
--algorithm DuelingDQN --type Test --resume_checkpoint path/to/model.pt --track_h 1300 --track_w 2100
```
