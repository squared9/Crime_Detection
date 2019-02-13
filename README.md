# Crime Detection
Detecting crime in progress by identifying unusual human pose sequences using Deep Learning

## Preparing training dataset

Instructions here assume you are running Ubunutu 18.04 or compatible system.

First, clone latest AlphaPose:

    git clone https://github.com/MVIG-SJTU/AlphaPose.git

Make sure you have two Ananconda environments prepared, one for AlphaPose (python3), 
the other for PoseFlow (python2). I call them "dl" and "dl2". Make sure you use requirements.txt
provided to each of the projects to set up your conda environments with pip, e.g.:

	conda create -n dl python=3
	source activate dl
	pip install -r AlphaPose/requirements.txt

and

	conda create -n dl2 python=2
	source activate dl2
	pip install -r AlphaPose/PoseFlow/requirements.txt


Make sure you have FFMPEG installed (with x264 support) by e.g.:

	sudo apt install ffmpeg


Download [UCF Crime Dataset](http://crcv.ucf.edu/cchen/) (roughly 100GB).
For training, normal videos with the following numbers are used:

	001, 004, 007, 008, 011, 012, 017, 020, 021, 023, 026, 028, 032, 037, 038,
	039, 044, 045, 052, 053, 054, 057, 058, 061, 065, 066, 068, 069, 071, 081

Copy normal videos in the form of Normal_Videos{id}_x264.mp4 to 'videos/normal' directory. {id} is from the list above.

Next, run the conversion. If you want to utilize multiple cores, edit run.sh and uncomment 'track_poses_parallel.sh'
and comment out 'track_poses.sh' that only uses 1 thread. In track_poses_parallel.sh the '-j' parameter controls how
many tracking threads are running (default is 16).

After you finished all previous steps, run the following:

	./run.sh

This could take a few hours/days/weeks depending on the performance of your computer. 
Once finished, it creates a file 'sequences.csv' that contain all training sequences.

## Training

Training would require another conda environment to be set up:

	conda create env -n cd python=3
	pip install -r requirements.txt
	source activate cd

For training, Jupyter notebook [Crime Detection](Crime_Detection.ipynb) is provided. You can run it by typing:

	jupyter notebook Crime_Detection.ipynb

There is an older notebook [Pose Detection](Pose_Detection.ipynb) that demonstrates how
to use Microsoft's state-of-art (01/2019) human pose detector, however without any tracking.

## Disclaimer

This is a work in progress




