./make_image_dir.sh
./extract_images.sh
./make_tracked_dir.sh
./extract_poses.sh
./track_poses.sh

source activate dl
python continuous_pose_extraction.py
python pose_pair_extraction.py
