source activate dl2
cd AlphaPose/PoseFlow

# parallel -j 16 python tracker-general.py --imgdir ../../images/normal/{} --in_json ../../tracked/{}/alphapose-results.json --out_json ../../tracked/{}/alphapose-results-forvis-tracked.json --visdir ../../tracked/{}/vis/ ::: 001 004 007 008 011 012 017 020 021 023 026 028 032 037 038 039 044 045 052 053 054 057 058 061 065 066 068 069 071 081
parallel -j 16 python tracker-general.py --imgdir ../../images/normal/{} --in_json ../../tracked/{}/alphapose-results.json --out_json ../../tracked/{}/alphapose-results-forvis-tracked.json --visdir ../../tracked/{}/vis/ ::: 045 052 053 054 057 058 061 065 066 068 069 071 081

cd ../..