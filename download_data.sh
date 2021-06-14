wget -P ./lp_rp/data/ https://nodepiece-data.s3.ca-central-1.amazonaws.com/lp/fb15k237_1000_anchors_1000_paths_d0.4_b0.0_p0.4_r0.2_pykeen_100sp.pkl
wget -P ./lp_rp/data/ https://nodepiece-data.s3.ca-central-1.amazonaws.com/lp/wn18rr_500_anchors_500_paths_d0.4_b0.0_p0.4_r0.2_pykeen_100sp.pkl
wget -P ./lp_rp/data/ https://nodepiece-data.s3.ca-central-1.amazonaws.com/lp/yago_10000_anchors_10000_paths_d0.4_b0.0_p0.4_r0.2_pykeen_100sp.pkl
wget -P ./lp_rp/data/ https://nodepiece-data.s3.ca-central-1.amazonaws.com/lp/codex_l_7000_anchors_7000_paths_d0.4_b0.0_p0.4_r0.2_pykeen_100sp.pkl

wget -P ./nc/data/ https://nodepiece-data.s3.ca-central-1.amazonaws.com/nc/wd50k_50_anchors_50_paths_d0.4_b0.0_p0.4_r0.2_pykeen_100sp.pkl
wget -P ./nc/data/clean/wd50k/statements/ https://nodepiece-data.s3.ca-central-1.amazonaws.com/nc/wd50k_embs.pkl

wget -P ./oos_lp/src/data/ https://nodepiece-data.s3.ca-central-1.amazonaws.com/oos-lp/FB15k-237_1000_anchors_1000_paths_d0.4_b0.0_p0.4_r0.2_pykeen_100sp.pkl
wget -P ./oos_lp/src/data/ https://nodepiece-data.s3.ca-central-1.amazonaws.com/oos-lp/YAGO3-10_10000_anchors_10000_paths_d0.4_b0.0_p0.4_r0.2_pykeen_100sp.pkl

wget -P ./oos_lp/datasets/FB15k-237/processed/ https://nodepiece-data.s3.ca-central-1.amazonaws.com/oos_datasets/ofb15k237/train.txt
wget -P ./oos_lp/datasets/FB15k-237/processed/ https://nodepiece-data.s3.ca-central-1.amazonaws.com/oos_datasets/ofb15k237/valid.json
wget -P ./oos_lp/datasets/FB15k-237/processed/ https://nodepiece-data.s3.ca-central-1.amazonaws.com/oos_datasets/ofb15k237/test.json

wget -P ./oos_lp/datasets/YAGO3-10/processed/ https://nodepiece-data.s3.ca-central-1.amazonaws.com/oos_datasets/oyago/train.txt
wget -P ./oos_lp/datasets/YAGO3-10/processed/ https://nodepiece-data.s3.ca-central-1.amazonaws.com/oos_datasets/oyago/valid.json
wget -P ./oos_lp/datasets/YAGO3-10/processed/ https://nodepiece-data.s3.ca-central-1.amazonaws.com/oos_datasets/oyago/test.json