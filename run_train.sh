python -u network.py --dataset SNIPS \
--data_dir /home/sh/data/JointSLU-DataSet/formal_snips \
--description_path data/snips_slot_description.txt \
--save_dir data/ --embed_file /home/sh/data/komninos_english_embeddings.gz \
--run_type train \
--target_domain AddToPlaylist \
--device cuda:0