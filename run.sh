intent=(AddToPlaylist BookRestaurant GetWeather PlayMusic RateBook SearchCreativeWork SearchScreeningEvent)
# BookRestaurant GetWeather PlayMusic RateBook SearchCreativeWork SearchScreeningEvent)

for ((i=0; i<${#intent[*]}; i++))
do
	python -u network.py --dataset SNIPS \
	--data_dir /home/sh/data/JointSLU-DataSet/formal_snips \
	--bidirectional Ture \
	--dropout 0.5 \
	--crf True \
	--epoch 20 \
	--log_every 20 \
	--log_valid 300 \
	--patience 5 \
	--max_num_trial 5 \
	--lr_decay 0.5 \
	--learning_rate 0.001 \
	--batch_size 16 \
	--description_path data/snips_slot_description.txt \
	--save_dir data1/ --embed_file /home/sh/data/komninos_english_embeddings.gz \
	--run_type train \
	--target_domain ${intent[$i]} \
	--device cuda:0
	
	python -u network.py --run_type test \
	--save_dir data1/ \
	--target_domain ${intent[$i]} \
	--device cuda:0
done

