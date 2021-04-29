CUDA_VISIBLE_DEVICES=1 python main.py --data_dir "/p/reviewde/data/ratebeer/graph/small_500/" --graph_dir "/p/reviewde/data/ratebeer/graph/small_500/graph_batch/" --epoch_num 5 --eval --batch_size 128 --learning_rate 0.0001 --optimizer "Adam" --user_embed_size 256 --item_embed_size 256 --feature_embed_size 256 --sent_embed_size 768 --hidden_size 256 --head_num 4 --ffn_inner_hidden_size 256 --ffn_dropout_rate 0.01 --attn_dropout_rate 0.01 --model_file "ratebeer_graph_sentence_extractor/model_best_4_27_6_43.pt"

# model_best_4_24_22_13.pt
# model_best_4_24_15_52.pt