resolution = "426:240"
skip_rate = 1
max_len = 32

num_workers = 4
total_batch_size = 32
train_batch_size = 8
predict_batch_size = 4
fps = 30

backbone_feat_dims = [80, 160, 320, 640]
front_feat_dims = [16, 32, 64, 128]
enc_num_heads = [1, 2, 4, 8]
enc_num_layers = [3, 6, 9, 12]
dec_num_heads = [1, 1, 2, 4]
dec_num_layers = [1, 2, 3, 4]
n_steps = 4
last_dim = 32
