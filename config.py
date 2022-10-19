resolution = "426:240"
skip_rate = 4
max_len = 32

num_workers = 16
total_batch_size = 32
train_batch_size = 8
predict_batch_size = 2
fps = 30

backbone_feat_dims = [80, 160, 320, 640]
front_feat_dims = [32, 64, 128, 256]
enc_num_heads = [1, 2, 4, 8]
enc_num_layers = [2, 2, 4, 8]
dec_num_heads = [1, 1, 2, 4]
dec_num_layers = [1, 1, 1, 1]
n_steps = 4
last_dim = 32
