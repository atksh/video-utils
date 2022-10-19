resolution = "426:240"
skip_rate = 4
max_len = 32

num_workers = 16
total_batch_size = 32
train_batch_size = 8
predict_batch_size = 2
fps = 30

backbone_feat_dims = [80, 160, 320, 640]
front_feat_dims = [32, 32, 64, 96]
enc_num_heads = [1, 1, 2, 3]
enc_num_layers = [4, 4, 32, 4]
dec_num_heads = [1, 1, 2, 3]
dec_num_layers = [2, 2, 16, 2]
n_steps = 4
last_dim = 32
