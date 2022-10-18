resolution = "426:240"
skip_rate = 1
max_len = 16

num_workers = 16
total_batch_size = 32
train_batch_size = 32
predict_batch_size = 4
fps = 30

backbone_feat_dims = [80, 160, 320, 640]
front_feat_dims = [32, 32, 64, 96]
enc_num_heads = [1, 1, 2, 3]
enc_num_layers = [2, 2, 4, 8]
dec_num_heads = [1, 1, 2, 2]
dec_num_layers = [2, 2, 4, 8]
n_steps = 4
last_dim = 32
