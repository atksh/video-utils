resolution = "426:240"
skip_rate = 4
max_len = 16

num_workers = 16
total_batch_size = 32
train_batch_size = 16
predict_batch_size = 4
fps = 30

backbone_feat_dims = [80, 160, 320, 640]
front_feat_dims = [32, 48, 64, 96]
enc_num_heads = [2, 3, 4, 6]
enc_num_layers = [3, 6, 9, 12]
dec_num_heads = [2, 2, 3, 4]
dec_num_layers = [1, 2, 3, 4]
n_steps = 4
last_dim = 32
