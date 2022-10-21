resolution = "448:252"
skip_rate = 1
max_len = 32

num_workers = 8
total_batch_size = 32
train_batch_size = 32
predict_batch_size = 4
fps = 30

backbone_depths = [8, 8, 8, 8]
enc_num_layers = [3, 6, 9, 12]
dec_num_layers = [2, 4, 6, 0]

feat_dims = [16, 32, 64, 128]
backbone_feat_dims = feat_dims
front_feat_dims = feat_dims
enc_num_heads = [1, 2, 4, 8]
dec_num_heads = [1, 1, 2, 4]
n_steps = 1
last_dim = 16
