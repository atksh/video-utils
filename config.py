resolution = "448:252"
skip_rate = 1
max_len = 32

num_workers = 8
total_batch_size = 32
train_batch_size = 16
predict_batch_size = 4
fps = 30

backbone_depths = [8, 16, 8, 4]
enc_num_layers = [8, 16, 8, 4]
dec_num_layers = [8, 16, 8, 0]

feat_dims = [16, 32, 64, 128]
backbone_feat_dims = feat_dims
front_feat_dims = feat_dims
enc_num_heads = [1, 2, 4, 8]
dec_num_heads = [1, 1, 2, 4]
n_steps = 1
last_dim = 16
