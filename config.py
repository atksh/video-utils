resolution = "448:252"
skip_rate = 1
max_len = 32

num_workers = 8
total_batch_size = 32
train_batch_size = 16
predict_batch_size = 4
fps = 30

backbone_depths = [4, 4, 4, 4, 4]
enc_num_layers = [2, 2, 2, 2, 2]
dec_num_layers = [1, 1, 1, 1, 0]

last_dim = 16
feat_dims = [16, 32, 64, 128, 256]
backbone_feat_dims = feat_dims
front_feat_dims = feat_dims
enc_num_heads = [1, 2, 4, 8, 16]
dec_num_heads = [1, 1, 2, 4, 8]
n_steps = 1
