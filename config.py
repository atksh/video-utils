resolution = "448:252"
skip_rate = 1
max_len = 32

num_workers = 8
total_batch_size = 32
train_batch_size = 8
predict_batch_size = 4
fps = 30
n_steps = 1


in_dim = 3
stem_dim = 16
widths = [16, 32, 64, 128, 256]
depths = [1, 1, 1, 1, 1]
heads = [1, 1, 2, 4, 8]
drop_p = 0.0
