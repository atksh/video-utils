resolution = "448:252"
skip_rate = 1
max_len = 32

num_workers = 8
train_batch_size = 1
predict_batch_size = 4
fps = 30
n_steps = 1


in_dim = 3
widths = [8, 16, 32, 64]
depths = [3, 3, 3, 3]
heads = [1, 2, 4, 8]
stem_dim = widths[0]
