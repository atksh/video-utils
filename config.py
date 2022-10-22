resolution = "448:252"
skip_rate = 1
max_len = 32
num_workers = 8

train_batch_size = 3
predict_batch_size = 1
fps = 30
n_steps = 1


in_dim = 3
widths = [32, 48, 64, 96]
depths = [3, 3, 3, 3]
heads = [2, 3, 4, 6]
stem_dim = widths[0]
