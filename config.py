resolution = "448:252"
skip_rate = 1
max_len = 32

num_workers = 8
train_batch_size = 6
predict_batch_size = 4
fps = 30
n_steps = 1


in_dim = 3
stem_dim = 32
widths = [32, 64, 96, 128, 160]
depths = [3, 3, 3, 3, 3]
heads = [1, 2, 3, 4, 5]
drop_p = 0.0
