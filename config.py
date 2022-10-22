resolution = "448:252"
skip_rate = 1
max_len = 32
num_workers = 8

train_batch_size = 1
predict_batch_size = 1
accumulate_grad_batchs = 8
fps = 30
n_steps = 1


in_dim = 3
widths = [32, 64, 128, 192]
depths = [2, 2, 4, 2]
heads = [1, 2, 4, 6]
stem_dim = widths[0]
