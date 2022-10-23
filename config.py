resolution = "448:252"
skip_rate = 1
max_len = 32
num_workers = 8

train_batch_size = 1
predict_batch_size = 1
accumulate_grad_batchs = 8
fps = 30
n_steps = 1


in_ch = out_ch = 3
widths = [8, 16, 32, 64]
depths = [2, 2, 4, 2]
heads = [1, 2, 4, 8]
head_widths = [8, 8, 8, 8]
block_sizes = [8, 8, 8, 8]
kernel_sizes = [3, 3, 3, 3]
dec_depths = [1, 1, 1]
