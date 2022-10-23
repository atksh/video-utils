resolution = "448:252"
skip_rate = 5
max_len = 8
num_workers = 8

train_batch_size = 4
predict_batch_size = 1
accumulate_grad_batchs = 1
fps = 30
n_steps = 1


in_ch = out_ch = 3
widths = [32, 64, 128, 256]
depths = [1, 2, 3, 4]
heads = [1, 2, 4, 8]
head_widths = [32, 32, 32, 32]
block_sizes = [16, 8, 4, 4]
kernel_sizes = [7, 7, 5, 3]
dec_depths = [1, 1, 1, 1]
resolution_scale = 2
