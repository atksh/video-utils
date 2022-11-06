resolution = "448:252"
skip_rate = 5
max_len = 16
num_workers = 8

train_batch_size = 1
predict_batch_size = 1
accumulate_grad_batchs = 16
fps = 30
n_steps = 1


in_ch = out_ch = 3
widths = [16, 32, 64, 128]
depths = [1, 1, 1, 1]
heads = [1, 2, 4, 8]
head_widths = [16, 16, 16, 16]
block_sizes = [16, 8, 4, 4]
kernel_sizes = [7, 7, 5, 3]
dec_depths = [1, 1, 1, 1]
resolution_scale = 2
