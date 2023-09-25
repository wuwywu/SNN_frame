# 参数 
thresh = 0.5    # neuronal threshold
lens = 0.5      # hyper-parameters of approximate function
decay = 0.2     # decay constants
batch_size = 100
learning_rate = 1e-3    # 学习率
num_epochs = 100         # max epoch
time_window = 10        # 时间窗口
bn_fn = False            # 是否使用tdBatchNorm False True

# 1、直流编码; 2、泊松编码; 3、Rate-Syn; 4、codingTTFS coding (Time-to-first-spike coding)
encodelayer_way = 1     

# ========== 噪声参数设定 ==========
D_noise = 0.0           # 噪声强度
delta_t = 1             # 噪声步长
noise_type = "white"    # 噪声类型("white", "color")
lam_color = 0.1         # 色噪声的相关率

# ========== NMNIST参数设定 ==========
# time_window = 10        # 时间窗口
DT = 5                  # 数据集采样，时间步长，仅在时序（DVS等）数据集下有意义 DVS(dynamic vision sensor)



