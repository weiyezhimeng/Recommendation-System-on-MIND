def GPU():
    import torch
    current_gpu_index = torch.cuda.current_device()
    # 获取GPU显存的总量和已使用量
    total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory / (1024 ** 3)  # 显存总量(GB)
    used_memory = torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)  # 已使用显存(GB)
    free_memory = total_memory - used_memory  # 剩余显存(GB)
    print(f"GPU显存总量：{total_memory:.2f} GB")
    print(f"已使用的GPU显存：{used_memory:.2f} GB")
    print(f"剩余GPU显存：{free_memory:.2f} GB")