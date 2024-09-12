initial_lr = 0.001
final_lr = 0.0001
total_iterations = 200
decay_rate = (final_lr / initial_lr) ** (1 / total_iterations)

for e in range(total_iterations):
    # 更新学习率
    lr_trigger = initial_lr * (decay_rate ** e)

    # 假设 cfg 是一个配置对象，并且我们需要在训练过程中使用 current_lr
    # 在实际代码中，cfg.lr_trigger 可以用于更新优化器的学习率
    # 例如：
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = current_lr

    # 这里放置你的训练代码，比如前向传播、计算损失、反向传播和优化步骤
    print(f"Iteration {e + 1}, Learning Rate: {lr_trigger}")
