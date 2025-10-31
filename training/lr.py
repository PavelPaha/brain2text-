from torch.optim.lr_scheduler import LRScheduler, ConstantLR, CosineAnnealingLR, ExponentialLR, LinearLR, SequentialLR



def get_main_scheduler(optimizer, lr_cfg, total_steps):
    if lr_cfg.type == 'cosine':
        if lr_cfg.warmup:
            total_steps -= int(total_steps * lr_cfg.warmup_steps)
        return CosineAnnealingLR(optimizer, T_max=total_steps-1, eta_min=lr_cfg.low_lr)
    if lr_cfg.type == 'constant':
        return ConstantLR(optimizer, factor=1.)
    if lr_cfg.type == 'exponential':
        return ExponentialLR(optimizer, gamma=lr_cfg.gamma)


def get_lr_scheduler(optimizer, lr_cfg, total_steps) -> LRScheduler:
    main_scheduler = get_main_scheduler(optimizer, lr_cfg, total_steps)

    if not lr_cfg.warmup:
        return main_scheduler
    
    warmup_steps = int(total_steps * lr_cfg.warmup_steps)
    print(int(total_steps*lr_cfg.warmup_steps))
    schedulers = [
        LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps),
        main_scheduler
    ]
    
    return SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=[warmup_steps]
    )