# train_semi.py
# train_semi.py

from utils.diffusion_utils import (
    init_diffusion_model,
    generate_aug_images
)
import torchvision.utils as vutils
import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val_dual as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save, one_flat_cycle)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss_tal_dual import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # Automatic DDP Multi-GPU argument
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = None  # check_git_info()

def train(hyp, opt, device, callbacks):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')
    # 在 train(...) 中，最开头的地方，或者在 save_dir 确定后
    save_dir_diffusion = save_dir / "diffusion_debug"  # 例如 "runs/train/exp/diffusion_debug"
    save_dir_diffusion2 = save_dir / "combined_debug"  # 例如 "runs/train/exp/diffusion_debug"
    save_dir_diffusion.mkdir(parents=True, exist_ok=True)
    save_dir_diffusion2.mkdir(parents=True, exist_ok=True)
    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    hyp['anchor_t'] = 5.0
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith('val2017.txt')  # COCO dataset

    # 初始化扩散模型(只需一次)
    diffusion_pipe = init_diffusion_model(device=device)
    
    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')

    # Initialize student and teacher models
    student_model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    teacher_model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    teacher_model.load_state_dict(student_model.state_dict())  # Initialize teacher model with student weights
    for param in teacher_model.parameters():
        param.requires_grad = False  # Freeze teacher model parameters

    # Load checkpoint if resuming
    ckpt = None
    csd = None
    # 1️⃣ 初始化优化器（提前到这里）
    optimizer = smart_optimizer(student_model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    # 创建 EMA（必须也先于 resume）
    ema = ModelEMA(student_model) if RANK in {-1, 0} else None
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        # 下面一行注意：如果是旧版PyTorch没有 weights_only 参数，请改成常规写法，如：ckpt = torch.load(weights, map_location='cpu')
        # ckpt = torch.load(weights, map_location='cpu', weights_only=False)
        ckpt = torch.load(weights, map_location='cpu', weights_only=False)
        state_dict = ckpt['model'].float().state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items()
                               if k in student_model.state_dict() and student_model.state_dict()[k].shape == v.shape}
        student_model.load_state_dict(filtered_state_dict, strict=False)
        start_epoch = 0

        # if resume:
        #     # 如果要真正恢复 optimizer、epoch 等，需要先定义 optimizer 后再执行 smart_resume
        #     # 下面仅保留示例:
            
        #     pass
        if resume and 'epoch' in ckpt:
            LOGGER.info(f"Resuming training from epoch {ckpt['epoch'] + 1}")
            start_epoch = ckpt['epoch'] + 1
            LOGGER.info(f"✅ Checkpoint loaded. Resuming training at epoch {start_epoch}/{epochs}")
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'best_fitness' in ckpt:
                best_fitness = ckpt['best_fitness']
            if 'ema' in ckpt and ema:
                ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                
    if ckpt is not None:
        del ckpt, csd  # free memory

    # Image size
    gs = max(int(student_model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(student_model, imgsz, amp=check_amp(student_model))
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(student_model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    elif opt.flat_cos_lr:
        lf = one_flat_cycle(1, hyp['lrf'], epochs)  # flat + cosine
    elif opt.fixed_lr:
        lf = lambda x: 1.0
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(student_model) if RANK in {-1, 0} else None

    # Trainloader (labeled data)
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              close_mosaic=opt.close_mosaic != 0,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              min_items=opt.min_items)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}'

    # Val loader
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            student_model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end', labels, names)

    # DDP mode
    if cuda and RANK != -1:
        student_model = smart_DDP(student_model)

    # Attach some attributes
    nl = de_parallel(student_model).model[-1].nl  # number of detection layers
    hyp['label_smoothing'] = opt.label_smoothing
    student_model.nc = nc
    student_model.hyp = hyp
    student_model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    student_model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations
    last_opt_step = -1
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = 0 - 1
    scaler = torch.amp.GradScaler('cuda', enabled=check_amp(student_model))
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(student_model)
    consistency_loss_fn = nn.MSELoss()  # Consistency loss
    callbacks.run('on_train_start')

    best_fitness = 0.0
    if not resume:
        start_epoch = 0  # ✅ 只有在不 resume 的时候才设为 0

    scheduler.last_epoch = start_epoch - 1  # ✅ 确保 lr scheduler 正确继续
    
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f"✅ Checkpoint loaded. Resuming training at epoch {start_epoch}/{epochs}\n"
                f'Starting training for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):  # epoch loop
        callbacks.run('on_train_epoch_start')
        student_model.train()

        # Update image weights if needed
        if opt.image_weights:
            cw = student_model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)

        if epoch == (epochs - opt.close_mosaic):
            LOGGER.info("Closing dataloader mosaic")
            dataset.mosaic = False

        mloss = torch.zeros(3, device=device)
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)

        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'cls_loss', 'dfl_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)
        optimizer.zero_grad()

        # -------------------- 1) 训练有标注数据（常规有监督部分） --------------------
        for i, (imgs, targets, paths, _) in pbar:
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # batch index overall
            imgs = imgs.to(device, non_blocking=True).float() / 255

            # 先调用你的扩散生成函数
            aug_imgs, aug_targets = generate_aug_images(imgs, targets, diffusion_pipe, device=device)
                # Warmup
            if i < 5:
                for idx in range(aug_imgs.size(0)):
                    img_to_save = aug_imgs[idx].detach().cpu()
                    save_path = save_dir_diffusion / f"epoch{epoch}_batch{i}_aug{idx}.png"
                    vutils.save_image(img_to_save, save_path, normalize=False)
                    
            # combined_imgs = torch.cat([imgs, aug_imgs], dim=0)
            # # 确保 targets 已在 device 上
            # if i < 5:
            #     for idx in range(combined_imgs.size(0)):
            #         img_to_save = combined_imgs[idx].detach().cpu()
            #         save_path = save_dir_diffusion2 / f"epoch{epoch}_batch{i}_aug{idx}.png"
            #         vutils.save_image(img_to_save, save_path, normalize=False)
                    
            # targets = targets.to(device)
            # aug_targets[:, 0] += imgs.shape[0]  # offset batch_idx
            # combined_targets = torch.cat([targets, aug_targets], dim=0)
       


            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi,
                                        [hyp['warmup_bias_lr'] if j == 0 else 0.0,
                                         x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi,
                                                  [hyp['warmup_momentum'],
                                                   hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5 + gs)) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # # Forward + loss
            # with torch.amp.autocast(device_type='cuda', enabled=check_amp(student_model)):
            #     # pred = student_model(imgs)
            #     # loss, loss_items = compute_loss(pred, targets.to(device))
            #     pred = student_model(combined_imgs)
            #     loss, loss_items = compute_loss(pred, combined_targets.to(device))
                 
            #################### 原图的监督训练 ####################
            with torch.amp.autocast(device_type='cuda', enabled=check_amp(student_model)):
                pred_orig = student_model(imgs)                           # 用原图
                loss_orig, loss_items_orig = compute_loss(pred_orig, targets.to(device))
                
            #################### 扩散图的监督训练 ###################
            with torch.amp.autocast(device_type='cuda', enabled=check_amp(student_model)):
                pred_aug = student_model(aug_imgs)                         # 用扩散图
                loss_aug, loss_items_aug = compute_loss(pred_aug, aug_targets.to(device))

            #################### 合并损失 ###########################
            loss = loss_orig + loss_aug
            print(f'loss_orig:{loss_orig} ')
            print(f'loss_aug:{loss_aug} ')
            loss_items = loss_items_orig + loss_items_aug  # 两个loss_items都是[box_loss, cls_loss, dfl_loss]之类，直接相加
            print(f'loss_items:{loss_items}')
                # if RANK != -1:
                #     loss *= WORLD_SIZE
                # if opt.quad:
                #     loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(student_model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', student_model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
        # -------------------- 有标注数据训练结束 --------------------

        # -------------------- 2) 训练无标注数据（半监督部分） --------------------
        if RANK in {-1, 0}:
            teacher_model.eval()
            # 让 student 模型是可训练状态
            student_model.train()
            for param in student_model.parameters():
                param.requires_grad = True

            # 这里以 /home/user/ccl/DSYM/data/NEU-DET/test_split/images/ 举例，请根据自己数据修改
            unlabeled_loader = create_dataloader(
                "/home/tinnel/ccl/DSYM/data/NEU-DET/Uns_train_split/images/",  # 未标注数据路径
                imgsz,
                batch_size // WORLD_SIZE,
                gs,
                single_cls,
                hyp=hyp,
                augment=True,
                cache=None if opt.cache == 'val' else opt.cache,
                rect=opt.rect,
                rank=LOCAL_RANK,
                workers=workers,
                image_weights=opt.image_weights,
                close_mosaic=opt.close_mosaic != 0,
                quad=opt.quad,
                prefix=colorstr('unlabeled: '))[0]

            # >>> CLIP: 在此插入 CLIP 相关初始化（只执行一次即可） <<<
            # --------------------------------------------------------------------
            # 如果本地还没有安装 clip 库，请先 pip install git+https://github.com/openai/CLIP.git
            try:
                import clip
                import torchvision.transforms as T
                from PIL import Image
            except ImportError:
                raise ImportError("CLIP package is not installed. Please install via 'pip install git+https://github.com/openai/CLIP.git'")

            # 加载 CLIP 模型（ViT-B/32 仅作示例）
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            clip_model.eval()
            for p in clip_model.parameters():
                p.requires_grad = False

            # 定义文本提示，这里仅示例缺陷类/正常类，也可根据需要添加更多
            text_prompts = ["defect", "normal"]
            text_tokens = clip.tokenize(text_prompts).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 设定一个阈值，若 CLIP 对图像的 "defect" 评分低于此值，则跳过该图
            clip_threshold = 0.3
            # --------------------------------------------------------------------

            # 降低置信度阈值
            confidence_threshold = 0.3

            for i, (imgs_u, _, paths_u, _) in enumerate(unlabeled_loader):
                ni = i + nb * epoch  # 这里可以自行定义或与上面分开计算
                imgs_u = imgs_u.to(device, non_blocking=True).float() / 255

                # >>> CLIP：先对该 batch 的每张图用 CLIP 判断是否可能是缺陷 <<<
                skip_batch = False
                for bi in range(imgs_u.shape[0]):
                    # 转成 PIL
                    pil_image = T.functional.to_pil_image(imgs_u[bi].cpu())
                    # CLIP 预处理
                    clip_input = clip_preprocess(pil_image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_features = clip_model.encode_image(clip_input)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        # 计算和文本的相似度，取 softmax 方便理解
                        logits_per_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)  # shape [1, 2]
                        defect_score = logits_per_image[0, 0].item()  # 第0个是 "defect"
                    if defect_score < clip_threshold:
                        LOGGER.info(f"[CLIP Filter] Image {bi} in batch {i} likely not defect (score={defect_score:.2f}), skip.")
                        skip_batch = True
                        break
                if skip_batch:
                    continue
                # >>> CLIP 过滤结束 <<<

                with torch.no_grad():
                    # 教师模型前向
                    with torch.amp.autocast(device_type='cuda', enabled=check_amp(teacher_model)):
                        teacher_pred = teacher_model(imgs_u)

                for idx_p, p in enumerate(teacher_pred[0]):
                    print(f"teacher_pred[0][{idx_p}] type: {type(p)}, shape: {p.shape if isinstance(p, torch.Tensor) else None}")

                # 1️⃣ 拼接多个尺度的输出
                teacher_pred_logits = torch.cat(teacher_pred[0], dim=2)  # 连接多个特征层

                # 2️⃣ 提取 `objectness` 置信度
                objectness_confidence = teacher_pred_logits[:, 4, :]  # 取出 objectness 分数，形状 [B, N]

                # 3️⃣ 提取最大类别置信度
                class_confidence = teacher_pred_logits[:, 5:, :].sigmoid().max(dim=1)[0]  # 形状 [B, N]

                # 4️⃣ 计算最终置信度（可自定义策略）
                teacher_confidence = 0.5 * objectness_confidence.sigmoid() + 0.5 * class_confidence

                # 打印范围
                print(f"teacher_confidence max: {teacher_confidence.max().item()}, min: {teacher_confidence.min().item()}")

                # 过滤低置信度伪标签
                valid_mask = teacher_confidence > confidence_threshold  # [B, N]

                # **修正索引方式**
                if valid_mask.sum() == 0:
                    print("No high-confidence pseudo labels found, skipping batch.")
                    continue

                print(f"valid_mask shape: {valid_mask.shape}")  # 确保 shape 是 [B, N]

                # 1️⃣ 调整 teacher_pred_logits 形状以匹配索引格式
                teacher_pred_logits = teacher_pred_logits.permute(0, 2, 1)  # [batch, num_anchors, num_classes+5]

                # 2️⃣ 确保 valid_mask 形状正确
                valid_mask = valid_mask.view(teacher_pred_logits.shape[0], -1)  # [batch, num_anchors]

                # 3️⃣ 逐个 batch 进行索引
                filtered_logits = []
                from torch.nn.utils.rnn import pad_sequence
                for ib in range(teacher_pred_logits.shape[0]):
                    if valid_mask[ib].sum() > 0:  # 仅处理存在有效数据的 batch
                        filtered_logits.append(teacher_pred_logits[ib][valid_mask[ib]])  # [N_valid, 10]
                    else:
                        filtered_logits.append(torch.zeros((1, teacher_pred_logits.shape[2]),
                                                   device=teacher_pred_logits.device))  # 避免空张量

                # 4️⃣ 处理 batch 内 anchor 数量不一致的问题
                filtered_logits = pad_sequence(filtered_logits, batch_first=True, padding_value=0.0)  # [batch, N_valid_max, 10]

                # 5️⃣ 变回 [batch, num_classes+5, N_valid_max]
                teacher_pred_logits = filtered_logits.permute(0, 2, 1)
                print(f"✅ Updated teacher_pred_logits shape: {teacher_pred_logits.shape}")  # e.g. [B, 10, N_valid_max]

                # 学生模型前向
                with torch.amp.autocast(device_type='cuda', enabled=check_amp(student_model)):
                    student_pred = student_model(imgs_u)

                # -------------------- 提取特征图并计算一致性损失 --------------------
                teacher_pred_tensor = None
                student_pred_tensor = None

                try:
                    # teacher_pred[1][0] 是一个 list, 包含3个张量 [B,70,H,W] for each scale
                    teacher_features = []
                    for feature in teacher_pred[1][0]:
                        teacher_features.append(feature.view(feature.size(0), feature.size(1), -1))
                    teacher_pred_tensor = torch.cat(teacher_features, dim=2)  # [B, 70, HW_total]

                    # student_pred[0] 同样是 3 个张量 [B,70,H,W], ...
                    student_features = []
                    for feature in student_pred[0]:
                        student_features.append(feature.view(feature.size(0), feature.size(1), -1))
                    student_pred_tensor = torch.cat(student_features, dim=2)

                    assert teacher_pred_tensor.shape == student_pred_tensor.shape, (
                        f"Shape mismatch: teacher_pred_tensor {teacher_pred_tensor.shape} "
                        f"vs student_pred_tensor {student_pred_tensor.shape}"
                    )

                except Exception as e:
                    LOGGER.warning(f"Error extracting features for unlabeled data: {e}")
                    continue

                LOGGER.info(f"Teacher pred tensor shape: {teacher_pred_tensor.shape}, requires_grad: {teacher_pred_tensor.requires_grad}")
                LOGGER.info(f"Student pred tensor shape: {student_pred_tensor.shape}, requires_grad: {student_pred_tensor.requires_grad}")

                # 计算一致性损失
                consistency_loss = consistency_loss_fn(student_pred_tensor, teacher_pred_tensor)

                # 反向传播
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(consistency_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                if ema:
                    ema.update(student_model)

        # -------------------- 同步/更新教师模型 --------------------
        alpha = 0.99
        for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)

        # 调整学习率
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        # 如果是主进程，进行验证和保存模型
        if RANK in {-1, 0}:
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(student_model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop

            # 计算 mAP
            if not noval or final_epoch:
                results, maps, _ = validate.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    half=check_amp(student_model),
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                )

                # 写到指定txt文件，比如 mAP_log.txt
                mAP50 = results[2]      # mAP@0.5
                mAP50_95 = results[3]   # mAP@0.5:0.95
                log_file = save_dir / "mAP_log.txt"
                with open(log_file, 'a') as f:
                        f.write(f"Epoch={epoch}, mAP@0.5={mAP50:.4f}, mAP@0.5:.95={mAP50_95:.4f}\n")

            # 更新 best_fitness
            fi = fitness(np.array(results).reshape(1, -1))
            stop = stopper(epoch=epoch, fitness=fi)
            if fi > best_fitness:
                best_fitness = fi

            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # 保存模型
            if (not nosave) or (final_epoch and not evolve):
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(student_model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,
                    'date': datetime.now().isoformat()
                }
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break

    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss
                    )
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)
        callbacks.run('on_train_end', last, best, epoch, results)

    torch.cuda.empty_cache()
    return results

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='yolo.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-high.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'LION'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--flat-cos-lr', action='store_true', help='flat cosine LR scheduler')
    parser.add_argument('--fixed-lr', action='store_true', help='fixed LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument')
    parser.add_argument('--min-items', type=int, default=0, help='Experimental')
    parser.add_argument('--close-mosaic', type=int, default=0, help='Experimental')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    if known:
        return parser.parse_known_args()[0]
    else:
        return parser.parse_args()

def main(opt, callbacks=Callbacks()):
    if RANK in {-1, 0}:
        print_args(vars(opt))

    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'
        opt_data = opt.data
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        # opt = argparse.Namespace(**d)
        # opt.cfg, opt.weights, opt.resume = '', str(last), True
                # 用 opt.yaml 中的参数更新 opt，但只替换原本是默认值的字段
        for k, v in d.items():
            current_v = getattr(opt, k, None)
            if current_v in [None, '', 0, False] and v not in [None, '', 0, False]:
                setattr(opt, k, v)

        # 强制设置 resume 相关字段
        opt.weights = str(last)
        opt.resume = True
        if is_url(opt_data):
            opt.data = check_file(opt_data)
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLO Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
    else:
        # 超参进化（如不需要可删除）
        pass

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
