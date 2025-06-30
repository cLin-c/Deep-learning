# utils/diffusion_utils.py

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

def dummy_safety_checker(images, clip_input):
    return images, [False] * len(images)


def init_diffusion_model(
    model_name_or_path="runwayml/stable-diffusion-inpainting",
    device="cuda",
):
    # 如果是 CPU，就用 float32；如果是 GPU (cuda)，就用 float16
    if device == "cpu":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype
    ).to(device)
    
    pipe.safety_checker = dummy_safety_checker
    return pipe


def inpaint_defect(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    pipe: StableDiffusionInpaintPipeline,
    prompt: str = "a realistic defect",
    num_inference_steps=50,
    guidance_scale=7.5
):
    """
    对单张图像进行缺陷区域的 inpaint 生成新图像
    image_tensor: (C, H, W)，归一化到0~1 或者 0~255都行，但要与pipe的预处理对应
    mask_tensor:  (H, W)，0/1
    prompt: 文本提示，用来告诉StableDiffusion你要生成什么“缺陷”

    返回 PIL.Image 或者同尺寸的 torch.Tensor
    """
    # 先转换到 CPU + PIL 以适配 pipeline (也可以自己写更高效的前后处理)
    device = pipe.device
    image_pil = to_pil_image(image_tensor)
    mask_pil  = to_pil_image(mask_tensor)
    
    # 调用 diffusers 的 inpaint
    result = pipe(
        prompt=prompt,
        image=image_pil,
        mask_image=mask_pil,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=640,  # <--- 加上
        width=640    # <--- 加上
    )
    # pipeline 的输出是一个 StableDiffusionPipelineOutput,
    # 其中 result.images 是 list[PIL.Image]
    out_pil = result.images[0]
    return out_pil

class_id_to_prompt = {
    0: "a metal surface with crazing cracks",
    1: "a metal surface with inclusion defects",
    2: "a metal surface with patch defects",
    3: "a metal surface with pitted surface",
    4: "a metal surface with rolled-in scale",
    5: "a metal surface with scratches",
}


def generate_aug_images(
    imgs: torch.Tensor,
    targets: torch.Tensor,
    pipe: StableDiffusionInpaintPipeline,
    device="cuda"
):
    """
    针对一个 batch 的图像 + 标注批量生成合成图像
    假设 targets 的格式 [img_idx, class, x, y, w, h] (YOLO归一化),
    你也可能用 [class, x, y, w, h] + 自己管理 batch 关系
    
    返回 (aug_imgs, aug_targets), 让 train_semi.py 拼回原 batch
    """

    # 准备列表用于存储合成后的图像 & 新的标注
    aug_imgs_list = []
    aug_targets_list = []

    # 获取 batch 内图像数
    batch_size = imgs.shape[0]
    _, _, h, w = imgs.shape

    for i in range(batch_size):
        # 当前图像张量
        img_i = imgs[i]  # shape [3, H, W]
        # 提取与 i 对应的标注
        # 这里假设 targets[:,0] 表示第几个 batch index
        t_indices = (targets[:, 0] == i)
        t = targets[t_indices]  # shape [N, 6], N是该图像的框数
        if len(t) == 0:
            # 没有缺陷标注，就跳过或者直接原图放入
            
            aug_imgs_list.append(img_i)
            aug_targets_list.append(t)
            continue

        # 根据标注构造 mask， ， ，考虑了离开麦克马克面积名面积昆明木刻版画国宾馆viu
        mask = torch.zeros((h, w), dtype=torch.float32)
        # YOLO 标注是 xywh(归一化), 先转换到像素坐标
        for box in t:
            # box: [img_idx, class, x, y, w, h]
            x_center = int(box[2] * w)
            y_center = int(box[3] * h)
            bw = int(box[4] * w)
            bh = int(box[5] * h)
            # 计算左上右下
            x1 = max(0, x_center - bw // 2)
            y1 = max(0, y_center - bh // 2)
            x2 = min(w, x_center + bw // 2)
            y2 = min(h, y_center + bh // 2)
            mask[y1:y2, x1:x2] = 1.0

        # 将原图的 [0,1] or [0,255] 转换到 pipe 默认处理范围
        # 如果 train_semi.py 里 imgs 是 0~1 float
        # 我们需要先转到 [0,255] 再变成 PIL
        to_send = img_i.clone()
        if to_send.max() <= 1.0:
            to_send = to_send * 255.0

        # # 调用 inpaint 函数
        # out_pil = inpaint_defect(
        #     to_send,
        #     mask,
        #     pipe,
        #     prompt="a realistic defect"
        # )
        # 获取当前图像中的第一个类别（如果多个类别你也可以选择处理多次或按比例采样）
        class_id = int(t[0, 1].item())
        prompt = class_id_to_prompt.get(class_id, "a realistic defect")

        out_pil = inpaint_defect(
            to_send,
            mask,
            pipe,
            prompt=prompt
        )

        # 再把 PIL 转成 torch.Tensor
        out_tensor = pil_to_tensor(out_pil).float()  # shape [3, H, W]
        # 归一化回 [0,1]
        out_tensor /= 255.0
        out_tensor = out_tensor.to(device)
        # aug_imgs_list.append(out_tensor)
        
        # 这里也可以对 targets 做一些随机扰动，例如随机平移等
        # 暂时不做

        aug_imgs_list.append(out_tensor)
        aug_targets_list.append(t)

    # 拼接新的 batch
    # aug_imgs = torch.stack(aug_imgs_list, dim=0).to(device)
    aug_imgs = torch.stack(aug_imgs_list, dim=0)
    # targets 要把原来的 batch_idx 重置，例如加上某个 offset，
    # 或者保持不变，然后外部再处理
    aug_targets = torch.cat(aug_targets_list, dim=0).to(device) if len(aug_targets_list) else targets
    return aug_imgs, aug_targets


# ----------------- 工具函数 ------------------
def to_pil_image(tensor: torch.Tensor):
    """
    将 tensor 转成 PIL.Image
    tensor 可能是:
      - shape [3,H,W] (RGB图像)
      - shape [H,W] (mask)
    """
    tensor = tensor.detach().cpu().clamp(0, 255).to(torch.uint8)
    if tensor.ndim == 2:
        # 说明它是 [H, W]，灰度图
        return Image.fromarray(tensor.numpy())  # 直接用灰度模式
    elif tensor.ndim == 3:
        # 说明它是 [C, H, W]，彩色图
        return Image.fromarray(tensor.permute(1,2,0).numpy())
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
def pil_to_tensor(img_pil: Image.Image):
    """
    PIL.Image -> [3,H,W] 的 torch.Tensor (0~255)
    """
    arr = np.array(img_pil, dtype=np.uint8)
    if len(arr.shape) == 2:  # 灰度图
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]
    return arr



# ================== 如果想单独测试，就在此加一个 main 函数 ==================
if __name__ == "__main__":
    import sys
    print("Python exec:", sys.executable)
    print("sys.path:", sys.path)

    # 测试初始化
    try:
        pipe = init_diffusion_model(device="cpu")  # 或 "cuda"
        print("StableDiffusionInpaintPipeline 初始化成功！")
    except Exception as e:
        print(f"初始化失败: {e}")
        sys.exit(1)

    # 简单做一个 inpaint 测试
    # 构造一张随机图 + 随机 mask
    c, h, w = 3, 256, 256
    dummy_img = torch.rand(c, h, w) * 255.0  # [0,255]
    dummy_mask = torch.zeros(h, w)
    # 在中间画一个 64x64 的“缺陷区”
    dummy_mask[96:160, 96:160] = 1.0

    try:
        out_pil = inpaint_defect(
            dummy_img,
            dummy_mask,
            pipe,
            prompt="a small square metal defect"
        )
        out_pil.save("test_inpaint_output.png")
        print("Inpaint 测试完成，已保存到 test_inpaint_output.png")
    except Exception as e:
        print(f"推理失败: {e}")