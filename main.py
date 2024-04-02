import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as F
from models.experimental import attempt_load
from utils.general import non_max_suppression

# 设置本地 YOLOv5 模型的路径
model_path = 'yolov5s.pt'

# 设置图像文件夹路径
image_folder = 'attachment1'

# 加载本地模型
model = attempt_load(model_path)

# 将模型设置为评估模式
model.eval()

# 加载图像文件夹中的所有图像
image_paths = list(Path(image_folder).glob('*'))
images = [Image.open(img_path) for img_path in image_paths]

# 运行模型进行预测
results = []
for i, image in enumerate(images):
    # 对图像进行预处理
    img_tensor = F.to_tensor(image).unsqueeze(0)

    # 进行预测
    with torch.no_grad():
        detections = model(img_tensor)[0]

    # 进行非极大值抑制
    detections = non_max_suppression(detections, conf_thres=0.5, iou_thres=0.5)

    # 将结果添加到列表中
    results.append(detections)

# 打印每张图像中检测到的苹果个数
for i, detections in enumerate(results):
    if detections is not None and len(detections) > 0:
        apple_count = len(detections)
        print(f"图像 {i+1}: 检测到 {apple_count} 个苹果")
    else:
        print(f"图像 {i+1}: 没有检测到苹果")

# 你也可以将检测结果保存为图片或者可视化检测结果，根据需要进行操作
