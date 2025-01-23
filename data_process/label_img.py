import torch as t
import numpy as np

# 此文件用于处理和比较大小模型的检测结果
# 主要功能:
# 1. 加载并比较大模型(YOLOv4)和小模型(MobileNet-YOLOv4)的目标检测结果
# 2. 根据检测结果的差异对图像进行标记(标记值为0或1)
# 3. 统计每个模型检测到的目标数量
# 4. 保存处理结果供后续训练使用

# 加载模型检测结果和真实标注数据
data_big=t.load("big_model_results")        # 加载大模型(YOLOv4)的检测结果
data_small=t.load("small_model_results")     # 加载小模型(MobileNet-YOLOv4)的检测结果
image_target_s=t.load("ground_truth_area")   # 加载图像真实标注区域
k_list=t.load("image_file_name")            # 加载图像文件名列表

# 初始化计数器
SUM_big=0      # 大模型检测到的总目标数
SUM_small=0    # 小模型检测到的总目标数

# 初始化结果存储字典
target_num_yolov4={}              # 存储大模型检测到的每张图像的目标数
target_num_mobilev1_yolov4={}     # 存储小模型检测到的每张图像的目标数
image_tag={}                      # 存储图像标签(1表示模型检测结果差异大，0表示差异小)
image_index_n=[]                  # 存储检测结果差异大的图像索引

print(len(k_list))  # 打印处理的图像总数

# 遍历所有图像进行处理
for j in k_list:
    # 处理大模型(YOLOv4)的检测结果
    a_big=t.from_numpy(np.array(data_big[j])[:,0])    # 获取检测置信度
    mask=a_big.ge(0.5000)                             # 创建置信度大于0.5的掩码
    b_big=a_big[mask].numel()                         # 计算高置信度检测框的数量
    
    # 确保检测到的目标数不超过真实标注数量
    if b_big<=len(image_target_s[j]):
        target_num_big=b_big
    else:
        target_num_big=len(image_target_s[j])
    target_num_yolov4[j]=target_num_big
    SUM_big+=target_num_big

    # 处理小模型(MobileNet-YOLOv4)的检测结果
    a_small=t.from_numpy(np.array(data_small[j])[:,0])
    mask=a_small.ge(0.5000)
    b_small=a_small[mask].numel()
    
    # 确保检测到的目标数不超过真实标注数量
    if b_small <= len(image_target_s[j]):
        target_num_small = b_small
    else:
        target_num_small = len(image_target_s[j])
    target_num_small=b_small
    SUM_small+=target_num_small
    target_num_mobilev1_yolov4[j]=target_num_small

    # 计算两个模型检测结果的差异并标记
    miss=target_num_big-target_num_small    # 计算检测目标数量的差异
    miss_thread=1                           # 设置差异阈值
    if miss>=miss_thread:                   # 如果差异超过阈值
        label=1                             # 标记为1（表示需要关注的图像）
        image_tag[j]=label
        image_index_n.append(j)
    else:
        label=0                             # 标记为0（表示检测结果相近的图像）
        image_tag[j]=label

# 保存处理结果
t.save(image_tag,'image_label')                                   # 保存图像标签
t.save(target_num_yolov4,'image_object_num_big_model')           # 保存大模型检测结果
t.save(target_num_mobilev1_yolov4,'image_object_num_samll_model')# 保存小模型检测结果

