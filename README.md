# 虹膜分割与识别 Baseline Pipeline

这是一个完整的虹膜分割和识别baseline pipeline，包含分割模型训练、识别模型训练和推理功能。

## 项目结构

```
iris_recognize/
├── data/                    # 数据目录
│   ├── images/             # 虹膜原始图像
│   └── masks/              # 分割标签掩码
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── segmentation.py     # 分割模型
│   └── recognition.py      # 识别模型
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── dataset.py          # 数据集加载
│   └── losses.py           # 损失函数
├── train_seg.py            # 分割模型训练脚本
├── train_recog.py          # 识别模型训练脚本
├── inference.py            # 推理脚本
├── requirements.txt        # 依赖包
└── README.md              # 项目说明
```

## 数据格式

数据位于 `data/` 文件夹：
- `images/`: 虹膜原始图像
- `masks/`: 对应分割标签（文件名与images对应，区别是文件名后缀多 `_mask`）

文件名规则：
- 前缀表示人员ID
- 中间L/R表示左右眼
- 后缀数字表示该人员该眼的多张照片编号

例如：`S5999L00.jpg` 和 `S5999L00_mask.png`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练分割模型

```bash
python train_seg.py \
    --data_root data \
    --output_dir outputs/segmentation \
    --img_size 256 \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-3 \
    --pretrained \
    --backbone mobilenet_v3
```

### 2. 训练识别模型

```bash
python train_recog.py \
    --data_root data \
    --output_dir outputs/recognition \
    --img_size 256 \
    --batch_size 32 \
    --epochs 100 \
    --lr 5e-4 \
    --pretrained \
    --backbone mobilenet_v3 \
    --embedding_size 512 \
    --arcface_weight 1.0 \
    --triplet_weight 0.5 \
    --use_mask
```

### 3. 训练识别模型

【推荐】使用复值网络（complex_irisnet）+ ArcFace + Triplet，并启用虹膜橡皮片归一化与按眼划分：

```bash
python train_recog.py \
  --data_root data \
  --output_dir outputs/recognition_complex_arcface \
  --backbone complex_irisnet \
  --embedding_size 512 \
  --batch_size 32 \
  --epochs 100 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --triplet_weight 0.5 \
  --triplet_margin 0.3 \
  --arcface_weight 1.0 \
  --use_mask \
  --split_mode eye \
  --normalize_iris \
  --norm_H 64 \
  --norm_W 256
```

- 如遇显存不足，可将 `--batch_size` 降为 16 或 8。
- `--arcface_weight > 0` 时会启用分类分支（ArcFace）；否则仅使用 Triplet。

### 3. 构建数据库

```bash
python build_database.py \
    --data_root data \
    --seg_model outputs/segmentation/best_model.pth \
    --recog_model outputs/recognition_complex_arcface/best_model.pth \
    --output_path database/database.pkl
```

- 默认启用虹膜橡皮片归一化（`--normalize_iris`，尺寸 `64x256`），与训练保持一致；可通过 `--norm_H/--norm_W` 调整。
- 也支持评估目录模式：使用 `--images_dir` 仅给定图像目录（无需 masks）。

### 4. 推理

#### 单张图像查询
```bash
python inference.py \
    --seg_model outputs/segmentation/best_model.pth \
    --recog_model outputs/recognition_complex_arcface/best_model.pth \
    --database_path database/database.pkl \
    --query_image test_image.jpg \
    --top_k 5 \
    --threshold 0.5
```

#### 批量评估
```bash
python inference.py \
    --seg_model outputs/segmentation/best_model.pth \
    --recog_model outputs/recognition_complex_arcface/best_model.pth \
    --database_path database/database.pkl \
    --eval_dir test_images/ \
    --results_csv results.csv \
    --sample_n 100
```

- 推理默认执行虹膜橡皮片归一化（`--normalize_iris`，64x256）；若关闭归一化，则会对原图像应用分割掩码后再提取特征。

### 5. 数据库分析

```bash
python analyze_database.py \
    --database_path database/database.pkl \
    --out_dir outputs/db_analysis \
    --topk 20 \
    --num_thresholds 400
```

- 输出包含：
  - FAR/FRR 随阈值变化的 CSV（含等错误率 EER）
  - ROC 曲线与 DET 曲线（PNG）
  - L2 范数与余弦相似度直方图、Top-K 最混淆对
- 说明：若要绘制 DET 曲线，建议安装 `scipy`（可选依赖）。

## 模型架构

### 分割模型
- **Backbone**: MobileNetV3 (或MobileNetV2作为备选)
- **Decoder**: UNet-style decoder with ASPP
- **Attention**: CBAM (Channel and Spatial Attention)
- **Loss**: BCEWithLogits + Dice + Boundary Loss
- **特性**: 支持多数据集平衡训练、混合精度训练(AMP)、余弦学习率调度

### 识别模型
- **Backbone**: MobileNetV3 或 ComplexIrisNet（复值轻量骨干）
- **Embedding**: 512维特征向量（归一化用于余弦度量）
- **Classification Head**: ArcFace（与 Triplet 可组合）
- **Loss**: ArcFace Loss + Triplet Loss（`arcface_weight` 控制是否启用分类分支）
- **特性**: 支持虹膜橡皮片归一化（64x256）、按眼划分数据集（L: train/val，R: test）

## 训练策略

### 分割模型训练
1. **基础训练**: 使用train_seg.py进行标准训练
2. **改进训练**: 使用seg/main.py支持多数据集平衡训练
3. **损失函数**: BCEWithLogits + Dice + Boundary Loss组合
4. **优化策略**: 余弦学习率调度、混合精度训练(AMP)
5. **评估指标**: Dice Score、IoU Score
6. **模型保存**: 保存最佳Dice分数的模型

### 识别模型训练
1. **数据预处理**: 默认启用虹膜橡皮片归一化（64x256）；若关闭则可选择使用掩码相乘
2. **数据划分**: 支持 `--split_mode eye`（左眼80/20为train/val；右眼为test）或随机划分
3. **损失函数**: ArcFace Loss + Triplet Loss组合（ArcFace 由 `arcface_weight` 控制）
4. **优化器**: AdamW优化器，支持学习率调度
5. **评估指标**: Top-1/Top-5（仅在启用分类分支时计算）
6. **模型保存**: 保存最佳验证指标的模型，支持从检查点恢复

## 推理流程

### 单张图像查询
1. **图像预处理**: 读取图像
2. **分割阶段**: 使用分割模型生成虹膜掩码
3. **归一化**: 执行虹膜橡皮片归一化到 64x256（默认启用）；若关闭则对原图像乘以掩码
4. **特征提取**: 使用识别模型提取512维嵌入向量
5. **相似度计算**: 与数据库中的嵌入向量计算余弦相似度
6. **结果排序**: 返回Top-K最相似的人员ID和相似度分数

### 批量评估
1. **目录扫描**: 扫描指定目录下的所有图像文件
2. **文件名解析**: 从文件名提取人员ID和眼部信息(如S5999L00.jpg)
3. **批量处理**: 对每张图像执行完整的推理流程（含归一化）
4. **性能评估**: 计算Top-1/Top-5准确率和平均相似度
5. **结果保存**: 将结果保存为CSV文件

## 数据库构建

### 标准数据集模式
- 遍历data_root下的images和masks目录
- 使用分割+识别模型提取每张图像的嵌入向量
- 按person_id_eye格式组织(如S5999_L)
- 支持多模板存储或平均嵌入向量

### 评估数据集模式
- 仅需要images目录，无需masks
- 使用分割模型自动生成掩码并执行归一化
- 从文件名解析人员ID和眼部信息
- 保存为pickle格式的字典文件

## 性能指标

### 分割模型
- **Dice Score**: 分割重叠度评估
- **IoU Score**: 交并比评估
- **Boundary Loss**: 边界质量评估
- **训练监控**: TensorBoard日志记录

### 识别模型
- **Top-1/Top-5准确率**: 分类性能评估
- **ArcFace Loss**: 角度边际损失
- **Triplet Loss**: 三元组损失
- **余弦相似度**: 特征向量相似度

### 数据库质量分析
- **嵌入向量统计**: L2范数分布
- **冒充者/真实分布**: 余弦相似度直方图
- **FAR/FRR 与 EER**: 阈值扫描得到的错误率指标（导出CSV与图像）
- **ROC/DET 曲线**: 识别性能曲线（DET 需 `scipy` 支持）
- **混淆对分析**: 最易混淆的人员对

## 注意事项

### 数据准备
1. **标准训练**: 确保data_root下有images/和masks/目录，文件名对应
2. **评估数据**: 仅需images目录，文件名格式如S5999L00.jpg
3. **文件命名**: 前缀为人员ID，L/R表示左右眼，数字为序号

### 训练配置
1. **GPU推荐**: 训练时强烈建议使用GPU加速
2. **内存要求**: 根据数据量调整batch_size，避免内存溢出
3. **学习率**: 分割模型1e-3，识别模型5e-4或3e-4为推荐起始值
4. **混合精度**: 使用--amp参数可显著加速训练

### 推理使用
1. **数据库构建**: 推理前必须先构建数据库
2. **阈值调整**: 相似度阈值可根据实际需求调整(推荐0.5)
3. **批量处理**: 支持目录级别的批量评估
4. **结果分析**: 使用analyze_database.py分析数据库质量

### 模型格式
1. **PyTorch模型**: .pth格式，包含完整训练状态
2. **ONNX模型**: .onnx格式，用于优化推理性能
3. **模型转换**: 训练完成后自动生成ONNX版本

### 依赖环境
1. **CUDA要求**: GPU推理需要CUDA 12.x和cuDNN 9.x
2. **ONNX Runtime**: 根据设备选择CPU或GPU版本
3. **可选**: `scipy`（用于DET曲线坐标变换）
