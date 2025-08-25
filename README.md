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
    --img_size 224 \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-3 \
    --pretrained \
    --backbone mobilenet_v3 \
    --embedding_size 512 \
    --arcface_weight 1.0 \
    --triplet_weight 0.5 \
    --use_mask
```

### 3. 推理

```bash
python inference.py \
    --seg_model outputs/segmentation/best_model.pth \
    --recog_model outputs/recognition/best_model.pth \
    --person_ids outputs/recognition/person_ids.json \
    --database_path database.pkl \
    --query_image test_image.jpg \
    --top_k 5 \
    --threshold 0.5
```

## 模型架构

### 分割模型
- **Backbone**: MobileNetV3 (或MobileNetV2作为备选)
- **Decoder**: UNet-style decoder with ASPP
- **Attention**: CBAM (Channel and Spatial Attention)
- **Loss**: BCE + Dice + Focal Loss

### 识别模型
- **Backbone**: MobileNetV3
- **Embedding**: 512维特征向量
- **Classification Head**: ArcFace
- **Loss**: ArcFace Loss + Triplet Loss

## 训练策略

### 分割模型训练
1. 独立训练分割模型
2. 使用组合损失函数：BCE + Dice + Focal
3. 保存最佳Dice分数的模型

### 识别模型训练
1. 使用预训练的分割模型生成掩码
2. 对输入图像应用掩码进行预处理
3. 使用ArcFace + Triplet Loss进行训练
4. 保存最佳准确率的模型

## 推理流程

1. **分割阶段**: 使用分割模型生成虹膜掩码
2. **预处理**: 应用掩码裁剪虹膜区域
3. **特征提取**: 使用识别模型提取512维嵌入向量
4. **相似度计算**: 与数据库中的嵌入向量计算余弦相似度
5. **结果排序**: 返回最相似的人员ID

## 数据库构建

数据库包含每个注册用户的平均嵌入向量：
- 遍历所有注册用户图像
- 使用分割+识别模型提取嵌入
- 计算每个用户的平均嵌入向量
- 保存为pickle文件

## 性能指标

### 分割模型
- Dice Score
- IoU Score
- 分割精度

### 识别模型
- 分类准确率
- ArcFace Loss
- Triplet Loss

## 扩展性

该pipeline设计为模块化架构，便于后续改进：
- 支持不同的backbone网络
- 可调整损失函数权重
- 易于集成新的数据增强策略
- 支持联合训练模式

## 注意事项

1. 确保数据格式正确，图像和掩码文件名对应
2. 训练时建议使用GPU加速
3. 可以根据数据量调整batch_size和学习率
4. 推理时需要先构建数据库
5. 相似度阈值可根据实际需求调整

## 许可证

MIT License
