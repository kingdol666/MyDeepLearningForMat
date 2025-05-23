# 材料带隙预测系统 (Material Band Gap Prediction System)

![材料科学](https://img.shields.io/badge/材料科学-深度学习-blue)
![Python](https://img.shields.io/badge/Python-3.6+-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red)

## 项目简介

材料带隙预测系统是一个基于深度学习的工具，用于预测无机材料的带隙值。带隙是材料电子能带结构中的关键参数，决定了材料的电子性质、光学性质和导电性能。准确预测材料带隙可以加速新材料的发现和设计，在太阳能电池、半导体、光学材料等领域具有重要应用价值。

本系统利用材料的基本特性（如元素组成、形成能等）预测其带隙值，并提供了命令行和Web界面两种交互方式。系统支持单模型预测和集成模型预测，后者通常具有更高的准确性。

## 主要特性

- **高精度预测**：基于优化的深度神经网络，提供精确的带隙预测
- **增强型特征工程**：利用元素特性、电子结构等多维度特征提高预测准确性
- **集成学习**：支持多模型集成，提高预测稳定性和准确性
- **用户友好界面**：提供直观的Web界面和便捷的命令行接口
- **材料分类**：根据预测带隙自动将材料分类为金属/导体、半导体或绝缘体

## 安装说明

### 环境要求

- Python 3.6+
- PyTorch 1.7+
- CUDA (可选，用于GPU加速)

### 安装步骤

1. 克隆代码仓库：
   ```bash
   git clone https://github.com/kingdol666/MyDeepLearningForMat
   ```

2. 安装依赖项：
   ```bash
   pip install -r requirements.txt
   ```

3. 安装PyTorch（根据您的系统和CUDA版本）：
   ```bash
   # CPU版本
   pip install torch torchvision
   
   # GPU版本示例(CUDA 11.3)
   # pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
   ```

## 使用说明

### 模型训练

使用`MaterialDL.py`进行模型训练：

```bash
python app.py
```

该命令会：
1. 检查并获取Materials Project的数据
2. 进行特征工程与数据预处理
3. 训练单个深度学习模型或集成模型
4. 保存训练好的模型和标准化器

### Web界面

启动Web界面进行交互式预测：

```bash
python web-ui.py
```

启动后，在浏览器中访问 http://127.0.0.1:7860 即可使用图形界面进行预测。

## 项目结构

- `MaterialDL.py`: 核心模型定义和训练脚本
- `MaterialPredictor.py`: 预测功能实现和命令行接口
- `web-ui.py`: Web界面实现
- `app.py`: 扩展应用程序接口
- `requirements.txt`: 项目依赖项
- `dataset/`: 数据和模型存储目录
  - `model.pth`: 训练好的模型
  - `scaler.json`: 特征标准化器
  - `ensemble/`: 集成模型目录
- `prediction_results/`: 预测结果存储目录

## 技术实现

本项目利用深度学习技术预测材料带隙，主要包括以下技术点：

### 模型架构

- **ImprovedMaterialsNet**: 优化的深度神经网络模型，包含多分支特征提取、批归一化、残差连接等
- **EnsembleModel**: 集成多个模型的结果，减少单一模型的方差和偏差

### 特征工程

项目使用了多种材料特征，包括：
- 元素电负性、原子半径、价电子数等基本属性
- 元素周期表位置特征（s区、p区、d区、f区元素比例）
- 电子结构特征（轨道比例、金属性比例等）
- 化学键特征（共价性、离子性）

### 预测流程

1. 输入材料的基本信息（元素组成、元素数量、形成能）
2. 进行特征工程，计算材料的特征向量
3. 对特征进行标准化处理
4. 使用训练好的模型进行预测
5. 根据预测带隙对材料进行分类

## 示例

### 预测二氧化硅(SiO₂)的带隙

```bash
python MaterialPredictor.py --elements "Si,O" --nelements 2 --formation_energy -5.23
```

输出示例：
```
预测带隙: 6.2451 eV
材料类型: 🟠 绝缘体 (带隙 > 3.0 eV)
```

### 预测砷化镓(GaAs)的带隙

```bash
python MaterialPredictor.py --elements "Ga,As" --nelements 2 --formation_energy -0.19
```

输出示例：
```
预测带隙: 1.4327 eV
材料类型: 🟢 半导体 (带隙 0.1-3.0 eV)
```

## 开发者信息

- 集成模型训练：使用`MaterialDL.py`中的`--ensemble`参数
- 模型自定义：修改`ImprovedMaterialsNet`类的结构
- 特征工程：修改`prepare_enhanced_features`函数

## 许可证

MIT License

## 参考文献

1. Materials Project (https://materialsproject.org/)
2. "Machine Learning for Materials Science" - Nature Reviews
3. "Deep Learning for Materials Discovery" - Science

## 鸣谢

感谢Materials Project提供的材料数据支持。
