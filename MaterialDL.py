import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import json
import re
from mp_api.client import MPRester
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch.nn.functional as F

# 设置 Materials Project API key
API_KEY = "aPl7jglRbeUu4oHRPdHtwnsA1F16Wbou"  # 使用您的有效API key

# 检测并设置设备


def get_device():
    """检测是否有可用的GPU并返回相应设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        print("未检测到GPU，使用CPU")
        return torch.device("cpu")


# 定义数据集路径
DATASET_DIR = "dataset"
DATASET_FILE = os.path.join(DATASET_DIR, "materials_data.csv")
SCALER_FILE = os.path.join(DATASET_DIR, "scaler.json")
MODEL_FILE = os.path.join(DATASET_DIR, "model.pth")

# 定义元素周期表属性
# 元素的电负性数据
ELECTRONEGATIVITY = {
    'H': 2.20, 'He': 0.00, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55,
    'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': 0.00, 'Na': 0.93, 'Mg': 1.31,
    'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': 0.00,
    'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66,
    'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
    'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 0.00,
    'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.60, 'Mo': 2.16,
    'Tc': 2.10, 'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69,
    'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.10, 'I': 2.66, 'Xe': 0.00,
    'Cs': 0.79, 'Ba': 0.89, 'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14,
    'Pm': 1.13, 'Sm': 1.17, 'Eu': 1.20, 'Gd': 1.20, 'Tb': 1.10, 'Dy': 1.22,
    'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.10, 'Lu': 1.27, 'Hf': 1.30,
    'Ta': 1.50, 'W': 2.36, 'Re': 1.90, 'Os': 2.20, 'Ir': 2.20, 'Pt': 2.28,
    'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 1.87, 'Bi': 2.02, 'Po': 2.00,
    'At': 2.20, 'Rn': 0.00, 'Fr': 0.70, 'Ra': 0.90, 'Ac': 1.10, 'Th': 1.30,
    'Pa': 1.50, 'U': 1.38, 'Np': 1.36, 'Pu': 1.28, 'Am': 1.30, 'Cm': 1.30,
    'Bk': 1.30, 'Cf': 1.30, 'Es': 1.30, 'Fm': 1.30, 'Md': 1.30, 'No': 1.30,
    'Lr': 1.30, 'Rf': 1.30, 'Db': 1.30, 'Sg': 1.30, 'Bh': 1.30, 'Hs': 1.30,
    'Mt': 1.30, 'Ds': 1.30, 'Rg': 1.30, 'Cn': 1.30, 'Nh': 1.30, 'Fl': 1.30,
    'Mc': 1.30, 'Lv': 1.30, 'Ts': 1.30, 'Og': 0.00
}

# 原子半径数据 (单位：埃)
ATOMIC_RADIUS = {
    'H': 0.25, 'He': 0.31, 'Li': 1.45, 'Be': 1.05, 'B': 0.85, 'C': 0.70,
    'N': 0.65, 'O': 0.60, 'F': 0.50, 'Ne': 0.38, 'Na': 1.80, 'Mg': 1.50,
    'Al': 1.25, 'Si': 1.10, 'P': 1.00, 'S': 1.00, 'Cl': 1.00, 'Ar': 0.71,
    'K': 2.20, 'Ca': 1.80, 'Sc': 1.60, 'Ti': 1.40, 'V': 1.35, 'Cr': 1.40,
    'Mn': 1.40, 'Fe': 1.40, 'Co': 1.35, 'Ni': 1.35, 'Cu': 1.35, 'Zn': 1.35,
    'Ga': 1.30, 'Ge': 1.25, 'As': 1.15, 'Se': 1.15, 'Br': 1.15, 'Kr': 0.88,
    'Rb': 2.35, 'Sr': 2.00, 'Y': 1.80, 'Zr': 1.55, 'Nb': 1.45, 'Mo': 1.45,
    'Tc': 1.35, 'Ru': 1.30, 'Rh': 1.35, 'Pd': 1.40, 'Ag': 1.60, 'Cd': 1.55,
    'In': 1.55, 'Sn': 1.45, 'Sb': 1.45, 'Te': 1.40, 'I': 1.40, 'Xe': 1.08,
    'Cs': 2.60, 'Ba': 2.15, 'La': 1.95, 'Ce': 1.85, 'Pr': 1.85, 'Nd': 1.85,
    'Pm': 1.85, 'Sm': 1.85, 'Eu': 1.85, 'Gd': 1.80, 'Tb': 1.75, 'Dy': 1.75,
    'Ho': 1.75, 'Er': 1.75, 'Tm': 1.75, 'Yb': 1.75, 'Lu': 1.75, 'Hf': 1.55,
    'Ta': 1.45, 'W': 1.35, 'Re': 1.35, 'Os': 1.30, 'Ir': 1.35, 'Pt': 1.35,
    'Au': 1.35, 'Hg': 1.50, 'Tl': 1.90, 'Pb': 1.80, 'Bi': 1.60, 'Po': 1.90,
    'At': 1.50, 'Rn': 1.20, 'Fr': 2.60, 'Ra': 2.15, 'Ac': 1.95, 'Th': 1.80,
    'Pa': 1.80, 'U': 1.75, 'Np': 1.75, 'Pu': 1.75, 'Am': 1.75, 'Cm': 1.75,
    'Bk': 1.75, 'Cf': 1.75, 'Es': 1.75, 'Fm': 1.75, 'Md': 1.75, 'No': 1.75,
    'Lr': 1.75, 'Rf': 1.60, 'Db': 1.45, 'Sg': 1.35, 'Bh': 1.35, 'Hs': 1.30,
    'Mt': 1.35, 'Ds': 1.35, 'Rg': 1.35, 'Cn': 1.50, 'Nh': 1.55, 'Fl': 1.45,
    'Mc': 1.45, 'Lv': 1.45, 'Ts': 1.45, 'Og': 1.20
}

# 电离能数据 (单位：eV)
IONIZATION_ENERGY = {
    'H': 13.598, 'He': 24.587, 'Li': 5.392, 'Be': 9.323, 'B': 8.298, 'C': 11.260,
    'N': 14.534, 'O': 13.618, 'F': 17.423, 'Ne': 21.565, 'Na': 5.139, 'Mg': 7.646,
    'Al': 5.986, 'Si': 8.152, 'P': 10.487, 'S': 10.360, 'Cl': 12.968, 'Ar': 15.760,
    'K': 4.341, 'Ca': 6.113, 'Sc': 6.561, 'Ti': 6.828, 'V': 6.746, 'Cr': 6.767,
    'Mn': 7.434, 'Fe': 7.902, 'Co': 7.881, 'Ni': 7.640, 'Cu': 7.726, 'Zn': 9.394,
    'Ga': 5.999, 'Ge': 7.899, 'As': 9.789, 'Se': 9.752, 'Br': 11.814, 'Kr': 14.000,
    'Rb': 4.177, 'Sr': 5.695, 'Y': 6.217, 'Zr': 6.634, 'Nb': 6.759, 'Mo': 7.092,
    'Tc': 7.280, 'Ru': 7.360, 'Rh': 7.459, 'Pd': 8.337, 'Ag': 7.576, 'Cd': 8.994,
    'In': 5.786, 'Sn': 7.344, 'Sb': 8.641, 'Te': 9.010, 'I': 10.451, 'Xe': 12.130,
    'Cs': 3.894, 'Ba': 5.212, 'La': 5.577, 'Ce': 5.539, 'Pr': 5.473, 'Nd': 5.525,
    'Pm': 5.582, 'Sm': 5.644, 'Eu': 5.670, 'Gd': 6.150, 'Tb': 5.864, 'Dy': 5.939,
    'Ho': 6.021, 'Er': 6.108, 'Tm': 6.184, 'Yb': 6.254, 'Lu': 5.426, 'Hf': 6.825,
    'Ta': 7.550, 'W': 7.864, 'Re': 7.833, 'Os': 8.438, 'Ir': 8.967, 'Pt': 8.959,
    'Au': 9.226, 'Hg': 10.437, 'Tl': 6.108, 'Pb': 7.417, 'Bi': 7.286, 'Po': 8.414,
    'At': 9.318, 'Rn': 10.748, 'Fr': 4.073, 'Ra': 5.279, 'Ac': 5.170, 'Th': 6.307,
    'Pa': 5.890, 'U': 6.194, 'Np': 6.266, 'Pu': 6.026, 'Am': 5.974, 'Cm': 5.991,
    'Bk': 6.198, 'Cf': 6.282, 'Es': 6.370, 'Fm': 6.500, 'Md': 6.580, 'No': 6.650,
    'Lr': 4.900, 'Rf': 6.000, 'Db': 6.800, 'Sg': 7.800, 'Bh': 7.600, 'Hs': 7.600,
    'Mt': 8.000, 'Ds': 8.000, 'Rg': 8.000, 'Cn': 8.000, 'Nh': 6.000, 'Fl': 7.000,
    'Mc': 7.000, 'Lv': 7.000, 'Ts': 7.000, 'Og': 7.000
}

# 电子亲和能数据 (单位：eV)
ELECTRON_AFFINITY = {
    'H': 0.754, 'He': 0.000, 'Li': 0.618, 'Be': 0.000, 'B': 0.277, 'C': 1.263,
    'N': -0.070, 'O': 1.461, 'F': 3.339, 'Ne': 0.000, 'Na': 0.548, 'Mg': 0.000,
    'Al': 0.441, 'Si': 1.385, 'P': 0.747, 'S': 2.077, 'Cl': 3.617, 'Ar': 0.000,
    'K': 0.501, 'Ca': 0.025, 'Sc': 0.188, 'Ti': 0.084, 'V': 0.526, 'Cr': 0.666,
    'Mn': 0.000, 'Fe': 0.151, 'Co': 0.662, 'Ni': 1.156, 'Cu': 1.228, 'Zn': 0.000,
    'Ga': 0.430, 'Ge': 1.232, 'As': 0.814, 'Se': 2.021, 'Br': 3.365, 'Kr': 0.000,
    'Rb': 0.486, 'Sr': 0.048, 'Y': 0.307, 'Zr': 0.426, 'Nb': 0.893, 'Mo': 0.746,
    'Tc': 0.550, 'Ru': 1.050, 'Rh': 1.137, 'Pd': 0.562, 'Ag': 1.302, 'Cd': 0.000,
    'In': 0.404, 'Sn': 1.112, 'Sb': 1.047, 'Te': 1.971, 'I': 3.059, 'Xe': 0.000,
    'Cs': 0.472, 'Ba': 0.145, 'La': 0.470, 'Ce': 0.500, 'Pr': 0.500, 'Nd': 0.500,
    'Pm': 0.500, 'Sm': 0.500, 'Eu': 0.500, 'Gd': 0.500, 'Tb': 0.500, 'Dy': 0.500,
    'Ho': 0.500, 'Er': 0.500, 'Tm': 0.500, 'Yb': 0.500, 'Lu': 0.500, 'Hf': 0.000,
    'Ta': 0.322, 'W': 0.816, 'Re': 0.150, 'Os': 1.100, 'Ir': 1.565, 'Pt': 2.128,
    'Au': 2.309, 'Hg': 0.000, 'Tl': 0.400, 'Pb': 0.364, 'Bi': 0.946, 'Po': 1.900,
    'At': 2.800, 'Rn': 0.000, 'Fr': 0.400, 'Ra': 0.100, 'Ac': 0.400, 'Th': 0.600,
    'Pa': 0.500, 'U': 0.500, 'Np': 0.500, 'Pu': 0.500, 'Am': 0.500, 'Cm': 0.500,
    'Bk': 0.500, 'Cf': 0.500, 'Es': 0.500, 'Fm': 0.500, 'Md': 0.500, 'No': 0.500,
    'Lr': 0.500, 'Rf': 0.000, 'Db': 0.000, 'Sg': 0.000, 'Bh': 0.000, 'Hs': 0.000,
    'Mt': 0.000, 'Ds': 0.000, 'Rg': 0.000, 'Cn': 0.000, 'Nh': 0.000, 'Fl': 0.000,
    'Mc': 0.000, 'Lv': 0.000, 'Ts': 0.000, 'Og': 0.000
}

# 价电子数
VALENCE_ELECTRONS = {
    'H': 1, 'He': 2, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
    'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8,
    'K': 1, 'Ca': 2, 'Sc': 3, 'Ti': 4, 'V': 5, 'Cr': 6, 'Mn': 7, 'Fe': 8, 'Co': 9,
    'Ni': 10, 'Cu': 11, 'Zn': 12, 'Ga': 3, 'Ge': 4, 'As': 5, 'Se': 6, 'Br': 7, 'Kr': 8,
    'Rb': 1, 'Sr': 2, 'Y': 3, 'Zr': 4, 'Nb': 5, 'Mo': 6, 'Tc': 7, 'Ru': 8, 'Rh': 9,
    'Pd': 10, 'Ag': 11, 'Cd': 12, 'In': 3, 'Sn': 4, 'Sb': 5, 'Te': 6, 'I': 7, 'Xe': 8,
    'Cs': 1, 'Ba': 2, 'La': 3, 'Ce': 4, 'Pr': 5, 'Nd': 6, 'Pm': 7, 'Sm': 8, 'Eu': 9,
    'Gd': 10, 'Tb': 11, 'Dy': 12, 'Ho': 13, 'Er': 14, 'Tm': 15, 'Yb': 16, 'Lu': 3,
    'Hf': 4, 'Ta': 5, 'W': 6, 'Re': 7, 'Os': 8, 'Ir': 9, 'Pt': 10, 'Au': 11, 'Hg': 12,
    'Tl': 3, 'Pb': 4, 'Bi': 5, 'Po': 6, 'At': 7, 'Rn': 8, 'Fr': 1, 'Ra': 2, 'Ac': 3,
    'Th': 4, 'Pa': 5, 'U': 6, 'Np': 7, 'Pu': 8, 'Am': 9, 'Cm': 10, 'Bk': 11, 'Cf': 12,
    'Es': 13, 'Fm': 14, 'Md': 15, 'No': 16, 'Lr': 3, 'Rf': 4, 'Db': 5, 'Sg': 6, 'Bh': 7,
    'Hs': 8, 'Mt': 9, 'Ds': 10, 'Rg': 11, 'Cn': 12, 'Nh': 3, 'Fl': 4, 'Mc': 5, 'Lv': 6,
    'Ts': 7, 'Og': 8
}

# 元素周期表轨道类型分类
S_BLOCK_ELEMENTS = set(['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',
                       'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'])

P_BLOCK_ELEMENTS = set(['B', 'C', 'N', 'O', 'F', 'Ne',
                       'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                        'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'])

D_BLOCK_ELEMENTS = set(['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                       'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'])

F_BLOCK_ELEMENTS = set(['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                       'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'])

# 金属元素集合
METAL_ELEMENTS = set(['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
                     'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                      'In', 'Sn', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
                      'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
                      'Bi', 'Po', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es',
                      'Fm', 'Md', 'No', 'Lr'])

# 定义神经网络模型


class ImprovedMaterialsNet(nn.Module):
    """增强的材料带隙预测神经网络模型

    特点:
    - 多分支特征提取
    - 深层网络架构
    - 批归一化层
    - 残差连接
    - 改进的激活函数和Dropout
    """

    def __init__(self, input_size):
        super(ImprovedMaterialsNet, self).__init__()

        # 主干网络
        self.backbone = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),  # 将斜率从0.2调整到0.1，更平滑的梯度传播
            nn.Dropout(0.25),   # 轻微降低dropout率，从0.3到0.25

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
        )

        # 多分支特征提取
        self.branch1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )

        self.branch2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.Linear(32, 1)
        )

        # 残差连接
        self.residual = nn.Linear(input_size, 64)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用He初始化而非默认初始化
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入批归一化
        x_backbone = self.backbone(x)

        # 多分支特征提取
        x1 = self.branch1(x_backbone)
        x2 = self.branch2(x_backbone)

        # 特征融合
        x_concat = torch.cat((x1, x2), dim=1)

        # 残差连接
        x_res = self.residual(x)
        x_fusion = self.fusion(x_concat) + x_res[:, :1]

        # 确保输出非负值 - 带隙不应为负
        output = torch.relu(x_fusion)

        return output


class EnsembleModel:
    """材料带隙预测集成模型，结合多个模型提高预测精度"""

    def __init__(self, input_size, n_models=6):  # 从5增加到6
        """初始化集成模型

        参数:
            input_size: 输入特征维度
            n_models: 集成的模型数量
        """
        self.models = []
        self.input_size = input_size
        self.n_models = n_models

        # 创建多个不同初始化的模型
        for i in range(n_models):
            model = ImprovedMaterialsNet(input_size)
            self.models.append(model)

    def train(self, train_loader, val_X, val_y, criterion, epochs=220):  # 增加训练轮数
        """训练集成模型中的每个子模型"""
        print(f"开始训练集成模型 (共{self.n_models}个子模型)...")

        # 获取设备(GPU或CPU)
        device = get_device()

        # 确保验证数据在正确的设备上
        val_X = val_X.to(device)
        val_y = val_y.to(device)

        for i, model in enumerate(self.models):
            print(f"\n训练子模型 {i+1}/{self.n_models}")

            # 将模型移动到GPU
            model = model.to(device)
            self.models[i] = model

            # 为每个模型使用不同的随机种子
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)

            # 为每个模型使用单独的优化器
            optimizer = optim.AdamW(
                model.parameters(),
                lr=0.0015,  # 微调学习率
                weight_decay=3e-5  # 调整权重衰减
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=15)

            # 训练循环
            best_val_loss = float('inf')
            best_model_state = None
            patience_counter = 0

            for epoch in range(epochs):
                # 训练模式
                model.train()
                train_loss = 0.0

                # 批量训练
                for batch_X, batch_y in train_loader:
                    # 确保数据在正确的设备上
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item() * batch_X.size(0)

                train_loss /= len(train_loader.dataset)

                # 评估模式
                model.eval()
                with torch.no_grad():
                    val_outputs = model(val_X)
                    val_loss = criterion(val_outputs, val_y).item()

                    # 更新学习率
                    scheduler.step(val_loss)

                    # 保存最佳模型
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict().copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                # 早停
                if patience_counter >= 25:
                    print(f"验证损失连续25轮未改善，停止子模型训练")
                    break

                # 每20轮输出一次进度
                if (epoch + 1) % 20 == 0:
                    print(
                        f'Epoch [{epoch+1}/{epochs}], 损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}')

            # 加载最佳模型参数
            if best_model_state:
                model.load_state_dict(best_model_state)

            print(f"子模型 {i+1} 训练完成，最佳验证损失: {best_val_loss:.6f}")

    def predict(self, X):
        """使用集成模型进行预测

        参数:
        - X: 输入特征张量

        返回:
        - 预测值张量
        """
        # 获取当前X所在的设备
        device = X.device

        # 确保所有模型都在正确的设备上
        models = [model.to(device) for model in self.models]

        # 进入评估模式
        for model in models:
            model.eval()

        # 收集所有模型的预测结果
        with torch.no_grad():
            predictions = []
            for model in models:
                pred = model(X)
                predictions.append(pred)

            # 计算平均预测值
            ensemble_pred = torch.mean(torch.stack(predictions), dim=0)

        return ensemble_pred

    def save(self, directory):
        """保存集成模型的所有子模型

        参数:
        - directory: 保存模型的目录
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 保存前将所有模型移动到CPU
        for i, model in enumerate(self.models):
            # 将模型移到CPU
            if next(model.parameters()).is_cuda:
                model = model.cpu()
                self.models[i] = model

            # 保存模型
            model_path = os.path.join(directory, f"model_{i}.pth")
            torch.save(model.state_dict(), model_path)

        # 保存元数据
        meta_path = os.path.join(directory, "ensemble_meta.json")
        with open(meta_path, 'w') as f:
            json.dump({
                'n_models': self.n_models,
                'input_size': self.input_size
            }, f)

        print(f"集成模型已保存到目录: {directory}")

    def load(self, directory, input_size):
        """加载集成模型"""
        # 加载元数据
        meta_path = os.path.join(directory, "ensemble_meta.json")
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            self.n_models = meta['n_models']
            self.input_size = meta['input_size']

            # 加载各子模型
            self.models = []
            for i in range(self.n_models):
                model_path = os.path.join(directory, f"model_{i}.pth")
                model = ImprovedMaterialsNet(self.input_size)
                model.load_state_dict(torch.load(
                    model_path, weights_only=True))
                model.eval()  # 设置为评估模式
                self.models.append(model)

            print(f"已加载包含{self.n_models}个子模型的集成模型")
            return True
        except Exception as e:
            print(f"加载集成模型时出错: {str(e)}")
            return False


# 数据获取和预处理函数


def get_materials_data():
    """从Materials Project获取材料数据"""
    try:
        # 检查数据目录是否存在
        if not os.path.exists(DATASET_DIR):
            os.makedirs(DATASET_DIR)
            print(f"创建数据集目录: {DATASET_DIR}")

        # 使用API获取数据
        with MPRester(API_KEY) as mpr:
            print("开始查询Materials Project API...")
            print("查询条件: 查找带隙和形成能数据")

            # 查询材料数据，获取关键属性
            docs = mpr.materials.summary.search(
                fields=["material_id", "band_gap", "formation_energy_per_atom",
                        "elements", "nelements", "formula_pretty"]
            )

            print(f"API查询完成，获取到 {len(docs)} 条记录，开始处理数据...")

            # 转换为DataFrame
            data = []
            for doc in docs:
                try:
                    # 检查必要的属性是否存在
                    if not hasattr(doc, 'band_gap') or not hasattr(doc, 'formation_energy_per_atom'):
                        continue

                    # 处理元素数据
                    elements_str = ""
                    if hasattr(doc, 'elements') and doc.elements:
                        try:
                            # 将每个Element对象转换为其符号(字符串)
                            element_symbols = [str(elem)
                                               for elem in doc.elements]
                            elements_str = ','.join(element_symbols)
                        except Exception as e:
                            print(f"转换元素数据时出错: {str(e)}")
                            elements_str = ""

                    data.append({
                        'band_gap': doc.band_gap,
                        'formation_energy_per_atom': doc.formation_energy_per_atom,
                        'nelements': doc.nelements,
                        'elements': elements_str,
                        'formula': doc.formula_pretty if hasattr(doc, 'formula_pretty') else '',
                        'material_id': doc.material_id if hasattr(doc, 'material_id') else ''
                    })
                except AttributeError as e:
                    print(f"处理数据时遇到属性错误: {str(e)}")
                    continue
                except Exception as e:
                    print(f"处理数据时遇到未知错误: {str(e)}")
                    continue

            df = pd.DataFrame(data)
            print(f"获取到原始数据 {len(df)} 条")

            if len(df) == 0:
                # 如果第一次查询没有结果，尝试不同的查询参数
                print("第一次查询未返回数据，尝试不同的查询参数...")

                docs = mpr.materials.summary.search(
                    nelements=[2, 3],  # 只检索二元和三元化合物
                    fields=["material_id", "band_gap", "formation_energy_per_atom",
                            "elements", "nelements", "formula_pretty"]
                )

                print(f"第二次API查询完成，获取到 {len(docs)} 条记录，开始处理数据...")

                # 重复处理数据的过程
                data = []
                for doc in docs:
                    try:
                        if not hasattr(doc, 'band_gap') or not hasattr(doc, 'formation_energy_per_atom'):
                            continue

                        # 处理元素数据
                        elements_str = ""
                        if hasattr(doc, 'elements') and doc.elements:
                            try:
                                element_symbols = [str(elem)
                                                   for elem in doc.elements]
                                elements_str = ','.join(element_symbols)
                            except Exception as e:
                                print(f"转换元素数据时出错: {str(e)}")
                                elements_str = ""

                        data.append({
                            'band_gap': doc.band_gap,
                            'formation_energy_per_atom': doc.formation_energy_per_atom,
                            'nelements': doc.nelements,
                            'elements': elements_str,
                            'formula': doc.formula_pretty if hasattr(doc, 'formula_pretty') else '',
                            'material_id': doc.material_id if hasattr(doc, 'material_id') else ''
                        })
                    except Exception as e:
                        print(f"处理数据时出错: {str(e)}")
                        continue

                df = pd.DataFrame(data)
                print(f"第二次查询获取到数据 {len(df)} 条")

            # 清理数据
            if len(df) > 0:
                # 删除缺失值
                df = df.dropna(
                    subset=['band_gap', 'formation_energy_per_atom'])

                # 检查elements列是否存在
                if 'elements' not in df.columns:
                    print("警告：数据中没有元素信息列")
                else:
                    # 检查elements列中是否有缺失值
                    missing_elements = df['elements'].isna().sum()
                    if missing_elements > 0:
                        print(f"警告：{missing_elements}条记录缺少元素信息")

                # 保存到CSV文件
                df.to_csv(DATASET_FILE, index=False)
                print(f"数据已保存到 {DATASET_FILE}，共 {len(df)} 条记录")
            else:
                print("未能获取到有效数据")

            return df

    except Exception as e:
        print(f"获取材料数据时发生错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return pd.DataFrame()


def save_scaler(scaler, file_path, feature_info=None):
    """保存StandardScaler参数到文件

    参数:
    - scaler: 训练好的StandardScaler对象
    - file_path: 保存路径
    - feature_info: 特征相关信息，如特征名称列表
    """
    data = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'var': scaler.var_.tolist()
    }

    # 如果提供了额外的特征信息，添加到保存数据中
    if feature_info:
        data.update(feature_info)

    with open(file_path, 'w') as f:
        json.dump(data, f)


def load_scaler(file_path):
    """从文件加载StandardScaler参数

    参数:
    - file_path: 保存路径

    返回:
    - scaler: 重建的StandardScaler对象
    - feature_names: 特征名称列表（如果存在）
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    scaler = StandardScaler()
    scaler.mean_ = np.array(data['mean'])
    scaler.scale_ = np.array(data['scale'])
    scaler.var_ = np.array(data['var'])

    # 提取特征名称（如果存在）
    feature_names = data.get('feature_names', [])

    return scaler, feature_names

# 特征工程函数


def calculate_avg_property(elements_str, property_dict):
    """计算材料中元素的平均属性值

    参数:
    - elements_str: 元素符号的字符串，以逗号分隔，如 'Si,O'
    - property_dict: 元素属性字典，如 ELECTRONEGATIVITY

    返回:
    - 平均属性值，如果无法计算则返回None
    """
    if not elements_str or pd.isna(elements_str):
        return None

    try:
        # 分割元素字符串并移除空白符
        elements = [elem.strip()
                    for elem in elements_str.split(',') if elem.strip()]
        if not elements:
            return None

        values = []
        for element in elements:
            if element in property_dict:
                values.append(property_dict[element])
            else:
                print(f"警告: 元素 '{element}' 在属性字典中不存在")

        if not values:
            return None

        return sum(values) / len(values)
    except Exception as e:
        print(f"计算平均属性时出错: {e}")
        return None


def calculate_property_diff(elements_str, property_dict):
    """计算材料中元素属性的最大差异

    参数:
    - elements_str: 元素符号的字符串，以逗号分隔，如 'Si,O'
    - property_dict: 元素属性字典，如 ELECTRONEGATIVITY

    返回:
    - 属性的最大差异，如果无法计算则返回0
    """
    if not elements_str or pd.isna(elements_str):
        return 0

    try:
        # 分割元素字符串并移除空白符
        elements = [elem.strip()
                    for elem in elements_str.split(',') if elem.strip()]
        if not elements:
            return 0

        values = []
        for element in elements:
            if element in property_dict:
                values.append(property_dict[element])

        if len(values) < 2:
            return 0

        return max(values) - min(values)
    except Exception as e:
        print(f"计算属性差异时出错: {e}")
        return 0


def most_common_valence(elements_str):
    """获取材料中最常见的价电子数

    参数:
    - elements_str: 元素符号的字符串，以逗号分隔，如 'Si,O'

    返回:
    - 最常见的价电子数，如果无法计算则返回None
    """
    if not elements_str or pd.isna(elements_str):
        return None

    try:
        # 分割元素字符串并移除空白符
        elements = [elem.strip()
                    for elem in elements_str.split(',') if elem.strip()]
        if not elements:
            return None

        valences = []
        for element in elements:
            if element in VALENCE_ELECTRONS:
                valences.append(VALENCE_ELECTRONS[element])

        if not valences:
            return None

        # 返回最常见的价电子数
        from collections import Counter
        counter = Counter(valences)
        return counter.most_common(1)[0][0]
    except Exception as e:
        print(f"获取常见价电子数时出错: {e}")
        return None


def calculate_property_variance(elements_str, property_dict):
    """计算材料中元素属性的方差

    参数:
    - elements_str: 元素符号的字符串，以逗号分隔，如 'Si,O'
    - property_dict: 元素属性字典，如 ELECTRONEGATIVITY

    返回:
    - 属性的方差值，如果元素少于2个则返回0
    """
    if not elements_str or pd.isna(elements_str):
        return 0

    try:
        # 分割元素字符串并移除空白符
        elements = [elem.strip()
                    for elem in elements_str.split(',') if elem.strip()]
        if len(elements) < 2:
            return 0

        values = []
        for element in elements:
            if element in property_dict:
                values.append(property_dict[element])

        if len(values) < 2:
            return 0

        # 计算方差
        import numpy as np
        return np.var(values)
    except Exception as e:
        print(f"计算属性方差时出错: {e}")
        return 0


def calculate_orbital_ratio(elements_str, orbital_type='s'):
    """计算材料中特定轨道类型元素的比例

    参数:
    - elements_str: 元素符号的字符串，以逗号分隔，如 'Si,O'
    - orbital_type: 轨道类型，'s', 'p', 'd' 或 'f'

    返回:
    - 特定轨道类型元素的比例 (0到1之间)
    """
    if not elements_str or pd.isna(elements_str):
        return 0

    try:
        # 分割元素字符串并移除空白符
        elements = [elem.strip()
                    for elem in elements_str.split(',') if elem.strip()]
        if not elements:
            return 0

        # 选择对应的轨道类型元素集合
        if orbital_type.lower() == 's':
            orbital_elements = S_BLOCK_ELEMENTS
        elif orbital_type.lower() == 'p':
            orbital_elements = P_BLOCK_ELEMENTS
        elif orbital_type.lower() == 'd':
            orbital_elements = D_BLOCK_ELEMENTS
        elif orbital_type.lower() == 'f':
            orbital_elements = F_BLOCK_ELEMENTS
        else:
            print(f"未知轨道类型: {orbital_type}")
            return 0

        # 计算轨道比例
        count = sum(1 for elem in elements if elem in orbital_elements)
        return count / len(elements)
    except Exception as e:
        print(f"计算轨道比例时出错: {e}")
        return 0


def calculate_metal_ratio(elements_str):
    """计算材料中金属元素的比例

    参数:
    - elements_str: 元素符号的字符串，以逗号分隔，如 'Si,O'

    返回:
    - 金属元素的比例 (0到1之间)
    """
    if not elements_str or pd.isna(elements_str):
        return 0

    try:
        # 分割元素字符串并移除空白符
        elements = [elem.strip()
                    for elem in elements_str.split(',') if elem.strip()]
        if not elements:
            return 0

        # 计算金属元素比例
        metal_count = sum(1 for elem in elements if elem in METAL_ELEMENTS)
        return metal_count / len(elements)
    except Exception as e:
        print(f"计算金属比例时出错: {e}")
        return 0


def prepare_features(df):
    """准备用于预测带隙的特征，包括元素数量、形成能和元素组分特征

    参数:
    - df: 包含材料数据的DataFrame

    返回:
    - X: 特征矩阵
    - y: 目标值（带隙）
    - feature_names: 特征名称列表，用于调试和分析
    """
    print("准备训练特征...")

    # 检查必要的列是否存在
    required_columns = ["nelements", "formation_energy_per_atom", "band_gap"]
    missing_columns = [
        col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"数据缺少必要的列: {', '.join(missing_columns)}")

    # 基础特征列表
    feature_names = ["nelements", "formation_energy_per_atom"]

    # 检查elements列是否存在
    if 'elements' not in df.columns:
        print("警告: 数据中没有元素组成信息，使用基本特征训练模型")
        X = df[feature_names].values
    else:
        print("发现元素组成数据，使用元素特性作为特征")

        # 处理elements列中的缺失值
        missing_elements = df['elements'].isna().sum()
        if missing_elements > 0:
            print(f"警告: {missing_elements}行数据缺少元素信息，将使用空字符串替代")
            df['elements'] = df['elements'].fillna('')

        # 检查elements列的格式
        sample_elements = df['elements'].iloc[0] if len(df) > 0 else ""
        print(f"元素数据示例: '{sample_elements}'")

        # 添加基于元素组成的特征
        print("生成元素特性特征...")

        # 计算平均电负性
        df['avg_electronegativity'] = df['elements'].apply(
            lambda x: calculate_avg_property(x, ELECTRONEGATIVITY))

        # 计算平均原子半径
        df['avg_atomic_radius'] = df['elements'].apply(
            lambda x: calculate_avg_property(x, ATOMIC_RADIUS))

        # 计算平均价电子数
        df['avg_valence'] = df['elements'].apply(
            lambda x: calculate_avg_property(x, VALENCE_ELECTRONS))

        # 计算电负性差异(最大值-最小值)
        df['electronegativity_diff'] = df['elements'].apply(
            lambda x: calculate_property_diff(x, ELECTRONEGATIVITY))

        # 计算原子半径差异(最大值-最小值)
        df['radius_diff'] = df['elements'].apply(
            lambda x: calculate_property_diff(x, ATOMIC_RADIUS))

        # 计算最常见价电子数
        df['common_valence'] = df['elements'].apply(
            lambda x: most_common_valence(x))

        # 额外计算电负性方差 - 反映元素电负性分布的均匀性
        df['electronegativity_var'] = df['elements'].apply(
            lambda x: calculate_property_variance(x, ELECTRONEGATIVITY))

        # 计算原子半径方差 - 反映原子尺寸分布的均匀性
        df['radius_var'] = df['elements'].apply(
            lambda x: calculate_property_variance(x, ATOMIC_RADIUS))

        # 计算s,p,d,f轨道元素的比例 - 这对能带结构有重要影响
        df['s_orbital_ratio'] = df['elements'].apply(
            lambda x: calculate_orbital_ratio(x, orbital_type='s'))
        df['p_orbital_ratio'] = df['elements'].apply(
            lambda x: calculate_orbital_ratio(x, orbital_type='p'))
        df['d_orbital_ratio'] = df['elements'].apply(
            lambda x: calculate_orbital_ratio(x, orbital_type='d'))

        # 计算金属元素比例 - 金属性对能隙有重要影响
        df['metal_ratio'] = df['elements'].apply(
            lambda x: calculate_metal_ratio(x))

        # 元素特性特征列表
        element_features = [
            'avg_electronegativity', 'avg_atomic_radius', 'avg_valence',
            'electronegativity_diff', 'radius_diff', 'common_valence',
            'electronegativity_var', 'radius_var',
            's_orbital_ratio', 'p_orbital_ratio', 'd_orbital_ratio',
            'metal_ratio'
        ]

        # 处理可能的缺失值
        for feature in element_features:
            missing_count = df[feature].isna().sum()
            if missing_count > 0:
                print(f"  特征 '{feature}' 有 {missing_count} 个缺失值，使用中位数填充")
                df[feature] = df[feature].fillna(df[feature].median())

        # 将特征名称扩展到列表中
        feature_names.extend(element_features)

        # 输出最终的特征集
        print(f"最终使用的特征 ({len(feature_names)}): {', '.join(feature_names)}")

        # 创建特征矩阵
        X = df[feature_names].values

    # 目标值
    y = df["band_gap"].values

    print(f"特征矩阵形状: {X.shape}, 目标值形状: {y.shape}")

    # 只返回特征名称以供参考，不在训练中使用
    return X, y, feature_names

# 训练模型


def train_model(X, y, feature_names=None, epochs=180):  # 增加训练轮数
    """训练神经网络模型，使用优化的训练策略

    参数:
    - X: 特征矩阵
    - y: 目标值
    - feature_names: 特征名称列表
    - epochs: 最大训练轮数

    返回:
    - model: 训练好的模型
    - scaler: 标准化器
    - feature_names: 特征名称列表
    """
    print(f"输入特征维度: {X.shape}, 目标值维度: {y.shape}")
    if feature_names:
        print(f"使用特征: {feature_names}")

    # 获取设备(GPU或CPU)
    device = get_device()

    # 划分训练集、验证集和测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42)
    print(
        f"训练集: {X_train.shape[0]}样本, 验证集: {X_val.shape[0]}样本, 测试集: {X_test.shape[0]}样本")

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 保存标准化器参数，包括特征名称
    feature_info = {'feature_names': feature_names} if feature_names else None
    save_result = save_scaler(scaler, SCALER_FILE, feature_info)
    if not save_result:
        print("警告: 保存标准化器参数失败，继续进行模型训练...")

    # 转换为张量并移动到指定设备上
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1)).to(device)

    # 创建数据加载器，使用小批量训练
    batch_size = min(64, len(X_train))
    train_dataset = torch.utils.data.TensorDataset(
        X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    input_size = X_train_scaled.shape[1]
    model = ImprovedMaterialsNet(input_size).to(device)  # 移动模型到GPU
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0012,  # 降低学习率，从0.001到0.0012
        weight_decay=2e-5,  # 增加权重衰减，从1e-5到2e-5
        eps=1e-8  # 添加数值稳定性参数
    )

    # 学习率调度器优化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.6,  # 从0.5调整到0.6，衰减更温和
        patience=15,  # 增加耐心值，从10到15
        min_lr=1e-6  # 设置最小学习率
    )

    # 用于保存最佳模型
    best_val_loss = float('inf')
    best_model_state = None
    patience = 30  # 增加早停耐心值，从25到30
    patience_counter = 0

    # 训练循环
    print(f"开始训练模型 (最大epochs={epochs})...")
    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_loss = 0.0

        # 使用批量训练
        for batch_X, batch_y in train_loader:
            # 确保数据在正确的设备上
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)

        # 评估模式
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

            # 调整学习率
            scheduler.step(val_loss)

            # 保存验证损失最小的模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # 早停
            if patience_counter >= patience:
                print(f"验证损失连续{patience}轮未改善，提前停止训练")
                break

        # 每10个epoch打印一次损失
        if (epoch+1) % 10 == 0:
            print(
                f'Epoch [{epoch+1}/{epochs}], 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}')

    print(f"训练完成，最佳验证损失: {best_val_loss:.6f}")

    # 加载最佳模型状态
    if best_model_state:
        model.load_state_dict(best_model_state)

    # 在测试集上评估模型
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor).item()

        # 计算R²分数
        from sklearn.metrics import r2_score, mean_absolute_error
        y_test_np = y_test
        test_pred_np = test_outputs.numpy().flatten()
        r2 = r2_score(y_test_np, test_pred_np)
        mae = mean_absolute_error(y_test_np, test_pred_np)

        print(f"测试集性能:")
        print(f"MSE: {test_loss:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")

    # 在加载最佳模型状态后移动回CPU，便于保存和后续使用
    if torch.cuda.is_available():
        model = model.cpu()
        print("模型已从GPU移到CPU，以便保存和预测")

    # 保存模型
    try:
        torch.save(model.state_dict(), MODEL_FILE)
        print(f"模型已保存到 {MODEL_FILE}")
    except Exception as e:
        print(f"保存模型时出错: {str(e)}")
        print("警告: 无法保存模型到文件，但模型仍可在当前会话中使用")

    return model, scaler, feature_names


def predict(model, scaler, nelements, formation_energy, elements=None, feature_names=None):
    """使用训练好的模型进行预测

    参数:
    - model: 训练好的模型
    - scaler: 特征缩放器
    - nelements: 元素数量
    - formation_energy: 每原子形成能
    - elements: 元素列表，例如 ['Si', 'O'] 或 'Si,O'
    - feature_names: 训练时使用的特征名称列表

    返回:
    - 预测的能隙值
    """
    model.eval()

    # 确保nelements和formation_energy是数值类型
    try:
        nelements = float(nelements)
        formation_energy = float(formation_energy)
    except (ValueError, TypeError) as e:
        print(f"错误：输入特征必须是数值类型 - {e}")
        return None

    # 基本特征
    features = [nelements, formation_energy]

    # 如果有元素信息，添加元素组分特征
    if elements:
        try:
            # 将元素列表转换为字符串
            if isinstance(elements, list):
                elements_str = ','.join(str(e) for e in elements)
            elif isinstance(elements, str):
                elements_str = elements
            else:
                print(f"警告：无法识别的元素格式: {type(elements)}")
                elements_str = str(elements)

            # 如果我们知道训练时使用了哪些特征，就按照相同顺序构建特征
            if feature_names and len(feature_names) > 2:  # 有除了基本特征外的特征
                # 重置特征列表，按照feature_names构建
                features = []

                # 构建一个字典，包含所有可能的特征值
                feature_values = {
                    'nelements': nelements,
                    'formation_energy_per_atom': formation_energy,
                    'avg_electronegativity': calculate_avg_property(elements_str, ELECTRONEGATIVITY) or 0.0,
                    'avg_atomic_radius': calculate_avg_property(elements_str, ATOMIC_RADIUS) or 0.0,
                    'avg_valence': calculate_avg_property(elements_str, VALENCE_ELECTRONS) or 0.0,
                    'electronegativity_diff': calculate_property_diff(elements_str, ELECTRONEGATIVITY) or 0.0,
                    'radius_diff': calculate_property_diff(elements_str, ATOMIC_RADIUS) or 0.0,
                    'common_valence': most_common_valence(elements_str) or 0.0,
                    'electronegativity_var': calculate_property_variance(elements_str, ELECTRONEGATIVITY) or 0.0,
                    'radius_var': calculate_property_variance(elements_str, ATOMIC_RADIUS) or 0.0,
                    's_orbital_ratio': calculate_orbital_ratio(elements_str, orbital_type='s') or 0.0,
                    'p_orbital_ratio': calculate_orbital_ratio(elements_str, orbital_type='p') or 0.0,
                    'd_orbital_ratio': calculate_orbital_ratio(elements_str, orbital_type='d') or 0.0,
                    'metal_ratio': calculate_metal_ratio(elements_str) or 0.0
                }

                # 按照特征名称顺序添加特征值
                for name in feature_names:
                    if name in feature_values:
                        features.append(feature_values[name])
                    else:
                        print(f"警告: 未知特征 '{name}'，使用0.0替代")
                        features.append(0.0)
            else:
                # 使用默认的特征计算方式
                # 计算元素相关特征
                avg_electronegativity = calculate_avg_property(
                    elements_str, ELECTRONEGATIVITY)
                avg_atomic_radius = calculate_avg_property(
                    elements_str, ATOMIC_RADIUS)
                avg_valence = calculate_avg_property(
                    elements_str, VALENCE_ELECTRONS)
                electronegativity_diff = calculate_property_diff(
                    elements_str, ELECTRONEGATIVITY)
                radius_diff = calculate_property_diff(
                    elements_str, ATOMIC_RADIUS)
                common_valence = most_common_valence(elements_str)

                # 处理可能的None值
                element_features = [
                    avg_electronegativity if avg_electronegativity is not None else 0.0,
                    avg_atomic_radius if avg_atomic_radius is not None else 0.0,
                    avg_valence if avg_valence is not None else 0.0,
                    electronegativity_diff if electronegativity_diff is not None else 0.0,
                    radius_diff if radius_diff is not None else 0.0,
                    common_valence if common_valence is not None else 0.0
                ]

                # 扩展特征列表
                features.extend(element_features)

        except Exception as e:
            print(f"处理元素特征时出错: {e}")
            print("将仅使用基本特征进行预测")

    # 转换为张量并进行预测
    try:
        with torch.no_grad():
            # 确保特征数量与训练时一致
            if len(features) != model.layer1.in_features:
                print(
                    f"警告: 特征数量不匹配! 预测使用 {len(features)} 个特征，但模型期望 {model.layer1.in_features} 个特征")
                # 如果特征过少，填充零值以匹配模型输入大小
                if len(features) < model.layer1.in_features:
                    features.extend(
                        [0.0] * (model.layer1.in_features - len(features)))
                # 如果特征过多，截断
                elif len(features) > model.layer1.in_features:
                    features = features[:model.layer1.in_features]

            # 转换为张量
            input_data = torch.FloatTensor(scaler.transform([features]))
            prediction = model(input_data)
            return prediction.item()
    except Exception as e:
        print(f"预测过程中出错: {e}")
        return None


def check_api_parameters():
    """检查MP API支持的参数并打印可用字段"""
    try:
        print("\n===== Materials Project API 参数检查 =====")
        with MPRester(API_KEY) as mpr:
            # 获取API版本
            try:
                import mp_api
                print(f"mp-api版本: {mp_api.__version__}")
            except (ImportError, AttributeError):
                print("无法获取mp-api版本信息")

            # 尝试获取可用字段
            try:
                print("\n可用字段:")
                available_fields = mpr.materials.summary.available_fields
                if available_fields:
                    print(f"共 {len(available_fields)} 个字段:")
                    # 每行显示5个字段
                    for i in range(0, len(available_fields), 5):
                        print(", ".join(available_fields[i:i+5]))
                else:
                    print("无可用字段或获取失败")
            except Exception as e:
                print(f"获取可用字段时出错: {str(e)}")

            print("\n推荐的查询参数:")
            print("1. elements - 包含特定元素的材料，例如: elements=['Si', 'O']")
            print("2. nelements - 元素数量的范围，例如: nelements=2 或 nelements=(2, 4)")
            print("3. band_gap - 带隙范围，例如: band_gap=(0.5, 2.0)")
            print("4. formula - 化学式，例如: formula='SiO2'")
            print("===== 参数检查结束 =====\n")

            return True
    except Exception as e:
        print(f"检查API参数时出错: {str(e)}")
        return False


def main():
    """主函数，获取数据、训练模型并进行测试预测"""
    print("\n===== MaterialDL: 材料带隙预测系统 =====\n")

    # 检查是否已存在数据文件
    if os.path.exists(DATASET_FILE):
        print(f"发现现有数据文件: {DATASET_FILE}")
        df = pd.read_csv(DATASET_FILE)
        print(f"加载了 {len(df)} 条材料数据")
    else:
        print(f"未找到数据文件，从Materials Project获取数据...")
        df = get_materials_data()
        if len(df) == 0:
            print("获取数据失败，请检查API密钥和网络连接")
            return

    # 提示用户选择特征工程方法
    print("\n特征工程选择:")
    print("1. 标准特征集 (基本特征)")
    print("2. 增强特征集 (包含更多材料科学特征，推荐)")
    feature_choice = input("请选择 [1/2] (默认为2): ").strip()

    # 数据处理
    if feature_choice == "1":
        print("\n使用标准特征集...")
        X, y, feature_names = prepare_features(df)
    else:
        print("\n使用增强特征集...")
        X, y, feature_names = prepare_features(df)

    if len(X) == 0:
        print("特征准备失败，无法继续训练")
        return

    # 提示用户选择训练模式
    print("\n模型选择:")
    print("1. 训练单个神经网络模型")
    print("2. 训练集成模型 (结合多个模型提高精度，推荐)")
    model_choice = input("请选择 [1/2] (默认为2): ").strip()

    if model_choice == "1":
        # 训练单个模型
        print("\n===== 开始训练单个神经网络模型 =====\n")
        model, scaler, feature_names = train_model(
            X, y, feature_names, epochs=300)

        # 示例预测
        if model is not None:
            print("\n===== 示例预测 =====\n")
            # 从测试集中随机选择几个样本进行预测
            indices = np.random.choice(len(X), min(5, len(X)), replace=False)

            model.eval()
            for idx in indices:
                element_info = ""
                if 'elements' in df.columns:
                    element_info = f", 元素: {df.iloc[idx]['elements']}"

                sample_x = torch.FloatTensor(scaler.transform([X[idx]]))

                with torch.no_grad():
                    pred = model(sample_x).item()
                    actual = y[idx]
                    print(
                        f"材料ID: {df.iloc[idx]['material_id'] if 'material_id' in df.columns else 'unknown'}{element_info}")
                    print(
                        f"实际带隙: {actual:.3f} eV, 预测带隙: {pred:.3f} eV, 误差: {abs(actual-pred):.3f} eV\n")

            print("模型训练和预测完成!")

        return model, scaler, feature_names
    else:
        # 训练集成模型
        print("\n===== 开始训练集成模型 =====\n")

        # 划分训练集、验证集和测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, random_state=42)
        print(
            f"训练集: {X_train.shape[0]}样本, 验证集: {X_val.shape[0]}样本, 测试集: {X_test.shape[0]}样本")

        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # 保存标准化器参数
        feature_info = {
            'feature_names': feature_names} if feature_names else None
        save_scaler(scaler, SCALER_FILE, feature_info)

        # 获取设备(GPU或CPU)
        device = get_device()

        # 转换为张量并移至设备
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1)).to(device)

        # 创建数据加载器
        batch_size = min(64, len(X_train))
        train_dataset = torch.utils.data.TensorDataset(
            X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)

        # 组合损失函数
        def combined_loss(pred, target):
            # 获取设备
            device = pred.device

            # 基础MSE损失
            mse = nn.MSELoss()(pred, target)

            # L1损失（MAE），对异常值不敏感
            mae = nn.L1Loss()(pred, target)

            # 调整权重比例
            return 0.7 * mse + 0.3 * mae  # 从0.1提高MAE的权重到0.3

        criterion = combined_loss

        # 创建并训练集成模型
        input_size = X_train_scaled.shape[1]
        ensemble = EnsembleModel(input_size, n_models=6)  # 从5增加到6
        ensemble.train(train_loader, X_val_tensor,
                       y_val_tensor, criterion, epochs=220)  # 增加训练轮数

        # 在测试集上评估集成模型
        print("\n===== 集成模型评估 =====\n")
        ensemble_pred = ensemble.predict(X_test_tensor)

        # 计算评估指标
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        test_pred = ensemble_pred.cpu().numpy().flatten()
        test_true = y_test_tensor.cpu().numpy().flatten()
        test_mse = mean_squared_error(test_true, test_pred)
        rmse = np.sqrt(test_mse)
        mae = mean_absolute_error(test_true, test_pred)
        r2 = r2_score(test_true, test_pred)

        print(f"测试集性能:")
        print(f"MSE: {test_mse:.6f}")
        print(f"RMSE: {rmse:.6f} eV")
        print(f"MAE: {mae:.6f} eV")
        print(f"R²: {r2:.6f}")

        # 创建集成模型目录
        ensemble_dir = os.path.join(DATASET_DIR, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)

        # 保存集成模型
        ensemble.save(ensemble_dir)

        # 保存单个最佳模型为标准文件
        best_model = ensemble.models[0]  # 选择第一个模型作为单个模型
        torch.save(best_model.state_dict(), MODEL_FILE)
        print(f"已保存单个模型到: {MODEL_FILE}")

        print("\n===== 集成模型训练完成 =====\n")

        # 示例预测
        print("\n===== 示例预测 =====\n")
        # 从测试集中随机选择几个样本
        indices = np.random.choice(
            len(X_test), min(5, len(X_test)), replace=False)

        for i, idx in enumerate(indices):
            # 从原始数据中获取材料信息
            original_idx = X_train.shape[0] + idx
            if original_idx < len(df) and 'material_id' in df.columns:
                material_id = df.iloc[original_idx]['material_id']
                print(f"材料ID: {material_id}")

            if original_idx < len(df) and 'elements' in df.columns:
                elements = df.iloc[original_idx]['elements']
                print(f"元素组成: {elements}")

            # 准备输入
            sample_x = X_test_tensor[idx].unsqueeze(0)

            # 使用集成模型预测
            with torch.no_grad():
                pred = ensemble.predict(sample_x).item()
                actual = y_test[idx]
                print(f"样本 {i+1}:")
                print(f"实际带隙: {actual:.4f} eV")
                print(f"预测带隙: {pred:.4f} eV")
                print(f"误差: {abs(actual-pred):.4f} eV\n")

        return ensemble, scaler, feature_names


# 检查命令行参数并运行主函数
if __name__ == "__main__":
    # 检查是否有命令行参数
    import sys
    if len(sys.argv) > 1:
        # 如果有参数 --check-api，则仅检查API参数
        if sys.argv[1] == "--check-api":
            check_api_parameters()
            print("\n是否继续执行主程序？(y/n): ", end="")
            choice = input().strip().lower()
            if choice == 'y':
                main()
            else:
                print("程序已退出。")
        else:
            print(f"未知参数: {sys.argv[1]}")
            print("可用参数: --check-api (检查API参数)")
            print("无参数则直接运行主程序")
    else:
        # 无参数则直接运行主程序
        main()


def to_cpu(tensor_or_model):
    """将张量或模型从GPU移动到CPU

    参数:
        tensor_or_model: 张量或PyTorch模型

    返回:
        移动到CPU的张量或模型
    """
    if torch.cuda.is_available():
        if hasattr(tensor_or_model, 'cpu'):
            return tensor_or_model.cpu()
        elif hasattr(tensor_or_model, 'to'):
            return tensor_or_model.to('cpu')
    return tensor_or_model
