import os
import json
import sys
import numpy as np
import pandas as pd
import warnings
from MaterialDL import ImprovedMaterialsNet, ELECTRONEGATIVITY, ATOMIC_RADIUS, VALENCE_ELECTRONS
from MaterialDL import calculate_avg_property, calculate_property_diff, most_common_valence
from MaterialDL import calculate_property_variance, calculate_orbital_ratio, calculate_metal_ratio
from MaterialDL import S_BLOCK_ELEMENTS, P_BLOCK_ELEMENTS, D_BLOCK_ELEMENTS, F_BLOCK_ELEMENTS, METAL_ELEMENTS
from MaterialDL import EnsembleModel
import torch
from sklearn.preprocessing import StandardScaler
import subprocess

# 定义数据和模型文件路径
DATASET_DIR = "dataset"
MODEL_FILE = os.path.join(DATASET_DIR, "model.pth")
SCALER_FILE = os.path.join(DATASET_DIR, "scaler.json")

# 检查依赖项
MISSING_DEPENDENCIES = []

try:
    import torch
except ImportError:
    MISSING_DEPENDENCIES.append("torch")
    warnings.warn("警告: 未找到PyTorch库。请运行 'pip install torch' 安装。")

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    MISSING_DEPENDENCIES.append("scikit-learn")
    warnings.warn("警告: 未找到scikit-learn库。请运行 'pip install scikit-learn' 安装。")

try:
    import pandas as pd
except ImportError:
    MISSING_DEPENDENCIES.append("pandas")
    warnings.warn("警告: 未找到pandas库。请运行 'pip install pandas' 安装。")

# 尝试导入MaterialDL模块的组件
try:
    # 这些导入已经在文件顶部完成，不需要重复
    MATERIAL_DL_IMPORTED = True
except ImportError as e:
    MISSING_DEPENDENCIES.append("MaterialDL")
    MATERIAL_DL_IMPORTED = False
    warnings.warn(f"警告: 导入MaterialDL模块失败: {str(e)}")


class MaterialPredictor:
    """材料带隙预测接口类"""

    def __init__(self, model_dir="dataset", use_ensemble=False):
        """初始化预测器

        参数:
        - model_dir: 模型和参数文件所在的目录
        - use_ensemble: 是否使用集成模型（如果存在）
        """
        self.model_dir = model_dir
        self.model_file = os.path.join(model_dir, "model.pth")
        self.scaler_file = os.path.join(model_dir, "scaler.json")
        self.ensemble_dir = os.path.join(model_dir, "ensemble")
        self.materialDL_path = "MaterialDL.py"
        self.use_ensemble = use_ensemble

        # 将是否使用了集成模型的标志
        self.is_ensemble = False

        # 确保模型目录存在
        self._ensure_model_dir()

        # 加载模型和标准化器
        self.model, self.scaler, self.feature_names = self._load_resources()

    def _ensure_model_dir(self):
        """确保模型目录和必要的文件存在"""
        try:
            # 确保目录存在
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
                print(f"已创建模型目录: {self.model_dir}")

            # 检查模型和标准化器文件是否存在
            model_exists = os.path.exists(self.model_file)
            scaler_exists = os.path.exists(self.scaler_file)

            if not model_exists or not scaler_exists:
                missing_files = []
                if not model_exists:
                    missing_files.append(f"模型文件({self.model_file})")
                if not scaler_exists:
                    missing_files.append(f"标准化器文件({self.scaler_file})")

                print(f"注意: 以下文件不存在: {', '.join(missing_files)}")
                print("这些文件将在模型训练后自动创建")

                # 向用户提供提示
                print("\n可以通过以下方式创建这些文件:")
                print("1. 直接运行MaterialDL.py来训练模型")
                print("2. 在使用该接口时选择'是'以自动训练模型")

            return True
        except Exception as e:
            print(f"创建模型目录或检查文件时出错: {str(e)}")
            return False

    def _load_resources(self):
        """加载模型和标准化器"""
        # 检查文件是否存在
        if not os.path.exists(self.model_file):
            print(f"模型文件不存在: {self.model_file}")
            print("请先运行 MaterialDL.py 训练模型")
            return None, None, None

        if not os.path.exists(self.scaler_file):
            print(f"标准化器文件不存在: {self.scaler_file}")
            print("请先运行 MaterialDL.py 训练模型")
            return None, None, None

        # 加载标准化器
        try:
            with open(self.scaler_file, 'r') as f:
                scaler_data = json.load(f)

            scaler = StandardScaler()
            scaler.mean_ = np.array(scaler_data['mean'])
            scaler.scale_ = np.array(scaler_data['scale'])
            scaler.var_ = np.array(scaler_data['var'])

            # 提取特征名称（如果存在）
            feature_names = scaler_data.get('feature_names', None)
            print(f"已加载标准化器，特征数量: {len(scaler.mean_)}")
        except Exception as e:
            print(f"加载标准化器时出错: {str(e)}")
            return None, None, None

        # 从MaterialDL导入必要的类
        try:
            sys.path.append(os.path.dirname(
                os.path.abspath(self.materialDL_path)))

            # 导入必要的类，移到条件判断外部
            from MaterialDL import ImprovedMaterialsNet

            # 尝试使用集成模型
            if self.use_ensemble and os.path.exists(self.ensemble_dir):
                from MaterialDL import EnsembleModel

                # 检查ensemble目录中是否有模型文件
                model_files = [f for f in os.listdir(self.ensemble_dir)
                               if f.startswith("model_") and f.endswith(".pth")]

                if model_files:
                    print(f"发现集成模型，加载中...")
                    input_size = len(scaler.mean_)
                    ensemble = EnsembleModel(input_size)

                    try:
                        ensemble.load(self.ensemble_dir, input_size)
                        self.is_ensemble = True
                        print(f"已加载集成模型，包含 {len(ensemble.models)} 个子模型")
                        return ensemble, scaler, feature_names
                    except Exception as e:
                        print(f"加载集成模型时出错: {str(e)}")
                        print("回退到加载单个模型...")
                        self.use_ensemble = False
                else:
                    print("未找到集成模型文件，使用单个模型...")
                    self.use_ensemble = False

            # 加载单个模型
            # 这里不需要重复导入，已经在上面导入过了

            input_size = len(scaler.mean_)
            model = ImprovedMaterialsNet(input_size)

            try:
                model.load_state_dict(torch.load(
                    self.model_file, weights_only=True))
                print(f"已加载模型，输入特征维度: {input_size}")
            except Exception as e:
                # 处理模型架构不匹配的情况
                print(f"加载模型状态字典时出错: {str(e)}")
                print("可能是模型架构已更新，尝试使用兼容性加载...")

                # 尝试加载与当前架构不完全匹配的模型
                state_dict = torch.load(self.model_file, weights_only=True)
                model_dict = model.state_dict()

                # 过滤掉不匹配的参数
                compatible_state_dict = {k: v for k, v in state_dict.items()
                                         if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(compatible_state_dict)
                model.load_state_dict(model_dict, strict=False)

                print(
                    f"已部分加载模型参数 ({len(compatible_state_dict)}/{len(state_dict)})")
                print("注意: 某些模型参数未能加载，预测结果可能会受到影响")
                print("建议重新运行 MaterialDL.py 训练新模型")

            model.eval()
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return None, None, None

        return model, scaler, feature_names

    def predict(self, nelements, formation_energy, elements=None):
        """预测材料的能带间隙

        参数:
        - nelements: 材料中的元素数量
        - formation_energy: 形成能 (eV/atom)
        - elements: 元素符号列表或逗号分隔的元素字符串

        返回:
        - band_gap: 预测的能带间隙值 (eV)
        """
        if self.model is None or self.scaler is None:
            print("错误: 模型未正确加载，无法进行预测")
            return None

        # 处理元素输入，统一格式为逗号分隔的字符串
        elements_str = None
        if elements:
            if isinstance(elements, list):
                elements_str = ','.join(elements)
            else:
                elements_str = str(elements)

        # 准备基本特征
        features_dict = {
            'nelements': nelements,
            'formation_energy_per_atom': formation_energy
        }

        # 如果提供了元素信息，计算元素特性
        if elements_str:
            # 计算平均电负性
            features_dict['avg_electronegativity'] = calculate_avg_property(
                elements_str, ELECTRONEGATIVITY) or 0.0

            # 计算平均原子半径
            features_dict['avg_atomic_radius'] = calculate_avg_property(
                elements_str, ATOMIC_RADIUS) or 0.0

            # 计算平均价电子数
            features_dict['avg_valence'] = calculate_avg_property(
                elements_str, VALENCE_ELECTRONS) or 0.0

            # 计算电负性差异(最大值-最小值)
            features_dict['electronegativity_diff'] = calculate_property_diff(
                elements_str, ELECTRONEGATIVITY) or 0.0

            # 计算原子半径差异(最大值-最小值)
            features_dict['radius_diff'] = calculate_property_diff(
                elements_str, ATOMIC_RADIUS) or 0.0

            # 计算最常见价电子数
            features_dict['common_valence'] = most_common_valence(
                elements_str) or 0.0

            # 计算电负性方差
            features_dict['electronegativity_var'] = calculate_property_variance(
                elements_str, ELECTRONEGATIVITY) or 0.0

            # 计算原子半径方差
            features_dict['radius_var'] = calculate_property_variance(
                elements_str, ATOMIC_RADIUS) or 0.0

            # 计算s,p,d轨道元素的比例
            features_dict['s_orbital_ratio'] = calculate_orbital_ratio(
                elements_str, orbital_type='s') or 0.0
            features_dict['p_orbital_ratio'] = calculate_orbital_ratio(
                elements_str, orbital_type='p') or 0.0
            features_dict['d_orbital_ratio'] = calculate_orbital_ratio(
                elements_str, orbital_type='d') or 0.0

            # 计算金属元素比例
            features_dict['metal_ratio'] = calculate_metal_ratio(
                elements_str) or 0.0

        # 如果有特征名称，使用指定特征
        X_pred = []
        if self.feature_names:
            # 收集所有请求的特征
            for feature in self.feature_names:
                if feature in features_dict:
                    X_pred.append(features_dict[feature])
                else:
                    print(f"警告: 缺少特征 '{feature}'，使用0代替")
                    X_pred.append(0.0)
        else:
            # 没有特征名称，使用所有可用特征
            X_pred = list(features_dict.values())

        # 转换为numpy数组
        X_pred_np = np.array(X_pred).reshape(1, -1)

        # 标准化特征
        X_pred_scaled = self.scaler.transform(X_pred_np)

        # 转换为PyTorch张量
        X_pred_tensor = torch.FloatTensor(X_pred_scaled)

        # 使用模型预测
        try:
            with torch.no_grad():
                if self.is_ensemble:
                    # 使用集成模型预测
                    prediction = self.model.predict(X_pred_tensor)
                else:
                    # 使用单个模型预测
                    self.model.eval()
                    prediction = self.model(X_pred_tensor)

                band_gap = prediction.item()
                return band_gap
        except Exception as e:
            print(f"预测时出错: {str(e)}")
            return None


# 提供一个简单的函数接口，方便直接导入使用
def predict_band_gap(nelements, formation_energy, elements=None, model_dir="dataset"):
    """预测材料的能带间隙

    参数:
    - nelements: 元素数量
    - formation_energy: 形成能
    - elements: 元素组成，如 "Si,O"
    - model_dir: 模型和参数文件所在的目录

    返回:
    - 预测的能带间隙值
    - 材料分类 (金属、半导体、绝缘体)
    """
    # 检查依赖项
    if MISSING_DEPENDENCIES:
        missing_deps = ", ".join(MISSING_DEPENDENCIES)
        return None, f"错误: 缺少必要的依赖项: {missing_deps}"

    predictor = MaterialPredictor(model_dir)
    if not predictor.model or not predictor.scaler:
        return None, "预测器未能正确初始化"

    return predictor.predict(nelements, formation_energy, elements)


# 提供安装依赖项的辅助函数
def install_dependencies():
    """尝试安装缺失的依赖项"""
    import subprocess

    print("正在检查并安装缺失的依赖项...")

    if "torch" in MISSING_DEPENDENCIES:
        print("正在安装 PyTorch...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "torch"])

    if "scikit-learn" in MISSING_DEPENDENCIES:
        print("正在安装 scikit-learn...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "scikit-learn"])

    if "pandas" in MISSING_DEPENDENCIES:
        print("正在安装 pandas...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pandas"])

    if "MaterialDL" in MISSING_DEPENDENCIES:
        print("错误: 无法自动安装 MaterialDL 模块")
        print("请确保 MaterialDL.py 文件位于当前目录或 Python 路径中")

    print("依赖项安装完成。请重新运行程序。")
    sys.exit(0)


# 如果直接运行此脚本，则展示示例用法
if __name__ == "__main__":
    # 检查是否有缺失的依赖项
    if MISSING_DEPENDENCIES:
        missing_deps = ", ".join(MISSING_DEPENDENCIES)
        print(f"错误: 缺少必要的依赖项: {missing_deps}")
        print("1. 自动安装依赖项")
        print("2. 退出程序")
        choice = input("请选择操作 (1/2): ")

        if choice == "1":
            install_dependencies()
        else:
            print("退出程序。请手动安装所需的依赖项后再运行。")
            sys.exit(1)

    # 示例1: 使用类
    try:
        predictor = MaterialPredictor()

        if predictor.model and predictor.scaler:
            # 预测SiO2
            band_gap, material_type = predictor.predict(2, -5.97, "Si,O")
            print(f"SiO2预测带隙: {band_gap:.2f} eV, 类型: {material_type}")

            # 预测Fe
            band_gap, material_type = predictor.predict(1, 0.0, "Fe")
            print(f"Fe预测带隙: {band_gap:.2f} eV, 类型: {material_type}")

            # 示例2: 使用函数接口
            band_gap, material_type = predict_band_gap(3, -3.2, "Li,Mn,O")
            print(f"LiMnO2预测带隙: {band_gap:.2f} eV, 类型: {material_type}")
    except Exception as e:
        print(f"运行示例时出错: {str(e)}")
        print("请确保已安装所有依赖项并且模型文件和标准化器文件正确配置。")
