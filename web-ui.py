# -*- coding: utf-8 -*-
from MaterialPredictor import MaterialPredictor, predict_band_gap
import gradio as gr
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
from io import BytesIO
import base64
import time

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                   'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体


# 确保可以导入MaterialDL
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 定义材料分类函数


def classify_material(band_gap):
    """根据带隙值对材料进行分类"""
    if band_gap < 0.1:
        return "金属/导体 (带隙 < 0.1 eV)"
    elif band_gap < 3.0:
        return "半导体 (带隙 0.1-3.0 eV)"
    else:
        return "绝缘体 (带隙 > 3.0 eV)"

# 定义预测函数


def predict_material(elements, nelements, formation_energy, use_ensemble=False):
    """使用模型预测材料带隙"""
    try:
        # 处理输入参数
        nelements = int(nelements)
        formation_energy = float(formation_energy)

        # 初始化预测器
        predictor = MaterialPredictor(use_ensemble=use_ensemble)

        # 检查模型是否正确加载
        if predictor.model is None:
            return "无法进行预测", "无法确定", "模型未正确加载，请检查模型文件是否存在，或先运行 MaterialDL.py 训练模型。"

        # 执行预测
        band_gap = predictor.predict(nelements, formation_energy, elements)

        if band_gap is None:
            return "预测失败", "无法确定", "预测过程中发生错误，请检查输入参数。"

        # 根据带隙值进行材料分类
        material_class = classify_material(band_gap)

        # 添加分类图标
        if band_gap < 0.1:
            material_class = "🔵 " + material_class  # 金属/导体
        elif band_gap < 3.0:
            material_class = "🟢 " + material_class  # 半导体
        else:
            material_class = "🟠 " + material_class  # 绝缘体

        # 构建详细信息
        elements_list = [e.strip() for e in elements.split(",") if e.strip()]
        details = f"材料组成: {', '.join(elements_list)}\n"
        details += f"元素数量: {nelements}\n"
        details += f"形成能: {formation_energy:.4f} eV/atom\n"
        details += f"预测带隙: {band_gap:.4f} eV\n"
        details += f"材料类型: {material_class}\n"
        details += f"使用模型: {'集成模型' if use_ensemble else '单一模型'}"

        # 更新可视化指示器
        # 注意：这部分代码不会直接影响返回值，但在实际应用中可以通过JavaScript更新UI
        # 这里只是为了保持代码完整性

        # 返回结果列表
        return f"{band_gap:.4f} eV", material_class, details

    except Exception as e:
        return "预测出错", "无法确定", f"发生错误: {str(e)}"

# 模型检查函数


def check_model_files():
    """检查模型文件是否存在"""
    dataset_dir = "dataset"
    model_file = os.path.join(dataset_dir, "model.pth")
    scaler_file = os.path.join(dataset_dir, "scaler.json")
    ensemble_dir = os.path.join(dataset_dir, "ensemble")

    # 检查目录
    if not os.path.exists(dataset_dir):
        return f"⚠️ 警告: 模型目录 '{dataset_dir}' 不存在。请先运行 MaterialDL.py 训练模型。"

    # 检查文件
    missing_files = []
    if not os.path.exists(model_file):
        missing_files.append(f"模型文件 ({model_file})")
    if not os.path.exists(scaler_file):
        missing_files.append(f"标准化器文件 ({scaler_file})")

    if missing_files:
        return f"⚠️ 警告: 以下文件不存在: {', '.join(missing_files)}。请先运行 MaterialDL.py 训练模型。"

    # 检查集成模型
    ensemble_status = ""
    if os.path.exists(ensemble_dir) and len(os.listdir(ensemble_dir)) > 0:
        ensemble_status = "✅ 集成模型已就绪"
    else:
        ensemble_status = "ℹ️ 集成模型不可用 (可选功能)"

    return f"✅ 模型文件就绪，可以开始预测。\n{ensemble_status}"

# 创建样本材料数据


def generate_samples():
    samples = [
        ["Si,O", 2, -5.23],  # 二氧化硅
        ["Fe,O", 2, -2.81],  # 氧化铁
        ["Ga,As", 2, -0.19],  # 砷化镓
        ["Cu,Zn,Sn,S", 4, -0.55],  # CZTS太阳能电池材料
        ["Ti,O", 2, -4.78],  # 二氧化钛
    ]
    return samples

# 绘制实际值与预测值拟合曲线函数


def plot_actual_vs_predicted(use_ensemble=False, force_regenerate=False):
    """绘制实际值与预测值的拟合曲线

    Args:
        use_ensemble (bool): 是否使用集成模型
        force_regenerate (bool): 是否强制重新生成测试数据

    Returns:
        tuple: (图表HTML, 统计信息HTML, 测试数据DataFrame, 状态信息HTML)
    """
    try:
        # 根据模型类型选择不同的结果文件
        dataset_dir = "dataset"
        model_type_str = "ensemble" if use_ensemble else "single"
        test_data_file = os.path.join(
            dataset_dir, f"test_results_{model_type_str}.csv")

        status_html = ""

        # 如果特定模型类型的测试结果文件不存在，尝试生成测试数据
        if not os.path.exists(test_data_file) or force_regenerate:
            status_html = f"<div style='color:blue; padding:10px; background:#e8f0f4; border-radius:5px;'>⏳ 正在生成{model_type_str}模型的测试数据...</div>"
            print(f"生成{model_type_str}模型的测试数据...")
            test_data = generate_test_data(
                use_ensemble, force_regenerate=force_regenerate)
            if test_data is None:
                return None, f"无法生成{model_type_str}模型的测试数据。请先运行 MaterialDL.py 训练模型。", None, f"<div style='color:red; padding:10px; background:#f4e8e8; border-radius:5px;'>❌ 测试数据生成失败</div>"
            else:
                status_html = f"<div style='color:green; padding:10px; background:#e8f4e8; border-radius:5px;'>✅ 成功生成{model_type_str}模型的测试数据</div>"
        else:
            # 加载已有的测试结果数据
            test_data = pd.read_csv(test_data_file, encoding='utf-8')
            status_html = f"<div style='color:green; padding:10px; background:#e8f4e8; border-radius:5px;'>✅ 使用已有的{model_type_str}模型测试数据</div>"

            # 检查文件是否为空或数据是否完整
            if len(test_data) < 5 or 'actual' not in test_data.columns or 'predicted' not in test_data.columns:
                status_html = f"<div style='color:blue; padding:10px; background:#e8f0f4; border-radius:5px;'>⏳ 已有的{model_type_str}模型测试数据不完整，重新生成中...</div>"
                print(f"已有的{model_type_str}模型测试数据不完整，重新生成...")
                test_data = generate_test_data(
                    use_ensemble, force_regenerate=True)
                if test_data is None:
                    return None, f"无法生成{model_type_str}模型的测试数据。请先运行 MaterialDL.py 训练模型。", None, f"<div style='color:red; padding:10px; background:#f4e8e8; border-radius:5px;'>❌ 测试数据重新生成失败</div>"
                else:
                    status_html = f"<div style='color:green; padding:10px; background:#e8f4e8; border-radius:5px;'>✅ 成功重新生成{model_type_str}模型的测试数据</div>"

        # 提取实际值和预测值
        actual = test_data['actual'].values
        predicted = test_data['predicted'].values

        # 计算评估指标
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        # 拟合线性回归
        slope, intercept = np.polyfit(actual, predicted, 1)

        # 创建散点图
        plt.figure(figsize=(10, 8))

        # 尝试配置支持中文显示
        try:
            # 根据操作系统设置合适的中文字体
            if os.name == 'nt':  # Windows
                if os.path.exists('C:/Windows/Fonts/msyh.ttc'):
                    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] + \
                        plt.rcParams['font.sans-serif']
            elif sys.platform == 'darwin':  # MacOS
                plt.rcParams['font.sans-serif'] = ['PingFang SC',
                                                   'Hiragino Sans GB'] + plt.rcParams['font.sans-serif']
            else:  # Linux和其他系统
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] + \
                    plt.rcParams['font.sans-serif']

            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.family'] = 'sans-serif'
        except Exception as e:
            print(f"配置中文字体时出错: {str(e)}")

        # 绘制散点图和拟合线
        plt.scatter(actual, predicted, alpha=0.7, s=60,
                    c='blue', label='数据点 | Data Points')

        # 绘制理想线 y=x
        ideal_line = np.linspace(min(actual), max(actual), 100)
        plt.plot(ideal_line, ideal_line, 'r--',
                 label='理想拟合线 y=x | Ideal Line', linewidth=2)

        # 绘制实际拟合线
        fit_line = slope * ideal_line + intercept
        plt.plot(ideal_line, fit_line, 'g-',
                 label=f'拟合线 y={slope:.2f}x+{intercept:.2f} | Fitted Line', linewidth=2)

        # 设置图表标题和标签
        model_title = "集成模型" if use_ensemble else "单一模型"
        plt.title(
            f'{model_title}预测性能 | {model_title.replace("模型", "")} Model Performance', fontsize=16)
        plt.xlabel('实际带隙值 (eV) | Actual Band Gap', fontsize=14)
        plt.ylabel('预测带隙值 (eV) | Predicted Band Gap', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)

        # 在图表上显示评估指标
        metrics_text = f'R² = {r2:.3f}\nMAE = {mae:.3f} eV\nRMSE = {rmse:.3f} eV'
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

        # 保存到内存中的图片
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_data = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close()

        # 创建HTML显示图片
        img_html = f'<img src="data:image/png;base64,{img_data}" alt="拟合曲线">'

        # 创建统计信息HTML
        stats_html = f"""
        <div style="padding: 15px; background-color: #f8f9fa; border-radius: 10px; margin-top: 10px;">
            <h3 style="margin-top: 0; color: #333;">模型性能统计</h3>
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                <tr>
                    <th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd; width: 30%;">指标</th>
                    <th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">值</th>
                    <th style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">解释</th>
                </tr>
                <tr>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;"><b>R²</b> (决定系数)</td>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">{r2:.4f}</td>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">越接近1表示模型解释了数据中越多的变异</td>
                </tr>
                <tr>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;"><b>MAE</b> (平均绝对误差)</td>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">{mae:.4f} eV</td>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">预测值与实际值差异的平均大小</td>
                </tr>
                <tr>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;"><b>RMSE</b> (均方根误差)</td>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">{rmse:.4f} eV</td>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">对较大误差更敏感的误差衡量</td>
                </tr>
                <tr>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;"><b>拟合线斜率</b></td>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">{slope:.4f}</td>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">理想值为1.0，表示预测值与实际值变化比例相同</td>
                </tr>
                <tr>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;"><b>拟合线截距</b></td>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">{intercept:.4f}</td>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">理想值为0.0，表示无系统性偏差</td>
                </tr>
                <tr>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;"><b>样本数量</b></td>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">{len(actual)}</td>
                    <td style="padding: 8px; text-align: left; border-bottom: 1px solid #ddd;">用于评估的材料样本总数</td>
                </tr>
            </table>
        </div>
        """

        # 计算绝对误差
        test_data['absolute_error'] = np.abs(
            test_data['actual'] - test_data['predicted'])

        # 按绝对误差大小排序
        test_data_sorted = test_data.sort_values(
            by='absolute_error', ascending=False)

        # 创建用于显示的数据框
        display_df = pd.DataFrame({
            '材料成分': test_data_sorted['elements'],
            '元素数量': test_data_sorted['nelements'],
            '形成能': test_data_sorted['formation_energy'].round(4),
            '实际带隙': test_data_sorted['actual'].round(4),
            '预测带隙': test_data_sorted['predicted'].round(4),
            '绝对误差': test_data_sorted['absolute_error'].round(4)
        })

        return img_html, stats_html, display_df, status_html

    except Exception as e:
        print(f"生成拟合曲线图时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"生成拟合曲线图时出错: {str(e)}", None, f"<div style='color:red; padding:10px; background:#f4e8e8; border-radius:5px;'>❌ 生成拟合曲线图时出错: {str(e)}</div>"

# 生成测试数据函数


def generate_test_data(use_ensemble=False, force_regenerate=False):
    """生成测试数据用于绘图

    Args:
        use_ensemble (bool): 是否使用集成模型
        force_regenerate (bool): 是否强制重新生成数据

    Returns:
        DataFrame: 包含测试结果的数据框
    """
    try:
        # 根据模型类型选择结果文件
        dataset_dir = "dataset"
        model_type_str = "ensemble" if use_ensemble else "single"
        results_file = os.path.join(
            dataset_dir, f"test_results_{model_type_str}.csv")

        # 如果结果文件已存在且不强制重新生成，则直接返回
        if os.path.exists(results_file) and not force_regenerate:
            try:
                results_df = pd.read_csv(results_file, encoding='utf-8')
                if len(results_df) > 0 and 'actual' in results_df.columns and 'predicted' in results_df.columns:
                    print(f"使用已有的{model_type_str}模型测试数据")
                    return results_df
            except Exception as e:
                print(f"读取已有测试结果出错: {str(e)}，将重新生成")
                # 如果读取出错，继续执行并重新生成数据

        # 如果强制重新生成，输出日志
        if force_regenerate and os.path.exists(results_file):
            print(f"强制重新生成{model_type_str}模型测试数据")

        # 加载Materials项目测试数据
        data_file = os.path.join(dataset_dir, "materials_data.csv")

        if not os.path.exists(data_file):
            print("找不到材料数据文件，无法生成测试数据")
            return None

        # 加载数据
        materials_df = pd.read_csv(data_file, encoding='utf-8')

        # 取一部分数据作为测试集 (随机60个样本)
        # 对于强制重新生成，使用不同的随机种子以获得不同样本
        random_seed = 42 if not force_regenerate else int(time.time()) % 1000
        np.random.seed(random_seed)
        print(f"使用随机种子 {random_seed} 生成测试样本")
        test_indices = np.random.choice(
            len(materials_df), min(60, len(materials_df)), replace=False)
        test_df = materials_df.iloc[test_indices].copy()

        print(f"初始化{model_type_str}模型预测器...")
        # 初始化预测器
        predictor = MaterialPredictor(use_ensemble=use_ensemble)

        if predictor.model is None:
            print(f"{model_type_str}模型未正确加载，无法生成测试数据")
            return None

        # 对测试集进行预测
        print(f"使用{model_type_str}模型进行预测...")
        predicted_values = []
        for idx, row in test_df.iterrows():
            elements = row['elements']
            nelements = row['nelements']
            formation_energy = row['formation_energy_per_atom']

            # 预测带隙
            band_gap = predictor.predict(nelements, formation_energy, elements)
            predicted_values.append(band_gap if band_gap is not None else 0.0)

        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'elements': test_df['elements'],
            'nelements': test_df['nelements'],
            'formation_energy': test_df['formation_energy_per_atom'],
            'actual': test_df['band_gap'],
            'predicted': predicted_values
        })

        # 保存结果
        results_df.to_csv(results_file, index=False, encoding='utf-8')
        print(f"{model_type_str}模型测试数据已保存到 {results_file}")

        return results_df

    except Exception as e:
        print(f"生成测试数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 清除测试数据函数


def clear_test_data(model_type_str):
    """清除测试数据文件

    Args:
        model_type_str: 模型类型字符串，"单一模型"或"集成模型"

    Returns:
        str: 处理结果消息
    """
    try:
        use_ensemble = model_type_str == "集成模型"
        dataset_dir = "dataset"
        model_type = "ensemble" if use_ensemble else "single"
        test_data_file = os.path.join(
            dataset_dir, f"test_results_{model_type}.csv")

        if os.path.exists(test_data_file):
            os.remove(test_data_file)
            return f"<div style='color:green; padding:10px; background:#e8f4e8; border-radius:5px;'>✅ 已成功清除{model_type_str}的测试数据。</div>"
        else:
            return f"<div style='color:blue; padding:10px; background:#e8f0f4; border-radius:5px;'>ℹ️ 没有找到{model_type_str}的测试数据文件。</div>"

    except Exception as e:
        return f"<div style='color:red; padding:10px; background:#f4e8e8; border-radius:5px;'>❌ 清除数据时出错: {str(e)}</div>"

# 批量预测函数
def batch_predict(file_path, use_ensemble=True):
    """批量预测材料带隙
    
    Args:
        file_path (str): 上传的CSV文件路径
        use_ensemble (bool): 是否使用集成模型
        
    Returns:
        tuple: (DataFrame, str) - 预测结果数据框和状态信息HTML
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return None, f"<div style='color:red; padding:10px; background:#f4e8e8; border-radius:5px;'>❌ 文件不存在</div>"
        
        # 检查文件扩展名
        if not file_path.lower().endswith('.csv'):
            return None, f"<div style='color:red; padding:10px; background:#f4e8e8; border-radius:5px;'>❌ 请上传CSV格式文件</div>"
        
        # 读取CSV文件
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            return None, f"<div style='color:red; padding:10px; background:#f4e8e8; border-radius:5px;'>❌ 读取CSV文件失败: {str(e)}</div>"
        
        # 检查必要的列是否存在
        required_columns = ['elements', 'nelements', 'formation_energy']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return None, f"<div style='color:red; padding:10px; background:#f4e8e8; border-radius:5px;'>❌ CSV文件缺少必要的列: {', '.join(missing_columns)}</div>"
        
        # 初始化预测器
        predictor = MaterialPredictor(use_ensemble=use_ensemble)
        
        # 检查模型是否正确加载
        if predictor.model is None:
            return None, f"<div style='color:red; padding:10px; background:#f4e8e8; border-radius:5px;'>❌ 模型未正确加载，请检查模型文件是否存在</div>"
        
        # 准备结果列表
        results = []
        errors = []
        
        # 对每一行数据进行预测
        for idx, row in df.iterrows():
            try:
                # 确保元素是字符串类型，并去除可能的引号
                elements = str(row['elements']).strip('"\'')
                nelements = int(row['nelements'])
                formation_energy = float(row['formation_energy'])
                
                # 预测带隙
                band_gap = predictor.predict(nelements, formation_energy, elements)
                
                if band_gap is None:
                    errors.append(f"行 {idx+1}: 预测失败")
                    band_gap = 0.0
                
                # 根据带隙值进行材料分类
                material_class = classify_material(band_gap)
                
                # 添加结果
                results.append({
                    'elements': elements,
                    'nelements': nelements,
                    'formation_energy': formation_energy,
                    'predicted_band_gap': band_gap,
                    'material_class': material_class
                })
            except Exception as e:
                errors.append(f"行 {idx+1}: {str(e)}")
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        
        # 生成状态信息
        if errors:
            status_html = f"<div style='color:orange; padding:10px; background:#fff9e6; border-radius:5px;'>⚠️ 批量预测完成，但有 {len(errors)} 个错误:<br>" + "<br>".join(errors) + "</div>"
        else:
            status_html = f"<div style='color:green; padding:10px; background:#e8f4e8; border-radius:5px;'>✅ 批量预测成功，共处理 {len(results)} 条数据</div>"
        
        return results_df, status_html
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"<div style='color:red; padding:10px; background:#f4e8e8; border-radius:5px;'>❌ 批量预测过程中发生错误: {str(e)}</div>"

# 生成下载链接函数
def download_batch_results(results_df):
    """生成批量预测结果的下载链接
    
    Args:
        results_df (DataFrame): 预测结果数据框
        
    Returns:
        str: 下载链接
    """
    if results_df is None or len(results_df) == 0:
        return None
    
    try:
        # 创建临时文件
        temp_file = os.path.join(os.getcwd(), "temp_batch_results.csv")
        
        # 复制DataFrame以避免修改原始数据
        export_df = results_df.copy()
        
        # 确保元素列的值被引号包围
        if 'elements' in export_df.columns:
            export_df['elements'] = export_df['elements'].apply(lambda x: f'"{x}"' if not str(x).startswith('"') else x)
        
        # 保存结果到临时文件
        export_df.to_csv(temp_file, index=False, encoding='utf-8', quoting=1)  # quoting=1 表示为非数值列添加引号
        
        # 读取文件内容
        with open(temp_file, 'rb') as f:
            content = f.read()
        
        # 删除临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # 编码文件内容
        b64_content = base64.b64encode(content).decode('utf-8')
        
        # 生成下载链接
        download_link = f"data:text/csv;base64,{b64_content}"
        
        return download_link
    
    except Exception as e:
        print(f"生成下载链接时出错: {str(e)}")
        return None

# 生成示例CSV文件下载链接
def get_example_csv():
    """生成示例CSV文件的下载链接
    
    Returns:
        str: 下载链接HTML
    """
    try:
        # 示例CSV内容
        csv_content = """elements,nelements,formation_energy
"Si,O",2,-4.5
"Fe,O",2,-3.2
"Al,O",2,-5.1
"Cu,Zn,Sn,S",4,-0.55
"Ti,O",2,-4.78"""
        
        # 编码文件内容
        b64_content = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
        
        # 生成下载链接
        download_html = f"""
        <a href="data:text/csv;base64,{b64_content}" download="example_materials.csv" 
           style="display: block; text-align: center; padding: 10px; background-color: #f0f9ff; 
                  border-radius: 5px; text-decoration: none; color: #2563eb; font-weight: 500;">
            ⬇️ 点击此处下载示例CSV文件
        </a>
        """
        
        return download_html
    
    except Exception as e:
        print(f"生成示例CSV下载链接时出错: {str(e)}")
        return "<div style='color:red;'>生成示例文件失败</div>"

# 添加一个函数来更新指示器
def update_indicator(band_gap):
    """更新带隙指示器
    
    Args:
        band_gap: 预测的带隙值
        
    Returns:
        HTML指示器代码
    """
    try:
        if band_gap is None or not isinstance(band_gap, (int, float)):
            # 提取数字
            if isinstance(band_gap, str):
                import re
                match = re.search(r'(\d+\.\d+)', band_gap)
                if match:
                    band_gap = float(match.group(1))
                else:
                    return None
            else:
                return None
        
        # 确保带隙值是浮点数
        band_gap = float(band_gap)
        
        # 计算位置百分比 (0-8 eV范围)
        max_band_gap = 8.0
        position = min(max((band_gap / max_band_gap) * 100, 0), 100)
        
        # 生成HTML
        indicator_html = f"""
        <div style="text-align: center; width: 100%;">
            <div style="display: inline-block; width: 100%; max-width: 300px; height: 30px; background: linear-gradient(to right, #3498db, #2ecc71, #f1c40f, #e74c3c); border-radius: 15px; position: relative; margin-top: 10px;">
                <div style="position: absolute; top: -10px; left: {position}%; transform: translateX(-50%);">
                    <div style="width: 20px; height: 20px; background-color: #333; border-radius: 50%; border: 3px solid white;"></div>
                    <div style="color: #333; font-weight: bold; margin-top: 5px;">{band_gap:.2f} eV</div>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 0 10px; margin-top: 35px; font-size: 0.8rem;">
                    <span>金属</span>
                    <span>半导体</span>
                    <span>绝缘体</span>
                </div>
            </div>
        </div>
        """
        return indicator_html
    except Exception as e:
        print(f"更新指示器错误: {str(e)}")
        return None

# 创建Gradio界面


def create_interface():
    # 检查模型文件
    model_status = check_model_files()
    

    # 定义输入组件
    with gr.Blocks(title="材料带隙预测系统", ) as demo:
        with gr.Row(elem_classes=["header-container"]):
            gr.HTML("""
            <div class="header">
                <h1>🔬 材料带隙预测系统</h1>
                <p class="subtitle">基于深度学习的无机材料带隙预测工具</p>
            </div>
            """)

        # 模型状态信息
        with gr.Column():
            # 修复f-string中使用反斜杠的问题
            formatted_status = model_status.replace("✅", "✅ ")
            formatted_status = formatted_status.replace("⚠️", "⚠️ ")
            formatted_status = formatted_status.replace("ℹ️", "ℹ️ ")
            formatted_status = formatted_status.replace("\n", "<br>")
            gr.HTML(
                f"""<div class='model-status'>
                    <div class="status-icon">📊</div>
                    <div class="status-content">
                        <h3>系统状态</h3>
                        {formatted_status}
                    </div>
                </div>"""
            )

        with gr.Tabs():
            with gr.Tab("🔍 单材料预测"):
                with gr.Row():
                    with gr.Column(scale=3):
                        # 输入组件
                        gr.Markdown("### 输入参数")
                        
                        with gr.Group(elem_classes=["input-card"]):
                            elements_input = gr.Textbox(
                                label="元素组成",
                                placeholder="输入元素符号，用逗号分隔 (例如: Si,O)"
                            )
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    nelements_input = gr.Number(
                                        label="元素数量",
                                        value=2,
                                        minimum=1,
                                        maximum=10,
                                        step=1
                                    )
                                
                                with gr.Column(scale=1):
                                    formation_energy_input = gr.Number(
                                        label="形成能 (eV/atom)",
                                        value=-3.0,
                                        minimum=-10,
                                        maximum=10
                                    )
                            
                            with gr.Row():
                                use_ensemble = gr.Checkbox(
                                    label="使用集成模型 (通常更准确)",
                                    value=False
                                )
                                
                                predict_btn = gr.Button(
                                    "🔮 预测带隙", 
                                    variant="primary",
                                    elem_id="predict-btn"
                                )
                            
                            # 添加元素周期表参考
                            with gr.Accordion("元素周期表参考", open=False):
                                gr.HTML("""
                                <div style="text-align: center;">
                                    <img src="https://upload.wikimedia.org/wikipedia/commons/4/44/%E5%85%83%E7%B4%A0%E5%91%A8%E6%9C%9F%E8%A1%A8.png" 
                                         alt="元素周期表" 
                                         style="max-width: 100%; border-radius: 8px; margin-top: 10px;">
                                    <p style="font-size: 0.8rem; color: #666; margin-top: 5px;">
                                        点击查看元素周期表，帮助输入正确的元素符号
                                    </p>
                                </div>
                                """)

                    with gr.Column(scale=2):
                        # 输出组件
                        gr.Markdown("### 预测结果")
                        
                        with gr.Group(elem_classes=["result-card"]):
                            band_gap_output = gr.Textbox(
                                label="预测带隙",
                                elem_id="band-gap-output"
                            )
                            
                            material_class_output = gr.Textbox(
                                label="材料分类",
                                elem_id="material-class-output"
                            )
                            
                            details_output = gr.Textbox(
                                label="详细信息", 
                                lines=6,
                                elem_id="details-output"
                            )
                            
                            # 添加可视化指示器
                            indicator_html = gr.HTML(
                                """<div style="text-align: center; padding: 20px;">
                                    <p style="color: #666;">预测完成后将显示带隙指示器</p>
                                </div>""",
                                elem_id="band-gap-visualization"
                            )

                # 样本数据
                with gr.Row():
                    gr.Markdown("### 示例材料")
                    gr.Markdown("点击下方示例快速测试预测功能:")

                sample_data = generate_samples()
                gr.Examples(
                    examples=sample_data,
                    inputs=[elements_input, nelements_input, formation_energy_input],
                    outputs=[band_gap_output, material_class_output, details_output, indicator_html],
                    fn=lambda e, n, f: predict_material(e, n, f, False) + (update_indicator(predict_material(e, n, f, False)[0]),)
                )

            with gr.Tab("📊 批量预测"):
                with gr.Column():
                    gr.Markdown("### 批量预测功能")
                    
                    with gr.Group(elem_classes=["batch-card"]):
                        with gr.Row():
                            with gr.Column(scale=3):
                                file_input = gr.File(
                                    label="上传CSV文件",
                                    file_types=[".csv"]
                                )
                            
                            with gr.Column(scale=1, min_width=200):
                                batch_use_ensemble = gr.Checkbox(
                                    label="使用集成模型",
                                    value=True
                                )
                                
                                batch_predict_btn = gr.Button(
                                    "🔮 批量预测", 
                                    variant="primary",
                                    interactive=True
                                )
                        
                        # 添加状态信息区域
                        batch_status = gr.HTML(
                            """<div style="text-align: center; padding: 10px;">
                                <p style="color: #666;">上传CSV文件并点击"批量预测"按钮开始处理</p>
                            </div>""",
                            label="处理状态"
                        )
                    
                    # 添加结果显示区域
                    with gr.Group(elem_classes=["batch-results-card"], visible=False) as batch_results_container:
                        gr.Markdown("### 预测结果")
                        
                        with gr.Row():
                            batch_results = gr.DataFrame(
                                label="批量预测结果",
                                interactive=False
                            )
                        
                        with gr.Row():
                            download_btn = gr.Button(
                                "📥 下载结果", 
                                variant="secondary"
                            )
                            
                            download_link = gr.HTML(visible=False)
                    
                    with gr.Accordion("CSV文件格式说明", open=True):
                        gr.Markdown("""
                        #### CSV文件格式要求
                        上传的CSV文件必须包含以下列：
                        
                        | 列名 | 描述 | 示例 |
                        |------|------|------|
                        | `elements` | 元素组成（用逗号分隔） | `Si,O` |
                        | `nelements` | 元素数量 | `2` |
                        | `formation_energy` | 形成能 (eV/atom) | `-3.0` |
                        
                        #### 示例CSV内容
                        """)
                        
                        # 添加示例CSV内容
                        gr.Code(
                            """elements,nelements,formation_energy
"Si,O",2,-4.5
"Fe,O",2,-3.2
"Al,O",2,-5.1
"Cu,Zn Sn S",4,-0.55
"Ti,O",2,-4.78""", 
                            language="markdown"
                        )
                        
                        gr.Markdown("""
                        #### 注意事项
                        - 确保CSV文件使用UTF-8编码
                        - 确保元素符号正确（例如：Fe而不是FE）
                        - 确保数值列（nelements和formation_energy）包含有效的数字
                        - 每行数据将单独进行预测，预测结果将包含预测的带隙值和材料分类
                        """)
                        
                        # 添加下载示例按钮
                        with gr.Row():
                            download_example_btn = gr.Button("下载示例CSV文件")
                            example_download_link = gr.HTML(visible=False)

            # 新增拟合曲线选项卡
            with gr.Tab("📈 模型性能评估"):
                with gr.Column():
                    gr.Markdown("### 模型预测性能评估")
                    
                    with gr.Group(elem_classes=["performance-card"]):
                        gr.Markdown("""
                        本页面展示模型在预测材料带隙时的性能表现。通过比较实际带隙值与预测带隙值，
                        可以评估模型的准确性和可靠性。您可以选择单一模型或集成模型进行评估。
                        """)
                        
                        with gr.Row():
                            with gr.Column(scale=2, min_width=250):
                                # 选择使用哪种模型
                                model_type = gr.Radio(
                                    label="选择评估的模型类型",
                                    choices=["单一模型", "集成模型"],
                                    value="单一模型",
                                    interactive=True,
                                    elem_classes=["model-type-selector"]
                                )
                            
                            with gr.Column(scale=1, min_width=200):
                                # 添加重新生成数据的选项
                                regenerate_data = gr.Checkbox(
                                    label="重新生成测试数据",
                                    value=False,
                                    info="选中此项将强制重新生成测试数据，会覆盖现有结果",
                                    elem_classes=["regenerate-checkbox"]
                                )
                        
                        with gr.Row():
                            # 生成拟合曲线按钮
                            plot_btn = gr.Button(
                                "📊 生成拟合曲线", 
                                variant="primary", 
                                elem_id="plot-btn",
                                scale=3,
                                min_width=200
                            )
                            
                            # 清除测试数据按钮
                            clear_btn = gr.Button(
                                "🗑️ 清除测试数据", 
                                variant="secondary", 
                                elem_id="clear-btn",
                                scale=1,
                                min_width=150
                            )
                        
                        # 显示处理状态信息
                        status_output = gr.HTML(
                            """<div style="text-align: center; padding: 10px; margin-top: 10px;">
                                <p style="color: #666;">点击"生成拟合曲线"按钮查看模型性能</p>
                            </div>""",
                            label="状态信息",
                            elem_id="plot-status",
                            elem_classes=["status-output"]
                        )
                    
                    # 显示拟合曲线和统计信息
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=3, min_width=400):
                            with gr.Group(elem_classes=["plot-container"]):
                                plot_output = gr.HTML(
                                    label="拟合曲线",
                                    elem_classes=["plot-output"]
                                )
                        
                        with gr.Column(scale=2, min_width=300):
                            with gr.Group(elem_classes=["stats-container"]):
                                stats_output = gr.HTML(
                                    label="性能统计",
                                    elem_classes=["stats-output"]
                                )
                    
                    # 解释
                    with gr.Accordion("图表说明", open=False):
                        gr.Markdown("""
                        #### 图表解释
                        - **蓝色散点**：实际值-预测值数据点
                        - **红色虚线**：理想拟合线 (y=x)，表示完美预测
                        - **绿色实线**：实际拟合线，显示预测值与实际值的整体趋势
                        - **R²**：决定系数，越接近1表示拟合效果越好
                        - **MAE**：平均绝对误差，单位为eV
                        - **RMSE**：均方根误差，单位为eV
                        
                        #### 如何解读结果
                        - 理想情况下，散点应紧密围绕理想拟合线(红色虚线)
                        - R²值越接近1，表明模型解释了数据中的大部分变异
                        - 集成模型通常比单一模型表现更好，尤其在复杂材料上
                        - 如果某种类型的材料点远离拟合线，可能表明模型对该类材料预测不佳
                        """)
                    
                    # 添加数据点详情
                    with gr.Accordion("测试数据详情", open=False):
                        with gr.Group(elem_classes=["data-details-container"]):
                            data_details = gr.DataFrame(
                                headers=["材料成分", "元素数量", "形成能",
                                         "实际带隙", "预测带隙", "绝对误差"],
                                type="pandas",
                                elem_classes=["data-details-table"],
                                wrap=True
                            )

            with gr.Tab("ℹ️ 使用说明"):
                with gr.Group(elem_classes=["guide-container"]):
                    gr.Markdown("""
                    # 使用指南
                    
                    ## 基本操作
                    1. 在"单材料预测"选项卡中输入材料信息
                    2. 点击"预测带隙"按钮获取结果
                    3. 结果将显示预测的带隙值、材料分类和详细信息
                    
                    ## 输入参数说明
                    
                    | 参数 | 说明 | 示例 |
                    |------|------|------|
                    | **元素组成** | 输入构成材料的元素符号，用逗号分隔 | `Si,O` 表示二氧化硅 |
                    | **元素数量** | 材料中不同元素的数量 | `2` (对于SiO₂) |
                    | **形成能** | 材料的形成能，单位为eV/atom | `-3.0` |
                    | **使用集成模型** | 如果可用，使用多个模型的集成进行预测，通常可提高准确性 | - |
                    
                    ## 材料分类标准
                    
                    <div style="display: flex; justify-content: space-between; margin: 20px 0;">
                        <div style="text-align: center; padding: 15px; background-color: #e6f7ff; border-radius: 8px; width: 30%;">
                            <div style="font-size: 24px; margin-bottom: 10px;">🔵</div>
                            <div style="font-weight: bold;">金属/导体</div>
                            <div>带隙 < 0.1 eV</div>
                        </div>
                        <div style="text-align: center; padding: 15px; background-color: #e6ffed; border-radius: 8px; width: 30%;">
                            <div style="font-size: 24px; margin-bottom: 10px;">🟢</div>
                            <div style="font-weight: bold;">半导体</div>
                            <div>带隙 0.1-3.0 eV</div>
                        </div>
                        <div style="text-align: center; padding: 15px; background-color: #fff7e6; border-radius: 8px; width: 30%;">
                            <div style="font-size: 24px; margin-bottom: 10px;">🟠</div>
                            <div style="font-weight: bold;">绝缘体</div>
                            <div>带隙 > 3.0 eV</div>
                        </div>
                    </div>
                    
                    ## 注意事项
                    - 确保输入的元素符号正确（例如：Fe而不是FE）
                    - 形成能通常为负值，表示材料的稳定性
                    - 如果预测结果不准确，可尝试使用集成模型
                    
                    ## 模型性能评估
                    - 使用"模型性能评估"选项卡可查看模型预测的准确性
                    - 拟合曲线显示了预测值与实际值的对比
                    - R²值越接近1，表示模型预测性能越好
                    """)
                    
                    # 添加常见问题解答
                    with gr.Accordion("常见问题解答", open=False):
                        gr.Markdown("""
                        ### 常见问题解答
                        
                        #### Q: 如何获取材料的形成能数据？
                        A: 形成能数据可以从Materials Project、OQMD等材料数据库获取，也可以通过第一性原理计算获得。
                        
                        #### Q: 预测结果的准确性如何？
                        A: 模型的准确性取决于训练数据的质量和数量。对于常见材料，预测误差通常在0.3-0.5 eV范围内。集成模型通常比单一模型更准确。
                        
                        #### Q: 为什么有些材料的预测结果不准确？
                        A: 以下因素可能导致预测不准确：
                        - 材料结构复杂或罕见，训练数据中缺少类似样本
                        - 输入参数不准确
                        - 材料存在特殊电子结构，如强关联效应
                        
                        #### Q: 如何提高预测准确性？
                        A: 尝试以下方法：
                        - 使用集成模型进行预测
                        - 确保输入参数准确
                        - 对于复杂材料，考虑使用更专业的计算方法
                        """)
                        
                    # 添加参考文献
                    with gr.Accordion("参考文献", open=False):
                        gr.Markdown("""
                        ### 参考文献
                        
                        1. Ward, L., Agrawal, A., Choudhary, A., & Wolverton, C. (2016). A general-purpose machine learning framework for predicting properties of inorganic materials. *npj Computational Materials*, 2, 16028.
                        
                        2. Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D., Dacek, S., ... & Persson, K. A. (2013). Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. *APL Materials*, 1(1), 011002.
                        
                        3. Isayev, O., Oses, C., Toher, C., Gossett, E., Curtarolo, S., & Tropsha, A. (2017). Universal fragment descriptors for predicting properties of inorganic crystals. *Nature Communications*, 8, 15679.
                        """)
                        
                    # 添加版本历史
                    with gr.Accordion("版本历史", open=False):
                        gr.Markdown("""
                        ### 版本历史
                        
                        #### v1.1 (当前版本)
                        - 改进用户界面
                        - 添加模型性能评估功能
                        - 优化预测算法
                        
                        #### v1.0
                        - 初始版本
                        - 基本预测功能
                        - 单一模型支持
                        """)

        # 设置提交函数 - 修改为包含指示器更新
        predict_btn.click(
            fn=predict_material,
            inputs=[elements_input, nelements_input,
                    formation_energy_input, use_ensemble],
            outputs=[band_gap_output, material_class_output, details_output]
        ).then(
            fn=lambda band_gap: update_indicator(band_gap),
            inputs=[band_gap_output],
            outputs=[indicator_html]
        )

        # 设置批量预测功能
        batch_predict_btn.click(
            fn=lambda file, use_ensemble: batch_predict(file.name if file else None, use_ensemble),
            inputs=[file_input, batch_use_ensemble],
            outputs=[batch_results, batch_status]
        ).then(
            fn=lambda df: [gr.update(visible=True), None] if df is not None else [gr.update(visible=False), None],
            inputs=[batch_results],
            outputs=[batch_results_container, download_link]
        )
        
        # 设置下载结果功能
        download_btn.click(
            fn=download_batch_results,
            inputs=[batch_results],
            outputs=[download_link]
        ).then(
            fn=lambda link: gr.update(visible=True, value=f"""
                <a href="{link}" download="batch_prediction_results.csv" 
                   style="display: block; text-align: center; padding: 10px; background-color: #f0f9ff; 
                          border-radius: 5px; text-decoration: none; color: #2563eb; font-weight: 500;">
                    ⬇️ 点击此处下载预测结果
                </a>
                """) if link else gr.update(visible=False),
            inputs=[download_link],
            outputs=[download_link]
        )
        
        # 设置下载示例CSV功能
        download_example_btn.click(
            fn=get_example_csv,
            inputs=[],
            outputs=[example_download_link]
        ).then(
            fn=lambda link: gr.update(visible=True, value=link),
            inputs=[example_download_link],
            outputs=[example_download_link]
        )

        # 设置绘图函数
        plot_btn.click(
            fn=lambda model_type, regenerate: plot_actual_vs_predicted(
                model_type == "集成模型", regenerate),
            inputs=[model_type, regenerate_data],
            outputs=[plot_output, stats_output, data_details, status_output]
        )

        # 设置清除测试数据函数
        clear_btn.click(
            fn=clear_test_data,
            inputs=[model_type],
            outputs=[status_output]
        )

        # 添加JavaScript代码，实现带隙指示器的动态更新
        gr.HTML("""
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // 每隔一段时间检查并更新指示器
                setInterval(function() {
                    updateBandGapIndicator();
                }, 500);
                
                // 更新带隙指示器位置和值
                function updateBandGapIndicator() {
                    // 尝试不同的选择器方式找到元素
                    const bandGapOutput = document.querySelector('[id="band-gap-output"] textarea') || 
                                        document.querySelector('[id="band-gap-output"]') ||
                                        document.querySelector('.gradio-textbox[id="band-gap-output"] textarea');
                    
                    if (!bandGapOutput) return;
                    
                    const bandGapText = bandGapOutput.value || bandGapOutput.textContent;
                    if (!bandGapText) return;
                    
                    const bandGapMatch = bandGapText.match(/(\d+\.\d+)/);
                    
                    if (bandGapMatch && bandGapMatch[1]) {
                        const bandGap = parseFloat(bandGapMatch[1]);
                        console.log("找到带隙值:", bandGap);
                        
                        // 尝试不同的选择器找到指示器
                        const indicator = document.getElementById('band-gap-indicator') || 
                                         document.querySelector('#band-gap-indicator') ||
                                         document.querySelector('[id="band-gap-indicator"]');
                        
                        if (indicator) {
                            console.log("找到指示器元素");
                            // 计算指示器位置 (0-8 eV范围)
                            const maxBandGap = 8.0;
                            let position = (bandGap / maxBandGap) * 100;
                            position = Math.min(Math.max(position, 0), 100);
                            
                            // 更新指示器位置和值
                            indicator.style.left = position + '%';
                            const valueElement = indicator.querySelector('div:last-child');
                            if (valueElement) {
                                valueElement.textContent = bandGap.toFixed(2) + ' eV';
                                console.log("更新指示器值为:", bandGap.toFixed(2));
                            }
                            
                            // 显示指示器
                            const visualizationRow = document.querySelector('.visualization-row');
                            if (visualizationRow) {
                                visualizationRow.style.display = 'block';
                                console.log("显示可视化行");
                            }
                        } else {
                            console.log("未找到指示器元素");
                        }
                    }
                }
                
                // 获取预测按钮
                const predictBtn = document.getElementById('predict-btn');
                const resultStatus = document.querySelector('[id="result-status"]');
                
                if (predictBtn && resultStatus) {
                    // 点击预测按钮时显示加载状态
                    predictBtn.addEventListener('click', function() {
                        resultStatus.innerHTML = '<div style="text-align: center; padding: 20px;"><div class="loading"></div><p style="color: #666; margin-top: 10px;">正在预测中，请稍候...</p></div>';
                        
                        // 预测按钮点击后，等待一段时间再尝试更新指示器
                        setTimeout(updateBandGapIndicator, 1000);
                        setTimeout(updateBandGapIndicator, 2000);
                        setTimeout(updateBandGapIndicator, 3000);
                    });
                }
                
                // 获取绘图按钮
                const plotBtn = document.getElementById('plot-btn');
                const plotStatus = document.querySelector('#plot-status');
                
                if (plotBtn && plotStatus) {
                    // 点击绘图按钮时显示加载状态
                    plotBtn.addEventListener('click', function() {
                        plotStatus.innerHTML = '<div style="text-align: center; padding: 20px;"><div class="loading"></div><p style="color: #666; margin-top: 10px;">正在生成拟合曲线，请稍候...</p></div>';
                    });
                }
            });
        </script>
        """)

        # 添加加载状态的JavaScript
        gr.HTML("""
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // 获取批量预测按钮
                const batchPredictBtn = document.querySelector('button:contains("批量预测")');
                const batchStatus = document.querySelector('[label="处理状态"]');
                
                setTimeout(function() {
                    const allButtons = document.querySelectorAll('button');
                    let batchBtn = null;
                    
                    for (let i = 0; i < allButtons.length; i++) {
                        if (allButtons[i].textContent.includes('批量预测')) {
                            batchBtn = allButtons[i];
                            break;
                        }
                    }
                    
                    const statusElem = document.querySelector('[aria-label="处理状态"]');
                    
                    if (batchBtn && statusElem) {
                        batchBtn.addEventListener('click', function() {
                            statusElem.innerHTML = '<div style="text-align: center; padding: 20px;"><div class="loading"></div><p style="color: #666; margin-top: 10px;">正在处理批量预测，请稍候...</p></div>';
                        });
                    }
                }, 1000);
            });
        </script>
        """)
        
        # 添加覆盖样式，移除Radio和Checkbox样式和动画
        gr.HTML("""
        <style>
            /* 删除Radio样式 */
            .model-type-selector .gradio-radio,
            .gradio-radio {
                display: block !important;
            }
            
            .model-type-selector .gradio-radio label,
            .gradio-radio label {
                display: inline !important;
                align-items: normal !important;
                padding: 0 !important;
                margin: 0 !important;
                border: none !important;
                background-color: transparent !important;
                color: inherit !important;
                font-weight: normal !important;
                transition: none !important;
                animation: none !important;
                transform: none !important;
                border-radius: 0 !important;
                box-shadow: none !important;
            }
            
            .model-type-selector .gradio-radio input:checked + label,
            .gradio-radio input:checked + label {
                background-color: transparent !important;
                color: inherit !important;
                border: none !important;
                font-weight: normal !important;
            }
            
            /* 删除Radio的动画效果 */
            .gradio-radio * {
                transition: none !important;
                animation: none !important;
                transform: none !important;
            }
            
            /* 恢复原生Radio样式 */
            .gradio-radio input[type="radio"] {
                -webkit-appearance: radio !important;
                -moz-appearance: radio !important;
                appearance: radio !important;
                opacity: 1 !important;
                position: static !important;
                width: auto !important;
                height: auto !important;
                margin-right: 5px !important;
                display: inline-block !important;
            }
            
            /* 删除Checkbox样式 */
            .regenerate-checkbox,
            .gradio-checkbox {
                display: block !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            
            .regenerate-checkbox label,
            .gradio-checkbox label {
                display: inline !important;
                align-items: normal !important;
                padding: 0 !important;
                margin: 0 !important;
                border: none !important;
                background-color: transparent !important;
                color: inherit !important;
                font-weight: normal !important;
                transition: none !important;
                animation: none !important;
                transform: none !important;
                border-radius: 0 !important;
                box-shadow: none !important;
            }
            
            /* 删除Checkbox的动画效果 */
            .gradio-checkbox * {
                transition: none !important;
                animation: none !important;
                transform: none !important;
            }
            
            /* 恢复原生Checkbox样式 */
            .gradio-checkbox input[type="checkbox"] {
                -webkit-appearance: checkbox !important;
                -moz-appearance: checkbox !important;
                appearance: checkbox !important;
                opacity: 1 !important;
                position: static !important;
                width: auto !important;
                height: auto !important;
                margin-right: 5px !important;
                display: inline-block !important;
            }
            
            /* 删除所有控件的自定义样式 */
            .model-type-selector, .regenerate-checkbox {
                padding: 0 !important;
                margin: 0 !important;
                border: none !important;
                background: none !important;
                box-shadow: none !important;
            }
            
            /* 页脚样式 - 确保居中显示在底部 */
            #footer-container {
                width: 100%;
                text-align: center;
                margin-bottom: 20px;
            }
            
            .footer {
                display: inline-block;
                text-align: center;
                max-width: 1200px;
                width: 90%;
                margin: 0 auto;
            }
            
            /* 页眉样式 */
            .header-container {
                margin-bottom: 2rem;
                padding: 1.5rem 0;
                background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary-color) 100%);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
            }
            
            .header {
                text-align: center;
                width: 100%;
            }
            
            .header h1 {
                margin-bottom: 0.5rem;
                color: white !important;
                text-align: center;
                color: rgba(136, 48, 48, 0.9) !important;
                text-shadow: 0 2px 4px rgba(62, 179, 159, 0.2);
                font-size: 2.5rem !important;
                font-weight: 700 !important;
            }
            
            .subtitle {
                color: rgba(136, 48, 48, 0.9) !important;
                font-size: 1.1rem;
                font-weight: 500;
                margin-top: 0;
            }
        </style>
        """)
        
        # 添加页脚到界面底部
        gr.HTML("""
        <div id="footer-container">
            <div class="footer">
                <div class="footer-content">
                    <div class="footer-section">
                        <h4>材料带隙预测系统</h4>
                        <p>版本 1.1 | 基于深度学习的材料科学工具</p>
                    </div>
                    <div class="footer-section">
                        <h4>关于</h4>
                        <p>© 2025 张昊峥测试项目</p>
                    </div>
                    <div class="footer-section">
                        <h4>联系方式</h4>
                        <p>邮箱: 15855147102@163.com</p>
                    </div>
                </div>
                <div class="footer-bottom">
                    <p>使用深度学习算法预测无机材料的带隙值</p>
                </div>
            </div>
        </div>
        """)

    # 返回Gradio界面实例
    return demo

# 启动Web界面
if __name__ == "__main__":
    demo = create_interface()
    
    # 启动Gradio界面，不再使用不支持的footer参数
    demo.launch(share=True)

