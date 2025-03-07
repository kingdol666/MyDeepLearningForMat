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
        details += f"材料类型: {material_class}"

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

        # 取一部分数据作为测试集 (随机30个样本)
        # 对于强制重新生成，使用不同的随机种子以获得不同样本
        random_seed = 42 if not force_regenerate else int(time.time()) % 1000
        np.random.seed(random_seed)
        print(f"使用随机种子 {random_seed} 生成测试样本")
        test_indices = np.random.choice(
            len(materials_df), min(30, len(materials_df)), replace=False)
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

# 创建Gradio界面


def create_interface():
    # 检查模型文件
    model_status = check_model_files()

    # 简单的CSS样式
    css = """
    h1 {
        text-align: center;
        color: #2C3E50;
        margin-bottom: 0.5em;
        background: linear-gradient(90deg, #3498DB, #2980B9);
        padding: 10px;
        border-radius: 8px;
        color: white;
    }
    h2, h3 {
        color: #2C3E50;
        border-left: 4px solid #3498DB;
        padding-left: 8px;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        padding-top: 10px;
        border-top: 1px solid #eee;
        color: #7F8C8D;
        font-size: 0.9em;
    }
    .gradio-button {
        background-color: #3498DB !important;
    }
    .model-status {
        background-color: #EBF5FB;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #3498DB;
    }
    .img-container img {
        max-width: 100%;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    """

    # 定义输入组件
    with gr.Blocks(title="材料带隙预测系统", css=css) as demo:
        gr.HTML('<h1>🔬 材料带隙预测系统</h1>')
        gr.Markdown("<p style='text-align:center'>基于深度学习的无机材料带隙预测工具</p>")

        # 模型状态信息
        with gr.Column():
            # 修复f-string中使用反斜杠的问题
            formatted_status = model_status.replace("✅", "✅ ")
            formatted_status = formatted_status.replace("⚠️", "⚠️ ")
            formatted_status = formatted_status.replace("ℹ️", "ℹ️ ")
            formatted_status = formatted_status.replace("\n", "<br>")
            gr.HTML(
                f"<div class='model-status'><h3>系统状态</h3>{formatted_status}</div>")

        with gr.Tabs():
            with gr.Tab("🔍 单材料预测"):
                with gr.Row():
                    with gr.Column():
                        # 输入组件
                        gr.Markdown("### 输入参数")

                        elements_input = gr.Textbox(
                            label="元素组成",
                            placeholder="输入元素符号，用逗号分隔 (例如: Si,O)"
                        )

                        with gr.Row():
                            nelements_input = gr.Number(
                                label="元素数量",
                                value=2,
                                minimum=1,
                                maximum=10,
                                step=1
                            )

                            formation_energy_input = gr.Number(
                                label="形成能 (eV/atom)",
                                value=-3.0,
                                minimum=-10,
                                maximum=10
                            )

                        use_ensemble = gr.Checkbox(
                            label="使用集成模型 (如果可用)",
                            value=False
                        )

                        predict_btn = gr.Button("🔮 预测带隙", variant="primary")

                    with gr.Column():
                        # 输出组件
                        gr.Markdown("### 预测结果")

                        band_gap_output = gr.Textbox(label="预测带隙")
                        material_class_output = gr.Textbox(label="材料分类")
                        details_output = gr.Textbox(label="详细信息", lines=6)

                # 样本数据
                gr.Markdown("### 示例材料")
                gr.Markdown("点击下方示例快速测试预测功能:")

                sample_data = generate_samples()
                gr.Examples(
                    examples=sample_data,
                    inputs=[elements_input, nelements_input,
                            formation_energy_input],
                    outputs=[band_gap_output,
                             material_class_output, details_output],
                    fn=lambda e, n, f: predict_material(e, n, f, False)
                )

            with gr.Tab("📊 批量预测"):
                with gr.Column():
                    gr.Markdown("### 批量预测功能 (开发中)")

                    file_input = gr.File(
                        label="上传CSV文件 (功能开发中)",
                        file_types=[".csv"]
                    )

                    gr.Markdown("""
                    #### 批量预测说明
                    未来版本将支持上传CSV文件进行批量预测。CSV文件应包含以下列：
                    - `elements`: 元素组成，用逗号分隔
                    - `nelements`: 元素数量
                    - `formation_energy`: 形成能 (eV/atom)
                    
                    预测结果将以CSV文件形式返回，包含原始数据和预测的带隙值。
                    """)

                    batch_predict_btn = gr.Button(
                        "批量预测 (即将推出)", interactive=False)

            # 新增拟合曲线选项卡
            with gr.Tab("📈 模型性能评估"):
                with gr.Column():
                    gr.Markdown("### 模型预测性能评估")
                    gr.Markdown("分析实际带隙值与预测带隙值的拟合情况，评估模型性能。")

                    with gr.Row():
                        # 选择使用哪种模型
                        model_type = gr.Radio(
                            label="选择评估的模型类型",
                            choices=["单一模型", "集成模型"],
                            value="单一模型"
                        )

                        # 添加重新生成数据的选项
                        regenerate_data = gr.Checkbox(
                            label="重新生成测试数据",
                            value=False,
                            info="选中此项将强制重新生成测试数据，会覆盖现有结果"
                        )

                    with gr.Row():
                        # 生成拟合曲线按钮
                        plot_btn = gr.Button(
                            "生成拟合曲线", variant="primary", scale=3)

                        # 清除测试数据按钮
                        clear_btn = gr.Button(
                            "清除测试数据", variant="secondary", scale=1)

                    # 显示处理状态信息
                    status_output = gr.HTML(label="状态信息")

                    # 显示拟合曲线图像
                    with gr.Column():
                        plot_output = gr.HTML(label="拟合曲线")
                        stats_output = gr.HTML(label="性能统计")

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
                        data_details = gr.DataFrame(
                            headers=["材料成分", "元素数量", "形成能",
                                     "实际带隙", "预测带隙", "绝对误差"],
                            type="pandas"
                        )

            with gr.Tab("ℹ️ 使用说明"):
                gr.Markdown("""
                ### 使用指南
                
                #### 基本操作
                1. 在"单材料预测"选项卡中输入材料信息
                2. 点击"预测带隙"按钮获取结果
                3. 结果将显示预测的带隙值、材料分类和详细信息
                
                #### 输入参数说明
                - **元素组成**: 输入构成材料的元素符号，用逗号分隔（例如：Si,O 表示二氧化硅）
                - **元素数量**: 材料中不同元素的数量
                - **形成能**: 材料的形成能，单位为eV/atom
                - **使用集成模型**: 如果可用，使用多个模型的集成进行预测，通常可提高准确性
                
                #### 材料分类标准
                - 🔵 **金属/导体**: 带隙 < 0.1 eV
                - 🟢 **半导体**: 带隙 0.1-3.0 eV
                - 🟠 **绝缘体**: 带隙 > 3.0 eV
                
                #### 注意事项
                - 确保输入的元素符号正确（例如：Fe而不是FE）
                - 形成能通常为负值，表示材料的稳定性
                - 如果预测结果不准确，可尝试使用集成模型
                
                #### 模型性能评估
                - 使用"模型性能评估"选项卡可查看模型预测的准确性
                - 拟合曲线显示了预测值与实际值的对比
                - R²值越接近1，表示模型预测性能越好
                """)

        # 版本信息
        gr.HTML("""
        <div class="footer">
            <p>材料带隙预测系统 v1.1 | 基于深度学习的材料科学工具</p>
            <p>© 2025 张昊峥测试项目</p>
        </div>
        """)

        # 设置提交函数 - 保持原功能不变
        predict_btn.click(
            fn=predict_material,
            inputs=[elements_input, nelements_input,
                    formation_energy_input, use_ensemble],
            outputs=[band_gap_output, material_class_output, details_output]
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

    return demo


# 启动Web界面
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)
