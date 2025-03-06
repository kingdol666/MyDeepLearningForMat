import gradio as gr
import os
import sys
import numpy as np
from MaterialPredictor import MaterialPredictor, predict_band_gap

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

    return demo


# 启动Web界面
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)
