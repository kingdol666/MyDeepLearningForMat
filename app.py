#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
材料带隙预测应用
==============
此应用程序使用MaterialPredictor接口预测材料的带隙值
"""

import os
import sys
import time
import csv
import pandas as pd
import subprocess
from datetime import datetime

# 尝试导入MaterialPredictor
try:
    from MaterialPredictor import MaterialPredictor, predict_band_gap
except ImportError as e:
    print(f"错误: 无法导入MaterialPredictor模块: {e}")
    print("请确保MaterialPredictor.py文件位于当前目录")
    sys.exit(1)

# 定义输出目录
RESULTS_DIR = "prediction_results"


def ensure_dirs_exist():
    """确保必要的目录存在"""
    try:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
            print(f"已创建结果目录: {RESULTS_DIR}")
        return True
    except Exception as e:
        print(f"创建目录时出错: {str(e)}")
        return False


def classify_material(band_gap):
    """根据带隙值对材料进行分类"""
    if band_gap is None:
        return "未知"
    elif band_gap <= 0.05:
        return "金属/导体"
    elif band_gap <= 1.5:
        return "窄带隙半导体"
    elif band_gap <= 3.0:
        return "半导体"
    elif band_gap <= 6.0:
        return "绝缘体"
    else:
        return "宽带隙绝缘体"


def get_material_description(band_gap):
    """根据带隙提供材料的基本描述"""
    if band_gap is None:
        return "无法确定材料特性"

    material_type = classify_material(band_gap)

    descriptions = {
        "金属/导体": "电导率高，能够自由传导电子，常用于电子元件的导线和电极",
        "窄带隙半导体": "在室温下具有一定的电导率，对红外光敏感，适用于红外探测器和热电材料",
        "半导体": "带隙适中，是电子器件(如晶体管、太阳能电池、LED)的基础材料",
        "绝缘体": "电导率低，常用于电子设备的绝缘层和保护层",
        "宽带隙绝缘体": "优异的绝缘性能，在高温和高电场下仍能保持稳定"
    }

    return descriptions.get(material_type, "未知特性")


def get_applications(band_gap):
    """根据带隙提供材料的潜在应用"""
    if band_gap is None:
        return "无法确定潜在应用"

    if band_gap <= 0.05:
        return "导线、电极、接触材料、电磁屏蔽"
    elif band_gap <= 1.0:
        return "红外探测器、热电材料、太阳能电池、低频光电器件"
    elif band_gap <= 2.0:
        return "太阳能电池、可见光探测器、光电二极管、发光二极管"
    elif band_gap <= 3.5:
        return "蓝色/紫外LED、紫外探测器、高频电子器件、电力电子器件"
    else:
        return "高压电子设备、高温电子设备、紫外光电器件、介电材料"


def save_prediction(material_name, nelements, formation_energy, elements, band_gap, material_type):
    """保存预测结果到文件"""
    # 确保结果目录存在
    if not os.path.exists(RESULTS_DIR):
        try:
            os.makedirs(RESULTS_DIR)
        except Exception as e:
            print(f"创建结果目录时出错: {str(e)}")
            return False

    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 定义文件名
    if material_name:
        filename = f"{RESULTS_DIR}/{material_name}_{timestamp}.txt"
    else:
        filename = f"{RESULTS_DIR}/prediction_{timestamp}.txt"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("材料带隙预测结果\n")
            f.write("="*40 + "\n")
            if material_name:
                f.write(f"材料名称: {material_name}\n")
            f.write(f"元素数量: {nelements}\n")
            f.write(f"形成能: {formation_energy} eV/atom\n")
            f.write(f"元素组成: {elements}\n")
            f.write("-"*40 + "\n")
            f.write(f"预测带隙: {band_gap:.2f} eV\n")
            f.write(f"材料类型: {material_type}\n")
            f.write("-"*40 + "\n")
            f.write("材料特性:\n")
            f.write(get_material_description(band_gap) + "\n\n")
            f.write("潜在应用:\n")
            f.write(get_applications(band_gap) + "\n")
            f.write("="*40 + "\n")
            f.write(f"预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"预测结果已保存到 {filename}")
        return True
    except Exception as e:
        print(f"保存预测结果时出错: {str(e)}")
        return False


def predict_single():
    """单一材料预测界面"""
    print("\n" + "="*50)
    print("单一材料带隙预测".center(40))
    print("="*50)

    material_name = input("材料名称 (可选): ").strip()

    # 获取元素数量
    while True:
        try:
            nelements = int(input("元素数量: ").strip())
            if nelements <= 0:
                print("元素数量必须大于0")
                continue
            break
        except ValueError:
            print("请输入有效的数字")

    # 获取形成能
    while True:
        try:
            formation_energy = float(input("形成能 (eV/atom): ").strip())
            break
        except ValueError:
            print("请输入有效的数字")

    # 获取元素组成
    while True:
        elements = input("元素组成 (用逗号分隔，例如 Si,O): ").strip()
        if not elements:
            print("请输入至少一个元素")
            continue
        break

    # 检查是否存在集成模型
    ensemble_dir = os.path.join("dataset", "ensemble")
    has_ensemble = (os.path.exists(ensemble_dir) and
                    any(f.startswith("model_") and f.endswith(".pth")
                        for f in os.listdir(ensemble_dir)))

    # 询问是否使用集成模型
    use_ensemble = False
    if has_ensemble:
        ensemble_choice = input("是否使用集成模型进行预测 (提高精度) (是/否): ").strip().lower()
        use_ensemble = ensemble_choice in ['是', 'y', 'yes']
    else:
        print("注意: 集成模型不可用，将使用单一模型进行预测")

    print("\n正在预测...")
    try:
        # 创建预测器实例，传递集成模型选择
        predictor = MaterialPredictor(use_ensemble=use_ensemble)

        # 检查预测器是否初始化成功
        if predictor.model is None or predictor.scaler is None:
            print("错误: 模型初始化失败，请检查模型文件是否存在")
            return

        # 进行预测
        band_gap = predictor.predict(nelements, formation_energy, elements)

        if band_gap is not None:
            # 根据带隙值确定材料类型
            if band_gap <= 0.1:
                material_type = "金属或半金属"
            elif band_gap <= 1.5:
                material_type = "窄带隙半导体"
            elif band_gap <= 3.0:
                material_type = "中等带隙半导体"
            else:
                material_type = "宽带隙半导体/绝缘体"

            print("\n" + "-"*50)
            print(f"预测带隙: {band_gap:.2f} eV")
            print(f"材料类型: {material_type}")
            print(f"使用的模型: {'集成神经网络' if predictor.is_ensemble else '标准神经网络'}")
            print("-"*50)
            print("材料特性:")
            print(get_material_description(band_gap))
            print("\n潜在应用:")
            print(get_applications(band_gap))
            print("-"*50)

            # 询问是否保存结果
            save_result = input("\n是否保存预测结果? (是/否): ").strip().lower()
            if save_result in ['是', 'y', 'yes']:
                save_prediction(
                    material_name, nelements, formation_energy, elements, band_gap, material_type)
        else:
            print("预测失败，请确保输入参数和模型正确")
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        import traceback
        print(traceback.format_exc())


def batch_predict():
    """批量预测功能"""
    print("\n" + "="*50)
    print("批量材料带隙预测".center(40))
    print("="*50)

    # 获取输入文件
    while True:
        input_file = input("输入CSV文件路径: ").strip()
        if not os.path.exists(input_file):
            print(f"文件不存在: {input_file}")
            retry = input("是否重新输入文件路径? (是/否): ").strip().lower()
            if retry not in ['是', 'y', 'yes']:
                return
        else:
            break

    # 检查是否存在集成模型
    ensemble_dir = os.path.join("dataset", "ensemble")
    has_ensemble = (os.path.exists(ensemble_dir) and
                    any(f.startswith("model_") and f.endswith(".pth")
                        for f in os.listdir(ensemble_dir)))

    # 询问是否使用集成模型
    use_ensemble = False
    if has_ensemble:
        ensemble_choice = input("是否使用集成模型进行预测 (提高精度) (是/否): ").strip().lower()
        use_ensemble = ensemble_choice in ['是', 'y', 'yes']
    else:
        print("注意: 集成模型不可用，将使用单一模型进行预测")

    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)

        required_columns = ['nelements', 'formation_energy']
        missing_columns = [
            col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"错误: 输入文件缺少必要的列: {', '.join(missing_columns)}")
            print("CSV文件必须包含 nelements 和 formation_energy 列")
            print("可选列: elements, material_name/id")
            return

        # 检查元素列
        has_elements = 'elements' in df.columns
        if not has_elements:
            print("警告: 未找到elements列，预测将仅基于元素数量和形成能")
            print("这可能降低预测精度")
            proceed = input("是否继续? (是/否): ").strip().lower()
            if proceed not in ['是', 'y', 'yes']:
                return

        # 创建输出文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            RESULTS_DIR, f"batch_prediction_{timestamp}.csv")

        # 确保结果目录存在
        ensure_dirs_exist()

        # 创建预测器实例
        predictor = MaterialPredictor(use_ensemble=use_ensemble)

        if predictor.model is None or predictor.scaler is None:
            print("错误: 模型初始化失败，请检查模型文件是否存在")
            return

        # 准备结果列表
        results = []
        total_rows = len(df)
        print(f"\n开始批量预测 {total_rows} 条数据...")

        # 显示使用的模型类型
        print(f"使用的模型: {'集成神经网络' if predictor.is_ensemble else '标准神经网络'}")

        # 进度更新频率
        update_freq = max(1, total_rows // 20)  # 至少每20步更新一次

        # 计时
        start_time = time.time()

        # 逐行预测
        for i, row in df.iterrows():
            try:
                # 获取必要参数
                nelements = int(row['nelements'])
                formation_energy = float(row['formation_energy'])

                # 获取元素（如果存在）
                elements = row['elements'] if has_elements and 'elements' in row and pd.notna(
                    row['elements']) else None

                # 获取材料名称/ID（如果存在）
                material_name = None
                for name_col in ['material_name', 'name', 'material_id', 'id', 'formula']:
                    if name_col in df.columns and pd.notna(row[name_col]):
                        material_name = str(row[name_col])
                        break

                # 预测带隙
                band_gap = predictor.predict(
                    nelements, formation_energy, elements)

                # 确定材料类型
                if band_gap is not None:
                    if band_gap <= 0.1:
                        material_type = "金属或半金属"
                    elif band_gap <= 1.5:
                        material_type = "窄带隙半导体"
                    elif band_gap <= 3.0:
                        material_type = "中等带隙半导体"
                    else:
                        material_type = "宽带隙半导体/绝缘体"
                else:
                    material_type = "预测失败"
                    band_gap = float('nan')

                # 添加到结果
                result = {
                    'material_name': material_name if material_name else f"Material_{i+1}",
                    'nelements': nelements,
                    'formation_energy': formation_energy,
                    'elements': elements if elements else '',
                    'predicted_band_gap': band_gap,
                    'material_type': material_type
                }
                results.append(result)

                # 更新进度
                if (i + 1) % update_freq == 0 or i + 1 == total_rows:
                    elapsed = time.time() - start_time
                    progress = (i + 1) / total_rows * 100
                    print(
                        f"进度: {progress:.1f}% ({i+1}/{total_rows}), 耗时: {elapsed:.1f}秒")

            except Exception as e:
                print(f"处理第 {i+1} 行时出错: {str(e)}")
                continue

        # 保存结果
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_file, index=False)
            print(f"\n预测完成，结果已保存到: {output_file}")
            print(f"总计 {len(results)} 条预测结果，总耗时: {time.time() - start_time:.1f}秒")

            # 显示统计信息
            print("\n预测结果统计:")
            success_count = results_df['predicted_band_gap'].notna().sum()
            print(
                f"成功预测: {success_count}/{len(results_df)} ({success_count/len(results_df)*100:.1f}%)")

            if success_count > 0:
                print(
                    f"平均带隙: {results_df['predicted_band_gap'].mean():.2f} eV")
                print(
                    f"带隙范围: {results_df['predicted_band_gap'].min():.2f} - {results_df['predicted_band_gap'].max():.2f} eV")

                # 材料类型分布
                type_counts = results_df['material_type'].value_counts()
                print("\n材料类型分布:")
                for type_name, count in type_counts.items():
                    print(
                        f"{type_name}: {count} ({count/len(results_df)*100:.1f}%)")
        else:
            print("没有生成任何预测结果")

    except Exception as e:
        print(f"批量预测过程中出错: {str(e)}")
        import traceback
        print(traceback.format_exc())


def show_help():
    """显示帮助信息"""
    print("\n" + "="*50)
    print("材料带隙预测工具帮助".center(40))
    print("="*50)
    print("本工具用于预测材料的带隙值，基于材料的元素组成和形成能")
    print("\n主要功能:")
    print("1. 单一材料预测 - 输入单个材料的信息并获取带隙预测")
    print("2. 批量材料预测 - 通过CSV文件批量预测多个材料的带隙")
    print("3. 帮助信息     - 显示本帮助内容")
    print("4. 退出         - 退出程序")
    print("\n输入参数说明:")
    print("- 元素数量: 材料中不同元素的数量")
    print("- 形成能: 材料的形成能，单位为eV/atom")
    print("- 元素组成: 材料中的元素符号，用逗号分隔，如Si,O")
    print("\n批量预测CSV文件格式:")
    print("必须包含的列: nelements, formation_energy")
    print("可选列: material_name, elements")
    print("\n预测结果说明:")
    print("- 带隙值: 单位为电子伏特(eV)")
    print("- 材料类型: 根据带隙值分类")
    print("  * 0-0.05 eV: 金属/导体")
    print("  * 0.05-1.5 eV: 窄带隙半导体")
    print("  * 1.5-3.0 eV: 半导体")
    print("  * 3.0-6.0 eV: 绝缘体")
    print("  * >6.0 eV: 宽带隙绝缘体")
    print("="*50)


def check_model_initialization():
    """检查模型是否已初始化，若未初始化则提示训练"""
    model_dir = "dataset"
    model_file = os.path.join(model_dir, "model.pth")
    scaler_file = os.path.join(model_dir, "scaler.json")

    # 确保目录存在
    if not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir)
            print(f"已创建模型目录: {model_dir}")
        except Exception as e:
            print(f"创建目录时出错: {str(e)}")
            return False

    # 检查文件是否存在
    model_exists = os.path.exists(model_file)
    scaler_exists = os.path.exists(scaler_file)

    if not model_exists or not scaler_exists:
        print("\n警告: 模型文件或标准化器文件不存在")

        missing_files = []
        if not model_exists:
            missing_files.append(f"模型文件({model_file})")
        if not scaler_exists:
            missing_files.append(f"标准化器文件({scaler_file})")

        print(f"缺少的文件: {', '.join(missing_files)}")

        if input("\n是否训练新模型? (是/否): ").strip().lower() in ['是', 'y', 'yes']:
            print("\n正在启动模型训练过程...")
            print("="*60)
            print("提示: 模型训练可能需要一些时间，请耐心等待")
            print("="*60)

            try:
                # 检查MaterialDL.py是否存在
                if not os.path.exists("MaterialDL.py"):
                    print("错误: MaterialDL.py不存在，无法训练模型")
                    return False

                # 执行训练脚本
                subprocess.run([sys.executable, "MaterialDL.py"], check=True)

                # 再次检查文件是否存在
                if os.path.exists(model_file) and os.path.exists(scaler_file):
                    print("\n模型训练成功完成!")
                    return True
                else:
                    print("\n模型训练完成，但某些文件仍然缺失:")

                    still_missing = []
                    if not os.path.exists(model_file):
                        still_missing.append(f"模型文件({model_file})")
                    if not os.path.exists(scaler_file):
                        still_missing.append(f"标准化器文件({scaler_file})")

                    print(f"缺少的文件: {', '.join(still_missing)}")
                    print("请检查MaterialDL.py中的文件路径设置是否正确")

                    return False
            except Exception as e:
                print(f"\n训练模型时出错: {str(e)}")
                return False
        else:
            print("继续使用程序，但预测功能可能无法正常工作")
            return False

    print("模型文件检查完成，所有必要文件已存在")
    return True


def main():
    """主函数"""
    # 确保目录存在
    ensure_dirs_exist()

    # 检查模型初始化
    check_model_initialization()

    while True:
        print("\n" + "="*50)
        print("材料带隙预测工具".center(40))
        print("="*50)
        print("1. 单一材料预测")
        print("2. 批量材料预测")
        print("3. 帮助信息")
        print("4. 退出")
        print("="*50)

        choice = input("请选择功能 (1-4): ").strip()

        if choice == '1':
            predict_single()
        elif choice == '2':
            batch_predict()
        elif choice == '3':
            show_help()
        elif choice == '4':
            print("感谢使用材料带隙预测工具，再见!")
            break
        else:
            print("无效的选择，请重新输入")


if __name__ == "__main__":
    main()
