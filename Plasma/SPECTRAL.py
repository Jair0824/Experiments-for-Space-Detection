import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'


SPECTRAL_LINES = {
    'Ar I 750.4 nm': (750.39, 1.0),
    'Ar I 763.5 nm': (763.51, 1.0),
    'Ar I 811.5 nm': (811.53, 1.0),
    'Ar I 842.5 nm': (842.46, 1.0),
}


def load_spectrum_data(filename):
    """从XLSX文件加载光谱数据，跳过前6行头文件。"""
    if not os.path.exists(filename):
        print(f"警告：文件 '{filename}' 不存在，将跳过。")
        return None
    try:
        df = pd.read_excel(
            filename, skiprows=6, header=None, names=['Wavelength', 'Intensity']
        )
        df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
        df['Intensity'] = pd.to_numeric(df['Intensity'], errors='coerce')
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"读取或处理文件 {filename} 时出错: {e}")
        return None


print("开始处理光谱数据 (带本底扣除)...")

# --- 加载本底数据 ---
background_filename = '4.21光谱探测本底.xlsx'
print(f"正在加载本底文件: {background_filename}...")
background_df = load_spectrum_data(background_filename)
if background_df is None or background_df.empty:
    print(f"错误：无法加载本底数据文件 '{background_filename}'，程序终止。")
    exit()

# --- 查找所有流量数据文件 ---
file_pattern = '[0-9]*.xlsx'
file_paths = glob.glob(file_pattern)

# --- 从待处理列表中移除本底文件 ---
if background_filename in file_paths:
    file_paths.remove(background_filename)
    print(f"已将本底文件 '{background_filename}' 从待处理列表中移除。")

if not file_paths:
    print(f"错误：未在当前目录下找到任何流量数据文件 (匹配模式: {file_pattern})。")
    exit()

print(f"找到 {len(file_paths)} 个流量数据文件，开始处理...")

results = {}
plt.figure(figsize=(12, 7))

for file_path in sorted(file_paths):
    try:
        flow_rate_str = os.path.basename(file_path).split('.')[0]
        flow_rate = int(flow_rate_str)
    except (IndexError, ValueError):
        print(f"无法从文件名 {file_path} 中解析流量。跳过此文件。")
        continue

    print(f"正在处理文件: {file_path} (流量: {flow_rate} sccm)...")

    spectrum_df = load_spectrum_data(file_path)
    if spectrum_df is None or spectrum_df.empty:
        continue

    # --- 插值并扣除本底 ---
    bg_intensity_interp = np.interp(
        spectrum_df['Wavelength'],
        background_df['Wavelength'],
        background_df['Intensity'],
    )
    spectrum_df['Intensity'] = spectrum_df['Intensity'] - bg_intensity_interp
    spectrum_df.loc[spectrum_df['Intensity'] < 0, 'Intensity'] = 0

    plt.plot(
        spectrum_df['Wavelength'],
        spectrum_df['Intensity'],
        label=f'{flow_rate} sccm',
        alpha=0.8,
    )

    results[flow_rate] = {}
    for line_name, (center_wl, window) in SPECTRAL_LINES.items():
        line_df = spectrum_df[
            (spectrum_df['Wavelength'] >= center_wl - window)
            & (spectrum_df['Wavelength'] <= center_wl + window)
        ]

        if not line_df.empty:
            peak_intensity = line_df['Intensity'].max()
            results[flow_rate][line_name] = peak_intensity
        else:
            results[flow_rate][line_name] = np.nan


plt.title('Background-Subtracted Spectra under Different Argon Flow Rates', fontsize=16)
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Intensity (a.u.)', fontsize=12)
plt.legend(title='Flow Rate', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('spectra_overlay_corrected.pdf', bbox_inches='tight')
plt.close()
print("\n图表 'spectra_overlay_corrected.pdf' 已保存。")

if not results:
    print("未能处理任何数据，无法生成谱线强度vs流量图。")
else:
    plot_data = defaultdict(lambda: {'flows': [], 'intensities': []})
    sorted_flows = sorted(results.keys())
    for flow in sorted_flows:
        for line_name, intensity in results[flow].items():
            plot_data[line_name]['flows'].append(flow)
            plot_data[line_name]['intensities'].append(intensity)

    plt.figure(figsize=(10, 7))

    for line_name, data in plot_data.items():
        plt.plot(
            data['flows'],
            data['intensities'],
            marker='o',
            linestyle='-',
            label=line_name,
        )

    plt.title('Argon Spectral Line Intensity vs. Flow Rate', fontsize=16)
    plt.xlabel('Argon Flow Rate (sccm)', fontsize=12)
    plt.ylabel('Peak Intensity (a.u.)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(sorted_flows)

    plt.savefig('line_intensity_vs_flow_rate.pdf', bbox_inches='tight')
    plt.close()
    print("图表 'line_intensity_vs_flow_rate.pdf' 已保存。")

print("\n数据处理完成。")
