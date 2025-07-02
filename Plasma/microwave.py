import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict


# 物理常数
e = 1.602176634e-19  # 基本电荷 (C)
m_e = 9.10938356e-31  # 电子质量 (kg)
epsilon_0 = 8.854187817e-12  # 真空介电常数 (F/m)
c = 299792458  # 光速 (m/s)


# 等离子体区域长度
L = 0.4  # (m)


experiment_setup = {
    '10sccm': {
        'flow_rate': 10,
        'background_files': ['000.CSV', '001.CSV', '002.CSV'],
        'conditions': {
            '15V': ['010.CSV', '011.CSV', '012.CSV'],
            '20V': ['020.CSV', '021.CSV', '022.CSV'],
            '25V': ['030.CSV', '031.CSV', '032.CSV'],
        },
    },
    '8sccm': {
        'flow_rate': 8,
        'background_files': ['100.CSV', '101.CSV', '102.CSV'],
        'conditions': {
            '15V': ['110.CSV', '111.CSV', '112.CSV'],
            '20V': ['120.CSV', '121.CSV', '122.CSV'],
            '25V': ['130.CSV', '131.CSV', '132.CSV'],
        },
    },
}


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'stix'  # 优化数学公式显示


def load_data(filename):
    """从CSV文件加载数据，使用第1和第2列，并进行清洗"""
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found. Skipping.")
        return None
    try:
        df = pd.read_csv(
            filename, header=None, usecols=[0, 1], names=['Frequency', 'Phase']
        )
        df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')
        df['Phase'] = pd.to_numeric(df['Phase'], errors='coerce')
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error reading or processing file {filename}: {e}")
        return None


def calculate_line_avg_density(phase_shift_rad, frequency, L):
    """根据相位移动和频率计算线平均电子密度。"""
    K = (4 * np.pi * c * epsilon_0 * m_e) / (e**2 * L)
    phase_freq_product = phase_shift_rad * frequency
    mean_product = np.mean(phase_freq_product)
    n_e = K * mean_product
    return n_e


plot_results = []

for series_name, series_data in experiment_setup.items():
    print(f"\n--- Processing series: {series_name} ---")

    all_bg_phases = []
    base_freq = None
    for bg_file in series_data['background_files']:
        bg_data = load_data(bg_file)
        if bg_data is not None:
            if base_freq is None:
                base_freq = bg_data['Frequency'].values
            interp_phase = np.interp(base_freq, bg_data['Frequency'], bg_data['Phase'])
            all_bg_phases.append(interp_phase)

    if not all_bg_phases:
        print(
            f"Error: Could not load any background files for {series_name}. Skipping series."
        )
        continue

    avg_background_phase = np.mean(all_bg_phases, axis=0)

    for condition_label, plasma_files in series_data['conditions'].items():
        print(f"  Processing condition: {condition_label}")

        repetition_densities = []
        for plasma_file in plasma_files:
            plasma_data = load_data(plasma_file)
            if plasma_data is None:
                continue

            interp_bg_phase = np.interp(
                plasma_data['Frequency'], base_freq, avg_background_phase
            )
            phase_shift_deg = plasma_data['Phase'] - interp_bg_phase
            phase_shift_rad = phase_shift_deg * np.pi / 180.0

            n_e = calculate_line_avg_density(
                phase_shift_rad.values, plasma_data['Frequency'].values, L
            )
            repetition_densities.append(n_e)

        if not repetition_densities:
            print(
                f"  Warning: All files for condition {condition_label} could not be processed."
            )
            continue

        mean_density = np.mean(repetition_densities)
        std_density = np.std(repetition_densities)

        voltage = int(condition_label.replace('V', ''))
        plot_results.append(
            {
                'flow_rate': series_data['flow_rate'],
                'voltage': voltage,
                'mean_ne': mean_density,
                'std_ne': std_density,
            }
        )


print("\n" + "=" * 50)
print("             计算结果汇总 (Calculated Results)")
print("=" * 50)

plot_results.sort(key=lambda x: (x['flow_rate'], x['voltage']))
for res in plot_results:
    print(f"工况: 流量 = {res['flow_rate']} sccm, 电压 = {res['voltage']} V")
    print(f"  -> 平均线密度 (Mean Density): {res['mean_ne']:.4e} m^-3")
    print(f"  -> 标准差 (Std Deviation):  {res['std_ne']:.4e} m^-3")
    print("-" * 50)


print("\nData processing complete. Generating plot...")

plt.figure(figsize=(10, 7))

grouped_results = defaultdict(list)
for res in plot_results:
    grouped_results[res['flow_rate']].append(res)

markers = ['o', 's', '^']
colors = ['blue', 'green', 'red']
i = 0

for flow_rate, data_points in sorted(grouped_results.items()):
    data_points.sort(key=lambda x: x['voltage'])

    voltages = [dp['voltage'] for dp in data_points]
    mean_densities = [dp['mean_ne'] for dp in data_points]
    std_densities = [dp['std_ne'] for dp in data_points]

    plt.errorbar(
        voltages,
        mean_densities,
        yerr=std_densities,
        label=f'Flow Rate = {flow_rate} sccm',
        fmt='-',
        capsize=5,
        marker=markers[i],
        color=colors[i],
    )
    i += 1

plt.title('Line-Averaged Electron Density vs. Grid Voltage', fontsize=16)
plt.xlabel('Grid Voltage (V)', fontsize=12)
plt.ylabel('Line-Averaged Density, $\\overline{n}_e$ ($m^{-3}$)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)


output_filename = 'density_vs_voltage.pdf'
plt.savefig(output_filename, bbox_inches='tight')
plt.close()

print(f"Plot generation complete! Figure saved as: {output_filename}")
