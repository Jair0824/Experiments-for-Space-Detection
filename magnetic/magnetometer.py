import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --- 字体与样式设置 ---

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 20


# --- 参数设置 ---
FILENAME = '250529140158.A119'
MIN_PLATFORM_DURATION_S = 10.0
CORRECTION_FACTOR = 1.3


VOLTAGE_TO_PLATFORM_MAP = {
    0.0: 3,
    0.1: 4,
    0.2: 5,
    0.3: 6,
    0.4: 7,
    0.5: 8,
    0.6: 9,
}


CALIBRATION_PARAMS = {
    'OFFSET_BX1': 17.514,
    'OFFSET_BY1': -0.391,
    'OFFSET_BZ1': -19.833,
    'SCALE_BX1': 0.990,
    'SCALE_BY1': 1.006,
    'SCALE_BZ1': 0.992,
    'ORTH_BXY1': 0.000,
    'ORTH_BXZ1': -0.007,
    'ORTH_BYZ1': 0.003,
}


# --- 主程序 ---
def main():
    raw_data = load_data_as_text(FILENAME)
    if raw_data is None:
        return

    calibrated_data = apply_calibration(raw_data, CALIBRATION_PARAMS)

    high_quality_plateaus = find_and_filter_plateaus(calibrated_data)
    if high_quality_plateaus is None:
        return

    if None in VOLTAGE_TO_PLATFORM_MAP.values():
        print("\n判断真实平台与电压的对应关系。")
        plot_B_components_with_plateaus_annotated(
            calibrated_data, high_quality_plateaus['intervals']
        )
        print("\n决策辅助")
        print(
            "根据图表和打印的平台信息，修改代码顶部的 VOLTAGE_TO_PLATFORM_MAP 字典，然后再次运行。"
        )
    else:
        print("\n已指定平台，开始完成实验...")

        process_to_1Hz(calibrated_data)
        visualize_1Hz_data()
        analyze_noise_level(
            calibrated_data, high_quality_plateaus, VOLTAGE_TO_PLATFORM_MAP
        )
        analyze_and_plot_final(
            high_quality_plateaus['vectors'], VOLTAGE_TO_PLATFORM_MAP
        )


# --- 核心分析与绘图 ---
def analyze_and_plot_final(all_detected_vectors, user_map):
    print("\n--- 计算螺线管线圈的匝数 ---")
    voltages = np.array(list(user_map.keys()))
    selected_indices = [v - 1 for v in user_map.values()]
    selected_vectors = all_detected_vectors[selected_indices]
    bx_p, by_p, bz_p = (
        selected_vectors[:, 0],
        selected_vectors[:, 1],
        selected_vectors[:, 2],
    )

    # --- 1. 分量拟合 ---
    slope_x, intercept_x, r_x, _, _ = linregress(voltages, bx_p)
    slope_y, intercept_y, r_y, _, _ = linregress(voltages, by_p)
    slope_z, intercept_z, r_z, _, _ = linregress(voltages, bz_p)

    print("\n各分量线性拟合结果 :")
    print(f"  Bx vs U: 斜率 = {slope_x:.2f} nT/V, R² = {r_x**2:.6f}")
    print(f"  By vs U: 斜率 = {slope_y:.2f} nT/V, R² = {r_y**2:.6f}")
    print(f"  Bz vs U: 斜率 = {slope_z:.2f} nT/V, R² = {r_z**2:.6f}")

    # --- 2. 绘制所有三张图 ---
    plot_components_vs_voltage(
        voltages,
        bx_p,
        by_p,
        bz_p,
        r_x,
        r_y,
        r_z,
        slope_x,
        intercept_x,
        slope_y,
        intercept_y,
        slope_z,
        intercept_z,
    )
    plot_total_change_vs_voltage(voltages, selected_vectors)

    # --- 3. 最终计算 ---
    slope_total_mag = np.sqrt(slope_x**2 + slope_y**2 + slope_z**2) * 1e-9
    L = 0.785
    R_ohm = 40.1799
    mu_0 = 4 * np.pi * 1e-7
    N = slope_total_mag * (L * R_ohm) / (mu_0 * CORRECTION_FACTOR)

    print("\n最终计算结果 :")
    print(f"计算出的螺线管匝数 N ≈ {N:.1f}")


def process_to_1Hz(calibrated_data):
    print("\n--- 要求3: 将200Hz数据处理为1Hz ---")
    time = np.arange(calibrated_data.shape[0]) / 200.0

    num_seconds = int(len(calibrated_data) / 200)
    data_1hz_avg = [
        [
            np.mean(time[i * 200 : (i + 1) * 200]),
            np.mean(calibrated_data[i * 200 : (i + 1) * 200, 1]),
            np.mean(calibrated_data[i * 200 : (i + 1) * 200, 2]),
            np.mean(calibrated_data[i * 200 : (i + 1) * 200, 3]),
        ]
        for i in range(num_seconds)
    ]
    np.savetxt(
        'data_1Hz_averaged.txt',
        data_1hz_avg,
        fmt='%.4f',
        header='Time(s) Bx(nT) By(nT) Bz(nT)',
    )
    print("方法1 (平均法): 已将1Hz数据保存至 data_1Hz_averaged.txt")


def visualize_1Hz_data():
    """新增：将1Hz数据可视化"""
    try:
        data_1hz = np.loadtxt('data_1Hz_averaged.txt')
        time, bx, by, bz = (
            data_1hz[:, 0],
            data_1hz[:, 1],
            data_1hz[:, 2],
            data_1hz[:, 3],
        )

        plt.figure(figsize=(15, 7))
        plt.plot(time, bx, 'o-', label='$B_x$ (1Hz Averaged)')
        plt.plot(time, by, 'o-', label='$B_y$ (1Hz Averaged)')
        plt.plot(time, bz, 'o-', label='$B_z$ (1Hz Averaged)')
        plt.title('1Hz Resampled Data vs. Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Magnetic Field (nT)')
        plt.grid(True)
        plt.legend()
        plt.savefig('Resampled_1Hz_Data.pdf', bbox_inches='tight', pad_inches=0.05)
        print("已将1Hz数据的可视化结果保存到 Resampled_1Hz_Data.pdf")
        plt.show()
    except FileNotFoundError:
        print("未能找到 data_1Hz_averaged.txt 文件，跳过1Hz数据可视化。")


def analyze_noise_level(calibrated_data, plateaus_info, user_map):
    print("\n--- 要求4: 分析磁通门磁强计测量数据的噪声水平 ---")
    if 0.0 not in user_map or user_map[0.0] is None:
        print("尚未指定0.0V对应的平台，无法进行噪声分析。")
        return

    platform_index = user_map[0.0] - 1
    start_idx, end_idx = plateaus_info['intervals'][platform_index]
    margin = int((end_idx - start_idx) * 0.3)
    stable_start, stable_end = start_idx + margin, end_idx - margin

    noise_bx = np.std(calibrated_data[stable_start:stable_end, 1])
    noise_by = np.std(calibrated_data[stable_start:stable_end, 2])
    noise_bz = np.std(calibrated_data[stable_start:stable_end, 3])

    print("在0.0V平台（背景场）期间测量的噪声水平如下：")
    print(f"  Bx分量噪声 (标准差): {noise_bx:.3f} nT")
    print(f"  By分量噪声 (标准差): {noise_by:.3f} nT")
    print(f"  Bz分量噪声 (标准差): {noise_bz:.3f} nT")


# --- 其他辅助函数 ---
def plot_components_vs_voltage(
    voltages,
    bx_p,
    by_p,
    bz_p,
    r_x,
    r_y,
    r_z,
    slope_x,
    intercept_x,
    slope_y,
    intercept_y,
    slope_z,
    intercept_z,
):

    plt.figure(figsize=(12, 8))
    plt.scatter(voltages, bx_p, label=f'$B_x$ ($R^2={r_x**2:.4f}$)')
    plt.plot(voltages, slope_x * voltages + intercept_x, '--')
    plt.scatter(voltages, by_p, label=f'$B_y$ ($R^2={r_y**2:.4f}$)')
    plt.plot(voltages, slope_y * voltages + intercept_y, '--')
    plt.scatter(voltages, bz_p, label=f'$B_z$ ($R^2={r_z**2:.4f}$)')
    plt.plot(voltages, slope_z * voltages + intercept_z, '--')
    plt.title('Magnetic Field Components vs. Confirmed Voltage')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Magnetic Field Component (nT)')
    plt.grid(True)
    plt.legend()
    plt.savefig('2_Components_vs_Voltage.pdf', bbox_inches='tight', pad_inches=0.05)
    print("已保存分析图表到 2_Components_vs_Voltage.pdf")
    plt.show()


def plot_total_change_vs_voltage(voltages, selected_vectors):
    """绘制总磁场变化vs电压图"""
    zero_voltage_index = np.where(voltages == 0.0)[0][0]
    B0_vec = selected_vectors[zero_voltage_index]
    delta_B_vectors = selected_vectors - B0_vec
    B_magnitudes = np.linalg.norm(delta_B_vectors, axis=1)
    slope_mag, intercept_mag, r_mag, _, _ = linregress(voltages, B_magnitudes)
    plt.figure(figsize=(12, 8))
    plt.scatter(voltages, B_magnitudes, label=r'Data Points: $|\vec{B}(U)-\vec{B}_0|$')
    plt.plot(
        voltages,
        slope_mag * voltages + intercept_mag,
        'r--',
        label=f'Linear Fit ($R^2={r_mag**2:.4f}$)',
    )
    plt.title(r'Change in Magnetic Field Magnitude vs. Confirmed Voltage')
    plt.xlabel('Voltage (V)')
    plt.ylabel(r'Change in Magnetic Field $|\Delta B|$ (nT)')
    plt.grid(True)
    plt.legend()
    plt.savefig('3_Total_Change_vs_Voltage.pdf', bbox_inches='tight', pad_inches=0.05)
    print("已保存分析图表到 3_Total_Change_vs_Voltage.pdf")
    plt.show()


def plot_B_components_with_plateaus_annotated(time, calibrated_data, intervals):
    bx, by, bz = calibrated_data[:, 1], calibrated_data[:, 2], calibrated_data[:, 3]
    plt.figure(figsize=(15, 8))
    plt.plot(time, bx, alpha=0.2)
    plt.plot(time, by, alpha=0.2)
    plt.plot(time, bz, alpha=0.2)
    colors = plt.cm.get_cmap('tab20', len(intervals))
    for i, (start_idx, end_idx) in enumerate(intervals):
        plt.axvspan(time[start_idx], time[end_idx], color=colors(i), alpha=0.4)
        text_x = time[start_idx] + (time[end_idx] - time[start_idx]) / 2
        text_y = np.mean(bz[start_idx:end_idx])
        plt.text(
            text_x,
            text_y,
            f'{i+1}',
            fontsize=16,
            color='black',
            weight='bold',
            ha='center',
            va='center',
        )
    plt.title('Annotated High-Quality Platforms for Selection')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetic Field (nT)')
    plt.grid(True)
    plt.savefig(
        'Annotated_Plateaus_for_Selection.pdf', bbox_inches='tight', pad_inches=0.05
    )
    print("\n已将带编号的平台图保存到 Annotated_Plateaus_for_Selection.pdf")
    plt.show()


def load_data_as_text(filename):
    try:
        data = pd.read_csv(filename, sep='\s+', header=None, on_bad_lines='skip')
        return data.to_numpy()
    except FileNotFoundError:
        return None


def apply_calibration(raw_data, params):
    calibrated_data = raw_data.copy()
    bx_raw, by_raw, bz_raw = raw_data[:, 1], raw_data[:, 2], raw_data[:, 3]
    bx_off = bx_raw - params['OFFSET_BX1']
    by_off = by_raw - params['OFFSET_BY1']
    bz_off = bz_raw - params['OFFSET_BZ1']
    bx_orth = bx_off - params['ORTH_BXY1'] * by_off - params['ORTH_BXZ1'] * bz_off
    by_orth = by_off + params['ORTH_BXY1'] * bx_off - params['ORTH_BYZ1'] * bz_off
    bz_orth = bz_off + params['ORTH_BXZ1'] * bx_off + params['ORTH_BYZ1'] * by_off
    calibrated_data[:, 1] = bx_orth / params['SCALE_BX1']
    calibrated_data[:, 2] = by_orth / params['SCALE_BY1']
    calibrated_data[:, 3] = bz_orth / params['SCALE_BZ1']
    return calibrated_data


def find_and_filter_plateaus(calibrated_data):
    time = np.arange(calibrated_data.shape[0]) / 200.0
    bx, by, bz = calibrated_data[:, 1], calibrated_data[:, 2], calibrated_data[:, 3]
    print("\nStarting adaptive platform detection...")
    b_magnitude_signal = np.sqrt(bx**2 + by**2 + bz**2)
    diff_signal = np.abs(np.diff(b_magnitude_signal))
    threshold = np.percentile(diff_signal, 99.8)
    change_indices = np.where(diff_signal > threshold)[0]

    clean_changes = []
    if len(change_indices) > 0:
        clean_changes.append(change_indices[0])
        for i in range(1, len(change_indices)):
            if change_indices[i] - clean_changes[-1] > 200:
                clean_changes.append(change_indices[i])

    stable_intervals, last_idx = [], 0
    for idx in clean_changes:
        stable_intervals.append((last_idx, idx))
        last_idx = idx + 1
    stable_intervals.append((last_idx, len(time) - 1))

    high_quality_vectors, high_quality_intervals = [], []
    print("\nDetected high-quality platforms (duration > 10s):")
    for start_idx, end_idx in stable_intervals:
        duration = time[end_idx] - time[start_idx]
        if duration > MIN_PLATFORM_DURATION_S:
            margin = int((end_idx - start_idx) * 0.3)
            stable_start, stable_end = start_idx + margin, end_idx - margin
            if stable_end > stable_start:
                b_vec = [
                    np.mean(bx[stable_start:stable_end]),
                    np.mean(by[stable_start:stable_end]),
                    np.mean(bz[stable_start:stable_end]),
                ]
                high_quality_vectors.append(b_vec)
                high_quality_intervals.append((start_idx, end_idx))
                print(
                    f"  Platform {len(high_quality_vectors)}: from {time[start_idx]:.2f}s to {time[end_idx]:.2f}s, Mean Bx: {b_vec[0]:.2f}"
                )

    return (
        {'vectors': np.array(high_quality_vectors), 'intervals': high_quality_intervals}
        if high_quality_vectors
        else None
    )


if __name__ == '__main__':
    main()
