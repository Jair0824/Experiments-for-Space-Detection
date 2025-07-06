import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd


def robust_klett_inversion(signal_profile, h_km, k=0.86, ref_range_km=4.0):
    """
    Klett反演函数，内置对边界值的检查。
    """
    num_points = len(signal_profile)
    r = np.arange(num_points) * h_km

    # 1. 计算S(r)
    signal_profile[signal_profile <= 0] = 1e-12
    pr2 = signal_profile * r**2
    pr2[0] = 1e-12
    s = np.log(pr2)

    # 2. 估算并检查边界条件 sigma_m
    sigma_m = 1e-3
    try:
        est_range_km = (ref_range_km - 0.2, ref_range_km)
        b = int(est_range_km[0] / h_km) if h_km > 0 else 0
        m_idx = int(est_range_km[1] / h_km) if h_km > 0 else 0

        if 0 <= b < m_idx < len(s):
            s_b, s_m_val = s[b], s[m_idx]
            integrand = np.exp((s[b : m_idx + 1] - s_m_val) / k)
            integral = np.sum(integrand) * h_km
            numerator = np.exp((s_b - s_m_val) / k) - 1.0
            denominator = (2.0 / k) * integral

            if denominator != 0 and not np.isnan(denominator):
                estimated_val = numerator / denominator
                if 1e-5 < estimated_val < 1.0:
                    sigma_m = estimated_val
    except Exception:
        pass

    # 3. 执行Klett后向积分
    m = int(ref_range_km / h_km) if h_km > 0 else 0
    if m >= num_points:
        m = num_points - 1
    s_m = s[m]

    integrand = np.exp((s - s_m) / k)
    cumulative_sum_from_m = np.cumsum(integrand[m::-1])[::-1] * h_km
    integral_values = np.zeros_like(s)
    integral_values[: m + 1] = cumulative_sum_from_m

    denominator = (1.0 / sigma_m) + (2.0 / k) * integral_values
    numerator = np.exp((s - s_m) / k)
    sigma = np.zeros_like(s)

    valid_denom_indices = (denominator > 0) & (~np.isnan(denominator))
    sigma[valid_denom_indices] = (
        numerator[valid_denom_indices] / denominator[valid_denom_indices]
    )

    return sigma


def main():
    DATA_DIR = '.'
    try:
        dat_files = sorted(
            [
                f
                for f in os.listdir(DATA_DIR)
                if f.endswith('.dat') and f[:-4].isdigit()
            ],
            key=lambda x: int(x.split('.')[0]),
        )
    except (ValueError, TypeError):
        dat_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.dat')])

    if not dat_files:
        print("错误: 在当前目录下未找到 .dat 文件。")
        return

    H_METERS = 7.5
    SMOOTHING_WINDOW_SIZE = 20
    MAX_DISPLAY_ALTITUDE_KM = 15.0
    FONT_NAME = "Times New Roman"

    all_sigma_profiles = []
    timestamps = []

    print(f"正在处理 {len(dat_files)} 个文件...")
    for filename in dat_files:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if len(lines) < 13:
                    print(f"警告: 文件 {filename} 行数不足，已跳过。")
                    continue
                time_str = lines[1].split()[1] + " " + lines[1].split()[2]
                current_timestamp = datetime.strptime(time_str, '%d/%m/%Y %H:%M:%S')

            data = pd.read_csv(
                filepath,
                header=None,
                skiprows=12,
                sep=r'\s+',
                usecols=[0],
                names=['signal'],
                engine='python',
            )

            numeric_signal = pd.to_numeric(data['signal'], errors='coerce')
            raw_signal = numeric_signal.fillna(0).to_numpy()

            if len(raw_signal) == 0 or np.all(raw_signal == 0):
                raise ValueError("读取后的信号数据为空或全为0。")

            background = np.mean(raw_signal[int(len(raw_signal) * 0.9) :])
            corrected_signal = raw_signal - background

            smoothed_signal = np.convolve(
                corrected_signal,
                np.ones(SMOOTHING_WINDOW_SIZE) / SMOOTHING_WINDOW_SIZE,
                mode='same',
            )

            sigma_profile = robust_klett_inversion(smoothed_signal, H_METERS / 1000.0)

            if np.all(~np.isfinite(sigma_profile)) or np.nanmean(sigma_profile) < 1e-5:
                all_sigma_profiles.append(None)
            else:
                all_sigma_profiles.append(sigma_profile)

            timestamps.append(current_timestamp)

        except Exception as e:
            print(f"处理文件 {filename} 时出错，已跳过: {e}")
            if len(timestamps) > len(all_sigma_profiles):
                all_sigma_profiles.append(None)
            continue

    if not any(p is not None for p in all_sigma_profiles):
        print("错误: 所有文件都未能成功处理。")
        return

    print("所有文件处理完毕。正在插值并生成最终图表...")

    df = pd.DataFrame(all_sigma_profiles, index=timestamps).T

    df.interpolate(axis=1, method='time', limit_direction='both', inplace=True)
    sigma_matrix = df.to_numpy()

    sigma_matrix[np.isinf(sigma_matrix)] = np.nan
    sigma_matrix[sigma_matrix < 0] = np.nan

    vmin = (
        np.nanpercentile(sigma_matrix, 5) if not np.all(np.isnan(sigma_matrix)) else 0
    )
    vmax = (
        np.nanpercentile(sigma_matrix, 95) if not np.all(np.isnan(sigma_matrix)) else 1
    )

    if (
        np.isnan(vmax)
        or vmax < 1e-5
        or vmax > 10
        or ((vmax - vmin) / (vmax + 1e-9) < 0.01)
    ):
        vmin = 0
        vmax = 3.0

    y_altitude_km = np.arange(sigma_matrix.shape[0]) * (H_METERS / 1000.0)
    x_dates = mdates.date2num(df.columns)

    try:
        plt.rcParams['font.family'] = FONT_NAME
    except:
        plt.rcParams['font.family'] = 'serif'

    fig, ax = plt.subplots(figsize=(12, 6))

    c = ax.pcolormesh(
        x_dates,
        y_altitude_km,
        sigma_matrix,
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        shading='gouraud',
    )

    ax.set_ylim(0, MAX_DISPLAY_ALTITUDE_KM)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Height (km)", fontsize=12)
    ax.set_title(
        "Atmospheric Extinction Coefficient Profile (Klett Method)", fontsize=14
    )

    cbar = fig.colorbar(c, ax=ax, extend='both')
    cbar.set_label('Extinction Coefficient (km$^{-1}$)', fontsize=12)

    plt.tight_layout()
    output_filename = 'lidar_extinction_profile.pdf'
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    print(f"图表已保存为 {output_filename}")


if __name__ == '__main__':
    main()
