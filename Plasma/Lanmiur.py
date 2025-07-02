import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import os
import pandas as pd
import glob
from collections import defaultdict

# --- 物理常数 ---
E_CHARGE = 1.602176634e-19  # 元电荷 (C)
M_E = 9.1093837139e-31  # 电子质量 (kg)
K_B = 1.380649e-23  # 玻尔兹曼常数 (J/K)
M_AR_AMU = 39.948  # 氩原子质量 (amu)
AMU_TO_KG = 1.66053906892e-27  # 原子质量单位到千克的转换
M_I = M_AR_AMU * AMU_TO_KG  # 氩离子质量 (kg)

# --- 探针几何参数  ---
PROBE_LENGTH = 0.02  # m (1 cm)
PROBE_DIAMETER = 0.005  # m (1 mm)
PROBE_AREA = np.pi * PROBE_DIAMETER * PROBE_LENGTH  # 探针侧面积 (m^2)


def linear_func(x, a, b):
    """简单的线性函数用于拟合"""
    return a * x + b


def analyze_langmuir_condition(filepaths, probe_area, num_bins=150):
    """
    分析单个实验工况下的所有重复测量数据。
    """

    all_V, all_I = [], []
    for fp in filepaths:
        try:
            V, I = np.loadtxt(fp, unpack=True)
            all_V.append(V)
            all_I.append(I)
        except Exception as e:
            print(f"警告: 无法读取文件 {fp}。错误: {e}")
            continue

    if not all_V:
        return None, None

    all_V = np.concatenate(all_V)
    all_I = np.concatenate(all_I)

    # 电压分箱与统计
    v_range = (all_V.min(), all_V.max())
    I_mean, bin_edges, _ = binned_statistic(
        all_V, all_I, statistic='mean', bins=num_bins, range=v_range
    )
    I_std, _, _ = binned_statistic(
        all_V, all_I, statistic='std', bins=bin_edges, range=v_range
    )
    I_count, _, _ = binned_statistic(
        all_V, all_I, statistic='count', bins=bin_edges, range=v_range
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 清理无效数据: 移除点数过少或std为nan的箱
    valid_mask = (I_count > 1) & ~np.isnan(I_mean) & ~np.isnan(I_std) & (I_std > 0)
    bin_centers = bin_centers[valid_mask]
    I_mean = I_mean[valid_mask]
    I_std = I_std[valid_mask]

    if len(bin_centers) < 10:
        print(f"警告: 文件组 {os.path.basename(filepaths[0])} 的有效分箱数据点不足。")
        return None, None

    # 3. 从分箱数据中提取参数
    # 3.1 寻找悬浮电位 Vf
    try:
        interp_func = interp1d(
            bin_centers, I_mean, kind='cubic', fill_value="extrapolate"
        )
        v_fine = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
        i_fine = interp_func(v_fine)
        # 找到电流穿过0的点
        zero_cross_index = np.where(np.diff(np.sign(i_fine)))[0]
        if len(zero_cross_index) > 0:
            Vf = v_fine[zero_cross_index[0]]
        else:
            Vf = np.nan
    except Exception:
        Vf = np.nan

    if np.isnan(Vf):
        print(f"警告: 文件组 {os.path.basename(filepaths[0])} 未能找到悬浮电位 Vf。")
        return None, None

    # 3.2 加权拟合离子饱和流
    ion_sat_mask = bin_centers < (Vf - 2.0)
    if np.sum(ion_sat_mask) < 2:
        print(f"警告: 文件组 {os.path.basename(filepaths[0])} 离子饱和区数据点不足。")
        return None, None

    V_ion = bin_centers[ion_sat_mask]
    I_ion = I_mean[ion_sat_mask]
    sigma_ion = I_std[ion_sat_mask]

    popt_ion, pcov_ion = curve_fit(
        linear_func, V_ion, I_ion, sigma=sigma_ion, absolute_sigma=True
    )
    I_i_fit_func = lambda v: linear_func(v, *popt_ion)
    I_i_fit = I_i_fit_func(bin_centers)

    # 3.3 分离电子电流
    I_e_mean = I_mean - I_i_fit
    I_e_mean[I_e_mean <= 0] = np.nan

    # 3.4 寻找等离子体电位 Vp
    dI_dV = np.gradient(I_mean, bin_centers)
    vp_index = np.nanargmax(dI_dV)
    Vp = bin_centers[vp_index]

    # 3.5 加权拟合计算电子温度 Te
    te_fit_mask = (
        (bin_centers > Vf + 1.0) & (bin_centers < Vp - 1.0) & ~np.isnan(I_e_mean)
    )
    if np.sum(te_fit_mask) < 2:
        print(f"警告: 文件组 {os.path.basename(filepaths[0])} 电子减速区数据点不足。")
        return None, None

    V_te = bin_centers[te_fit_mask]
    log_Ie = np.log(I_e_mean[te_fit_mask])
    sigma_log_Ie = I_std[te_fit_mask] / I_e_mean[te_fit_mask]

    popt_te, pcov_te = curve_fit(
        linear_func, V_te, log_Ie, sigma=sigma_log_Ie, absolute_sigma=True
    )
    slope_te, intercept_te = popt_te
    slope_te_err, _ = np.sqrt(np.diag(pcov_te))

    Te_eV = 1.0 / slope_te
    Te_eV_err = slope_te_err / (slope_te**2)

    # 3.6 计算饱和流
    I_esat = interp1d(bin_centers, I_e_mean, kind='linear', fill_value="extrapolate")(
        Vp
    )
    I_isat = abs(I_i_fit_func(Vp))

    # 3.7 计算电子和离子密度
    Ne = (I_esat / (E_CHARGE * probe_area)) * np.sqrt(
        (2 * np.pi * M_E) / (E_CHARGE * Te_eV)
    )
    Ni = I_isat / (0.61 * E_CHARGE * probe_area * np.sqrt((E_CHARGE * Te_eV) / M_I))

    dNe_dTe = -0.5 * Ne / Te_eV
    Ne_err = abs(dNe_dTe * Te_eV_err)

    results = {
        'Vp': Vp,
        'Vf': Vf,
        'Te_eV': Te_eV,
        'Te_eV_err': Te_eV_err,
        'Ne': Ne,
        'Ne_err': Ne_err,
        'Ni': Ni,
    }

    I_e_fit_on_bins = np.full_like(bin_centers, np.nan)
    I_e_fit_on_bins[te_fit_mask] = np.exp(linear_func(V_te, *popt_te))
    I_total_fit = I_i_fit + I_e_fit_on_bins
    # 使用插值法获得平滑的曲线
    valid_fit_mask = ~np.isnan(I_total_fit)
    interp_fit_func = interp1d(
        bin_centers[valid_fit_mask],
        I_total_fit[valid_fit_mask],
        kind='cubic',
        fill_value='extrapolate',
    )

    data_for_plot = {
        'raw_V': all_V,
        'raw_I': all_I,
        'bin_centers': bin_centers,
        'I_mean': I_mean,
        'I_std': I_std,
        'interp_fit_func': interp_fit_func,
    }

    return results, data_for_plot


def plot_iv_analysis_detailed(results, data, condition_name, save_dir):
    """
    绘制指定工况的详细I-V分析图。
    """
    if results is None or data is None:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        data['raw_V'],
        data['raw_I'],
        'o',
        color='black',
        markersize=2,
        alpha=0.5,
        label='Raw data',
    )
    ax.errorbar(
        data['bin_centers'],
        data['I_mean'],
        yerr=data['I_std'],
        fmt='^',
        color='blue',
        markersize=4,
        capsize=3,
        ecolor='lightblue',
        label='bin centers (I mean ± I std)',
    )

    '''v_smooth = np.linspace(data['bin_centers'].min(), data['bin_centers'].max(), 500)
    i_fit_smooth = data['interp_fit_func'](v_smooth)
    ax.plot(v_smooth, i_fit_smooth, '-', color='red', linewidth=2.5, label='fit')'''

    ax.axvline(
        results['Vp'],
        color='purple',
        linestyle='-.',
        label=f"Vp = {results['Vp']:.2f} V",
    )
    ax.axvline(
        results['Vf'],
        color='orange',
        linestyle='-.',
        label=f"Vf = {results['Vf']:.2f} V",
    )
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)

    ax.set_xlabel(r'voltage(V)', fontsize=14)
    ax.set_ylabel(r'current(A)', fontsize=14)
    ax.set_title(
        f'Analysis of I-V characteristic curves({condition_name[:2]}sccm {condition_name[3]}A)',
        fontsize=16,
    )
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(np.min(data['I_mean']) * 3.2, np.max(data['I_mean']) * 0.5)
    ax.set_xlim(np.min(data['bin_centers']) * 1.2, results['Vp'] * 2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f'IV_Analysis_{condition_name}.pdf'),
        bbox_inches='tight',
        pad_inches=0.1,
        dpi=300,
    )
    plt.close(fig)


def main():
    data_directory = '.'
    output_directory = 'analysis_results'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    filepaths = glob.glob(os.path.join(data_directory, '*.txt'))
    conditions = defaultdict(list)
    for fp in filepaths:
        basename = os.path.basename(fp)
        try:
            parts = os.path.splitext(basename)[0].split('-')
            # -------------------------
            if len(parts) == 3:
                condition_key = f"{parts[0]}-{parts[1]}"
                conditions[condition_key].append(fp)
            else:
                print(
                    f"文件名 {basename} 格式不符 (应为 '流量-电流-组号.txt')，已跳过。"
                )
        except (IndexError, ValueError):
            print(f"文件名 {basename} 格式不符，已跳过。")

    all_results = []
    for condition_key, files in sorted(conditions.items()):
        print(f"--- 正在分析工况: {condition_key} ---")
        flow, current = map(int, condition_key.split('-'))

        results, data_for_plot = analyze_langmuir_condition(files, PROBE_AREA)

        if results:
            results['flow'] = flow
            results['current'] = current
            all_results.append(results)

            if condition_key == "55-7":
                print(f"--- 正在为工况 {condition_key} 生成详细I-V图 ---")
                plot_iv_analysis_detailed(
                    results, data_for_plot, condition_key, output_directory
                )

    if not all_results:
        print("未能成功分析任何数据。请检查文件和代码。")
        return

    df = pd.DataFrame(all_results)
    df.sort_values(by=['flow', 'current'], inplace=True)
    print("\n--- 所有工况分析结果汇总 ---")
    print(df.to_string())

    #  Ne vs 电流 (固定流量)
    unique_flows = df['flow'].unique()
    if 55 in unique_flows:
        df_const_flow = df[df['flow'] == 55]
        if not df_const_flow.empty:
            plt.figure(figsize=(8, 6))
            plt.errorbar(
                df_const_flow['current'],
                df_const_flow['Ne'],
                yerr=df_const_flow['Ne_err'],
                fmt='-o',
                capsize=5,
                markersize=8,
                label=r'$N_e$',
            )
            plt.xlabel(r'$I(A)$', fontsize=12)
            plt.ylabel(r'$N_e (m^{-3})$', fontsize=12)
            plt.title(
                r'Changes in electron density with discharge current ($flux=55 sccm$)',
                fontsize=14,
            )
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.savefig(
                os.path.join(output_directory, 'Trend_Ne_vs_Current.pdf'),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=300,
            )
            plt.close()
            print("\n已生成: Trend_Ne_vs_Current.png")

    # Ne vs 流量 (固定电流)
    unique_currents = df['current'].unique()
    if 7 in unique_currents:
        df_const_current = df[df['current'] == 7]
        if not df_const_current.empty:
            plt.figure(figsize=(8, 6))
            plt.errorbar(
                df_const_current['flow'],
                df_const_current['Ne'],
                yerr=df_const_current['Ne_err'],
                fmt='-o',
                capsize=5,
                markersize=8,
                label=r'$N_e$',
            )
            plt.xlabel(r'$Flux (sccm)$', fontsize=12)
            plt.ylabel(r'$N_e (m^{-3})$', fontsize=12)
            plt.title(
                r'Changes in electron density with gas flow rate ($I=7 A$)', fontsize=14
            )
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.savefig(
                os.path.join(output_directory, 'Trend_Ne_vs_Flow.pdf'),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=300,
            )
            plt.close()
            print("已生成: Trend_Ne_vs_Flow.png")


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    main()
