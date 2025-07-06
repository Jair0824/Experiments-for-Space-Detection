import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# --- 0. Configure Matplotlib for Publication Quality ---
# Set the default font to Times New Roman and increase font sizes
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams.update(
    {
        'font.size': 14,  # General font size
        'axes.titlesize': 18,  # Title font size
        'axes.labelsize': 16,  # X and Y labels font size
        'xtick.labelsize': 14,  # X tick labels
        'ytick.labelsize': 14,  # Y tick labels
        'legend.fontsize': 14,  # Legend font size
        #'axes.labelweight': 'bold',  # Make axis labels bold
        #'axes.titleweight': 'bold',  # Make title bold
    }
)

# --- 1. 定义常量和参数 ---
C_LIGHT = 299792458
RANGE_RESOLUTION = (C_LIGHT * (1 / 10e6)) / 2
KLETT_K = 0.7
R_M_METERS = 4500
R_B_METERS = 750


# --- 2. 数据加载和预处理函数 ---
def load_data(file_path):
    """从txt文件加载激光雷达数据"""
    try:
        data = np.loadtxt(file_path, delimiter=',')
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None, None


def load_geom_factor(file_path):
    """加载几何因子文件"""
    try:
        data = np.loadtxt(file_path)
        return data[:, 0] * 1000, data[:, 1]
    except Exception as e:
        print(f"Error loading geometric factor from {file_path}: {e}")
        return None, None


# --- 3. 主程序 ---
if __name__ == "__main__":
    # --- 文件路径 ---
    signal_file = '200812151717A.avr'
    background_file = '200812151718A.avr'
    geom_factor_file = 'GF.txt'

    # --- 加载所有数据 ---
    idx, signal_raw = load_data(signal_file)
    _, background_raw = load_data(background_file)
    geom_range, geom_factor = load_geom_factor(geom_factor_file)

    if idx is None or background_raw is None or geom_factor is None:
        print("Failed to load data files. Exiting.")
    else:
        # --- 数据准备 ---
        range_m = idx * RANGE_RESOLUTION
        range_km = range_m / 1000

        # --- 步骤 3: 绘制原始信号和背景廓线图 (带局部放大) ---
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        ax1.plot(
            range_km,
            signal_raw,
            label='Raw Lidar Signal',
            color='darkblue',
            linewidth=2,
        )
        ax1.plot(
            range_km,
            background_raw,
            label='Background Noise',
            color='gray',
            linestyle='--',
            linewidth=2,
        )
        ax1.set_yscale('log')
        ax1.set_xlabel('Height (km)')
        ax1.set_ylabel('Signal Strength (Log Scale)')
        ax1.set_title('Raw Signal and Background Profile')
        # **最终修改**: 将图例移动到左下角以避免重叠
        ax1.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.8)
        ax1.grid(True, which="both", ls="--")
        ax1.set_xlim(0, 15)
        ax1.set_ylim(bottom=1)

        # 创建并正确设置局部放大子图
        ax_inset = ax1.inset_axes([0.45, 0.45, 0.5, 0.5])
        ax_inset.plot(range_km, signal_raw, color='darkblue', linewidth=2)
        ax_inset.plot(
            range_km, background_raw, color='gray', linestyle='--', linewidth=2
        )

        x1, x2 = 6, 12
        idx_zoom = (range_km >= x1) & (range_km <= x2)
        y_min_zoom = min(np.min(signal_raw[idx_zoom]), np.min(background_raw[idx_zoom]))
        y_max_zoom = max(np.max(signal_raw[idx_zoom]), np.max(background_raw[idx_zoom]))
        y_padding = (y_max_zoom - y_min_zoom) * 0.1
        y1_zoom, y2_zoom = y_min_zoom - y_padding, y_max_zoom + y_padding

        ax_inset.set_xlim(x1, x2)
        ax_inset.set_ylim(y1_zoom, y2_zoom)
        ax_inset.tick_params(labelleft=True, labelbottom=True)
        ax_inset.grid(True, ls="--")

        mark_inset(ax1, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

        fig1.tight_layout()
        fig1.savefig(
            'plot_1_raw_signal_with_inset.pdf', format='pdf', bbox_inches='tight'
        )
        print("Figure 1 saved as 'plot_1_raw_signal_with_inset.pdf'")

        # --- 步骤 4: 扣背景处理与绘图 ---
        signal_clean = signal_raw - background_raw
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(
            range_km,
            signal_clean,
            label='Background-Subtracted Signal',
            color='green',
            linewidth=2,
        )
        ax2.set_yscale('log')
        ax2.set_xlabel('Height (km)')
        ax2.set_ylabel('Signal Strength (Log Scale)')
        ax2.set_title('Background-Subtracted Signal Profile')
        ax2.legend(frameon=True, facecolor='white', framealpha=0.8)
        ax2.grid(True, which="both", ls="--")
        ax2.set_xlim(0.5, 10)
        ax2.set_ylim(bottom=1)
        fig2.tight_layout()
        fig2.savefig(
            'plot_2_background_subtracted.pdf', format='pdf', bbox_inches='tight'
        )
        print("Figure 2 saved as 'plot_2_background_subtracted.pdf'")

        # --- 步骤 5: 滤波处理与绘图 ---
        window_size = 5
        weights = np.repeat(1.0, window_size) / window_size
        signal_smooth = np.convolve(signal_clean, weights, 'same')
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(
            range_km,
            signal_smooth,
            label='Smoothed Signal',
            color='purple',
            linewidth=2,
        )
        ax3.set_yscale('log')
        ax3.set_xlabel('Height (km)')
        ax3.set_ylabel('Signal Strength (Log Scale)')
        ax3.set_title('Smoothed Signal Profile')
        ax3.legend(frameon=True, facecolor='white', framealpha=0.8)
        ax3.grid(True, which="both", ls="--")
        ax3.set_xlim(0.5, 10)
        ax3.set_ylim(bottom=1)
        fig3.tight_layout()
        fig3.savefig('plot_3_smoothed_signal.pdf', format='pdf', bbox_inches='tight')
        print("Figure 3 saved as 'plot_3_smoothed_signal.pdf'")

        # --- 步骤 6 & 7: Klett反演与绘图 ---
        geom_interp = np.interp(range_m, geom_range, geom_factor)
        signal_final = np.divide(
            signal_smooth,
            geom_interp,
            out=np.zeros_like(signal_smooth),
            where=geom_interp > 0.1,
        )
        valid_indices = (range_m > 500) & (signal_final > 0)
        r = range_m[valid_indices]
        P = signal_final[valid_indices]
        S = np.log(P * r**2)

        try:
            m_index = np.where(r >= R_M_METERS)[0][0]
            b_index = np.where(r >= R_B_METERS)[0][0]
        except IndexError:
            print(f"Error: Reference heights are beyond the data range.")
        else:
            Sm = S[m_index]
            Sb = S[b_index]

            integral_for_sigma_m = np.trapezoid(
                np.exp((S[b_index : m_index + 1] - Sm) / KLETT_K),
                r[b_index : m_index + 1],
            )
            sigma_m = (np.exp((Sb - Sm) / KLETT_K) - 1) / (
                (2 / KLETT_K) * integral_for_sigma_m
            )

            sigma = np.zeros_like(r)
            sigma[m_index] = sigma_m

            for i in range(m_index - 1, -1, -1):
                integral_backward = np.trapezoid(
                    np.exp((S[i : m_index + 1] - Sm) / KLETT_K), r[i : m_index + 1]
                )
                sigma[i] = np.exp((S[i] - Sm) / KLETT_K) / (
                    (1 / sigma_m) + (2 / KLETT_K) * integral_backward
                )

            fig4, ax4 = plt.subplots(figsize=(10, 6))
            sigma_km = sigma * 1000
            plot_indices = sigma > 0
            ax4.plot(
                r[plot_indices] / 1000,
                sigma_km[plot_indices],
                label='Extinction Coefficient (Klett)',
                color='red',
                linewidth=2,
            )
            ax4.set_xlabel('Height (km)')
            ax4.set_ylabel('Extinction Coefficient (km$^{-1}$)')
            ax4.set_title(f'Atmospheric Extinction Coefficient Profile (k={KLETT_K})')
            ax4.legend(frameon=True, facecolor='white', framealpha=0.8)

            valid_sigma_for_plot = sigma_km[
                plot_indices & (r / 1000 < 5.0) & (r / 1000 > 0.5)
            ]
            if len(valid_sigma_for_plot) > 0:
                ax4.set_ylim(0, np.max(valid_sigma_for_plot) * 1.1)

            ax4.set_xlim(0.5, 5.0)
            fig4.tight_layout()
            fig4.savefig(
                'plot_4_extinction_coefficient.pdf', format='pdf', bbox_inches='tight'
            )
            print("Figure 4 saved as 'plot_4_extinction_coefficient.pdf'")

            # 显示所有生成的图
            plt.show()
