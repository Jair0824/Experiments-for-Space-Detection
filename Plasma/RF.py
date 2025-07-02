import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict


e = 1.602176634e-19  # Elementary charge (C)
m_e = 9.10938356e-31  # Electron mass (kg)
epsilon_0 = 8.854187817e-12  # Vacuum permittivity (F/m)

B_nT = 50353.5  # Geomagnetic field in Hefei (nT)
B_T = B_nT * 1e-9  # Convert to Tesla (T)


FREQ_THRESHOLD = 1e7  # 10 MHz


experimental_conditions = [
    # Series 1: Fixed flow @ 4sccm, varying voltage (3 reps each)
    {'file_index': 1, 'voltage': 13, 'flow_rate': 4},
    {'file_index': 2, 'voltage': 15, 'flow_rate': 4},
    {'file_index': 3, 'voltage': 17, 'flow_rate': 4},
    {'file_index': 4, 'voltage': 13, 'flow_rate': 4},
    {'file_index': 5, 'voltage': 15, 'flow_rate': 4},
    {'file_index': 6, 'voltage': 17, 'flow_rate': 4},
    {'file_index': 7, 'voltage': 13, 'flow_rate': 4},
    {'file_index': 8, 'voltage': 15, 'flow_rate': 4},
    {'file_index': 9, 'voltage': 17, 'flow_rate': 4},
    # Series 2: Fixed voltage @ 13V, varying flow (3 reps each)
    {'file_index': 10, 'voltage': 13, 'flow_rate': 2},
    {'file_index': 11, 'voltage': 13, 'flow_rate': 4},
    {'file_index': 12, 'voltage': 13, 'flow_rate': 6},
    {'file_index': 13, 'voltage': 13, 'flow_rate': 2},
    {'file_index': 14, 'voltage': 13, 'flow_rate': 4},
    {'file_index': 15, 'voltage': 13, 'flow_rate': 6},
    {'file_index': 16, 'voltage': 13, 'flow_rate': 2},
    {'file_index': 17, 'voltage': 13, 'flow_rate': 4},
    {'file_index': 18, 'voltage': 13, 'flow_rate': 6},
]


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


def load_data(filename):
    """Loads data from a CSV file, skipping headers and cleaning the data."""
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found. Skipping.")
        return None

    df = pd.read_csv(
        filename, skiprows=2, header=None, names=['Frequency', 'Real', 'Imag']
    )
    for col in ['Frequency', 'Real', 'Imag']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def calculate_electron_density(f_uhr, B):
    """Calculates electron density from the upper hybrid resonance frequency."""
    if f_uhr is None or pd.isna(f_uhr):
        return None
    f_ce = (e * B) / (2 * np.pi * m_e)
    if f_uhr**2 <= f_ce**2:
        print(
            f"Warning: f_uhr ({f_uhr/1e6:.2f} MHz) <= f_ce ({f_ce/1e6:.2f} MHz). Cannot calculate density."
        )
        return None
    f_pe_sq = f_uhr**2 - f_ce**2
    n_e = (f_pe_sq * (4 * np.pi**2 * epsilon_0 * m_e)) / e**2
    return n_e / 1e6


base_data = load_data('0.CSV')
if base_data is None or base_data.empty:
    print(
        "Error: Background data file 0.CSV could not be loaded or is empty. Terminating."
    )
    exit()

results = []
for i in range(1, 19):
    filename = f'{i}.CSV'
    plasma_data = load_data(filename)
    if plasma_data is None or plasma_data.empty:
        print(f"Warning: Skipping file {filename} as it is missing or invalid.")
        continue

    base_real_interp = np.interp(
        plasma_data['Frequency'], base_data['Frequency'], base_data['Real']
    )
    base_imag_interp = np.interp(
        plasma_data['Frequency'], base_data['Frequency'], base_data['Imag']
    )

    delta_real = plasma_data['Real'] - base_real_interp
    delta_imag = plasma_data['Imag'] - base_imag_interp
    modulus = np.sqrt(delta_real**2 + delta_imag**2)
    phase = np.arctan2(delta_imag, delta_real) * 180 / np.pi

    valid_range = plasma_data['Frequency'] > FREQ_THRESHOLD
    if not valid_range.any() or modulus[valid_range].empty:
        f_uhr = None
    else:
        peak_index = modulus[valid_range].idxmax()
        f_uhr = plasma_data['Frequency'][peak_index]

    n_e = calculate_electron_density(f_uhr, B_T)

    condition = next(
        (item for item in experimental_conditions if item["file_index"] == i), None
    )
    if condition:
        results.append(
            {
                **condition,
                'frequency': plasma_data['Frequency'],
                'modulus': modulus,
                'phase': phase,
                'f_uhr': f_uhr,
                'n_e_cm3': n_e,
            }
        )


print("\nGenerating and saving plots as PDF files...")


plt.figure(figsize=(8, 5))
series_to_plot_1 = [1, 2, 3]
for res in results:
    if res['file_index'] in series_to_plot_1:
        label = f"V = {res['voltage']} V, Flow = {res['flow_rate']} sccm"
        plt.plot(res['frequency'] / 1e6, res['modulus'], label=label)


plt.yscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.6)


plt.title('Impedance Modulus vs. Frequency (Fixed Flow)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Impedance Modulus, $|\Delta Z|$ ($\Omega$)')
plt.legend()
plt.savefig('plot_1_modulus_vs_freq_log.pdf', bbox_inches='tight', pad_inches=0.05)
plt.close()


plt.figure(figsize=(8, 5))
series_to_plot_2 = [10, 11, 12]
for res in results:
    if res['file_index'] in series_to_plot_2:
        label = f"V = {res['voltage']} V, Flow = {res['flow_rate']} sccm"
        plt.plot(res['frequency'] / 1e6, res['phase'], label=label)
plt.title('Impedance Phase vs. Frequency (Fixed Voltage)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Phase (degrees)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('plot_2_phase_vs_freq.pdf', bbox_inches='tight', pad_inches=0.05)
plt.close()


plt.figure(figsize=(8, 5))
plasma_data_1 = load_data('1.CSV')
if (
    plasma_data_1 is not None
    and not plasma_data_1.empty
    and base_data is not None
    and not base_data.empty
    and results
):
    plt.plot(
        base_data['Frequency'] / 1e6,
        np.sqrt(base_data['Real'] ** 2 + base_data['Imag'] ** 2),
        label='Background Impedance, $|Z_{base}|$',
        linestyle='--',
        color='gray',
    )
    plt.plot(
        plasma_data_1['Frequency'] / 1e6,
        np.sqrt(plasma_data_1['Real'] ** 2 + plasma_data_1['Imag'] ** 2),
        label='Raw Measured Impedance, $|Z_{plasma}|$',
        linestyle=':',
    )
    result_1 = next((r for r in results if r['file_index'] == 1), None)
    if result_1:
        plt.plot(
            result_1['frequency'] / 1e6,
            result_1['modulus'],
            label='Corrected Impedance, $|\Delta Z|$',
            color='red',
            linewidth=2,
        )
        if result_1['f_uhr'] and pd.notna(result_1['f_uhr']):
            if not result_1['modulus'][
                result_1['frequency'] == result_1['f_uhr']
            ].empty:
                peak_modulus = result_1['modulus'][
                    result_1['frequency'] == result_1['f_uhr']
                ].values[0]
                plt.axvline(
                    x=result_1['f_uhr'] / 1e6,
                    color='b',
                    linestyle='--',
                    label=f'Resonance Peak, $f_{{UHR}}$ = {result_1["f_uhr"]/1e6:.2f} MHz',
                )
                plt.scatter(
                    result_1['f_uhr'] / 1e6, peak_modulus, color='blue', zorder=5
                )
plt.title('Impedance Correction Example')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Impedance Modulus ($\Omega$)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('plot_3_correction_example.pdf', bbox_inches='tight', pad_inches=0.05)
plt.close()


plt.figure(figsize=(8, 5))
exp1_results = defaultdict(list)
for res in results:
    if 1 <= res['file_index'] <= 9 and pd.notna(res['n_e_cm3']):
        exp1_results[res['voltage']].append(res['n_e_cm3'])
if exp1_results:
    voltages = sorted(exp1_results.keys())
    mean_ne = [np.mean(exp1_results[v]) for v in voltages]
    std_ne = [np.std(exp1_results[v]) for v in voltages]
    plt.errorbar(
        voltages,
        mean_ne,
        yerr=std_ne,
        fmt='-o',
        capsize=5,
        label='Mean Electron Density',
    )
    plt.title('Electron Density vs. Grid Voltage (Fixed Flow: 4 sccm)')
    plt.xlabel('Grid Voltage (V)')
    plt.ylabel('Electron Density, $n_e$ (cm$^{-3}$)')
    plt.xticks(voltages)
    plt.grid(True, linestyle='--', alpha=0.6)
else:
    plt.text(0.5, 0.5, 'No valid data for this plot', ha='center', va='center')
plt.savefig('plot_4a_ne_vs_voltage.pdf', bbox_inches='tight', pad_inches=0.05)
plt.close()


plt.figure(figsize=(8, 5))
exp2_results = defaultdict(list)
for res in results:
    if 10 <= res['file_index'] <= 18 and pd.notna(res['n_e_cm3']):
        exp2_results[res['flow_rate']].append(res['n_e_cm3'])
if exp2_results:
    flows = sorted(exp2_results.keys())
    mean_ne_flow = [np.mean(exp2_results[f]) for f in flows]
    std_ne_flow = [np.std(exp2_results[f]) for f in flows]
    plt.errorbar(
        flows,
        mean_ne_flow,
        yerr=std_ne_flow,
        fmt='-s',
        capsize=5,
        label='Mean Electron Density',
    )
    plt.title('Electron Density vs. Flow Rate (Fixed Voltage: 13 V)')
    plt.xlabel('Flow Rate (sccm)')
    plt.ylabel('Electron Density, $n_e$ (cm$^{-3}$)')
    plt.xticks(flows)
    plt.grid(True, linestyle='--', alpha=0.6)
else:
    plt.text(0.5, 0.5, 'No valid data for this plot', ha='center', va='center')
plt.savefig('plot_4b_ne_vs_flow.pdf', bbox_inches='tight', pad_inches=0.05)
plt.close()

print("\nAll plots have been successfully saved as PDF files.")
