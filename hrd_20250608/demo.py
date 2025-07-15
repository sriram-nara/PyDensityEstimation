# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: tiegcm_rope_env
#     language: python
#     name: python3
# ---

# +
#Latest version 2025-06-08

import numpy as np
from datetime import datetime, timedelta
import orekit
import time
import pymsis
from pymsis import msis
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import utilities_ds as u
from orekit.pyhelpers import download_orekit_data_curdir, setup_orekit_curdir
from os import path
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.linalg import expm, logm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import rope_class_hrd as rope   

download_orekit_data_curdir( './orekit-data.zip' )  # Comment this out once this file has already been downloaded for repeated runs
vm = orekit.initVM()
setup_orekit_curdir( './' )

print ( 'Java version:', vm.java_version )
print ( 'Orekit version:', orekit.VERSION )

# +
#User inputs
forward_propagation = 1 #forward propagation time
init_date = pd.to_datetime('2003-05-10 00:00:00')

latitude_values = [89.] # degrees
local_time_values = [8.] # hours
altitude_values = [358.] # km
lla_array = np.vstack((latitude_values, local_time_values, altitude_values)).T.reshape((-1, 3))
timestamps = [pd.to_datetime('2003-05-12 00:00:00')]

sindy = rope.rope_propagator()
sindy.propagate_models(init_date = init_date, forward_propagation = forward_propagation)


rope_density = rope.rope_data_interpolator( data = sindy )

all_models, dmd_density, ensemble_density, density_std = rope_density.interpolate(timestamps, lla_array)

ensemble_density, density_std

# +
init_date = pd.to_datetime('2003-10-29 00:00:00')

execution_times = []
max_propagation_days = 4
propagation_resolution = 1

for n, forward_propagation in enumerate(list(range(1, max_propagation_days + 1))[::propagation_resolution]):
    
    start_time = time.time()
    sindy = rope.rope_propagator()
    sindy.propagate_models(init_date = init_date, forward_propagation = forward_propagation)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    if n >= 1:
        slope = elapsed_time - execution_times[n-1]['execution_time_sec']
    else:
        slope = 0
    execution_times.append({
        'forward_propagation': forward_propagation,
        'execution_time_sec': elapsed_time,
        'slope': slope
    })
    print(f"Completed forward_propagation={forward_propagation} in {elapsed_time:.3f} seconds")

execution_times_df = pd.DataFrame(execution_times)


xdata = execution_times_df['forward_propagation'].values
x_fit = np.linspace(min(xdata), max(xdata), 500)
ydata = execution_times_df['execution_time_sec'].values


plt.plot(xdata, ydata, 'o-', label='Actual Execution Time')
plt.xlabel(r'Propagation Steps (days)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Forward Propagation')
plt.legend()
plt.grid(True)

for i in range(0, len(execution_times_df)):
    if i % 2 != 0:
        continue
    x = xdata[i]
    y = ydata[i]
    slope = execution_times_df['slope'][i]
    plt.annotate(f'{slope:.2f}s/day', (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8, color='blue')
plt.tight_layout()
plt.xticks(np.arange(1, max_propagation_days + 1, propagation_resolution))
plt.yticks(np.arange(0, max(execution_times_df['execution_time_sec']) + 1, .6))
# plt.savefig('./imgs/execution_time_vs_forward_propagation.png', dpi=300, bbox_inches='tight')
plt.show()

# +
simulation_points = 1200000
lat_values = np.linspace(-90, 90, simulation_points)
lt_values = np.linspace(0, 24, simulation_points)
alt_values = np.linspace(100, 1000, simulation_points)
lla_array_full = np.vstack((lat_values, lt_values, alt_values)).T
max_interp_samples = simulation_points


base_timestamp = pd.to_datetime("2003-10-29 00:00:00")
time_deltas = pd.to_timedelta(np.arange(simulation_points), unit="ms")
timestamps_full = base_timestamp + time_deltas

execution_times_interp = []
for n, interp_points in enumerate(list(range(1, max_interp_samples + 1))[::100000]):
    timestamps = timestamps_full[:interp_points]
    lla_array = lla_array_full[:interp_points]

    start_time = time.time()
    rope_density = rope.rope_data_interpolator( data = sindy )
    interpolated_models2, density_dmd2, density2, density_std2 = rope_density.interpolate(timestamps, lla_array)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    if n >= 2:
        slope = elapsed_time - execution_times_interp[n-2]['execution_time_sec']
    else:
        slope = 0
    execution_times_interp.append({
        'interp_points': interp_points,
        'execution_time_sec': elapsed_time,
        'slope': slope
    })
    print(f"Interpolated {interp_points} points in {elapsed_time:.6f} seconds")
    
execution_times_interp_df = pd.DataFrame(execution_times_interp)

xdata = execution_times_interp_df['interp_points'].values
x_fit = np.linspace(min(xdata), max(xdata), 500)
ydata = execution_times_interp_df['execution_time_sec'].values


plt.plot(xdata, ydata, 'o-', label='Actual Execution Time')
plt.xlabel(r'Interpolation points (absolute number)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Interpolation Points')
plt.legend()
plt.grid(True)

for i, interp_points in enumerate(list(execution_times_interp_df.interp_points)):
    if i % 25 != 0:
        continue
    x = xdata[i]
    y = ydata[i]
    slope = execution_times_interp_df['slope'][i]
    plt.annotate(f'{slope:.2f}s/10k-points', (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8, color='blue')
plt.tight_layout()
# plt.savefig('./imgs/execution_time_vs_interpolation_points.png', dpi=300, bbox_inches='tight')
plt.show()
# -





# +
#To import CHAMP, GRACE-FO and SWARM data, use Filezilla at 
# thermosphere.tudelft.nl
# user: anonymous
# password:

import pandas as pd
import glob

columns = [
    'date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
    'accelerometer_density', 'dens_mean', 'flag_dens', 'flag_dens_mean'
]

file_paths = sorted(glob.glob('./champ_data/*.txt'))

df_list = []

for file_path in file_paths:
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        comment='#',
        names=columns
    )
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    df.drop(columns=['date', 'time', 'GPS'], inplace=True)
    
    df_list.append(df)

champ_all = pd.concat(df_list)

champ_all.sort_index(inplace=True)

columns = [
    'date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
    'accelerometer_density', 'dens_mean', 'flag_dens', 'flag_dens_mean'
]

file_paths = sorted(glob.glob('./grace_data/*.txt'))

df_list = []

for file_path in file_paths:
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        comment='#',
        names=columns
    )
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    df.drop(columns=['date', 'time', 'GPS'], inplace=True)
    
    df_list.append(df)

grace_all = pd.concat(df_list)

grace_all.sort_index(inplace=True)

columns = [
    'date', 'time', 'GPS', 'alt', 'lon', 'lat', 'lst', 'arglat',
    'accelerometer_density'
]

file_paths = sorted(glob.glob('./swarm_data/SA_*.txt'))

df_list = []

for file_path in file_paths:
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        comment='#',
        names=columns
    )
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    df.drop(columns=['date', 'time', 'GPS'], inplace=True)
    
    df_list.append(df)

swarma_all = pd.concat(df_list)

swarma_all.sort_index(inplace=True)

file_paths = sorted(glob.glob('./swarm_data/SC_*.txt'))

df_list = []

for file_path in file_paths:
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        comment='#',
        names=columns
    )
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    df.drop(columns=['date', 'time', 'GPS'], inplace=True)
    
    df_list.append(df)

swarmc_all = pd.concat(df_list)

swarmc_all.sort_index(inplace=True)

# +
msis_version = 2.1
omega = 45

def plot_densities(outputs_df, satellite_name = 'GRACE-FO', plot_name = '2023a'):
    fig, ax = plt.subplots(1, 1, figsize=(25, 10), sharex=False)

    # Subplot 1 — Density comparison
    ax.plot(outputs_df.datetime, outputs_df.accelerometer_density, label=f"{satellite_name} acceleromenter density", color='tab:blue', linewidth=2)
    ax.plot(outputs_df.datetime, outputs_df.msis, label="NRL-MSIS 2.1", color='green', linewidth=1.5)
    ax.plot(outputs_df.datetime, outputs_df.ensemble_density, label="Ensemble density", color='orange', linewidth=2)
    ax.plot(outputs_df.datetime, outputs_df.debiased_ensemble_density, label="Debiased ensemble density", color='red', linewidth=1.)
    ax.fill_between(outputs_df.datetime, outputs_df.debiased_ensemble_density - outputs_df.density_std, outputs_df.debiased_ensemble_density + outputs_df.density_std,
                       color='red', alpha=0.2, label="Ensemble confidence interval")
    # ax.plot(outputs_df.datetime, outputs_df.dmd, label="DMD", color='black', linewidth=1.0)


    # Labels and legend
    ax.set_ylabel(r"$\rho$ (kg/m$^3$)", fontsize=14)
    ax.set_title(f"Neutral density comparison – {satellite_name} vs ROPE vs NRL-MSIS 2.1", fontsize=16)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    

    ax12 = ax.twinx()
    ax12.plot(outputs_df.datetime, outputs_df.kp_prop, label="$K_p$", color='tab:red', linestyle='--', linewidth=1)
    ax12.set_ylabel(r"$K_p$", fontsize=14)

    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=90))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_rotation(25)
        label.set_fontweight('bold')
        label.set_fontsize(12)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=25)
    # ax.set_ylim(0.15e-12, 5.2e-11)
    plt.tight_layout()
    # plt.savefig(f'./imgs/{satellite_name.lower()}_vs_rope_ensemble_{plot_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def bias_calculator(start_date, bias_propagation, dataset):

    start_debias = pd.to_datetime(start_date) - timedelta(hours= bias_propagation * 24 + delta_rho_ic + 1)
    end_debias = pd.to_datetime(start_date) - timedelta(hours= delta_rho_ic + 1)

    sindy = rope.rope_propagator(selected_bf_dict = selected_bf_dict, delta_rho_ic = delta_rho_ic)
    sindy.propagate_models(init_date = start_debias, forward_propagation = bias_propagation)
    rope_density_interpolator = rope.rope_data_interpolator( data = sindy )


    debias_dataset = dataset.loc[start_debias:end_debias]
    debias_dataset['hour'] = debias_dataset.index.hour
    debias_dataset['hour_minute'] = debias_dataset.index.minute/60 + debias_dataset['hour']
    debias_dataset['hms'] = debias_dataset.index.second + debias_dataset.index.minute*100 + debias_dataset['hour'] * 1000
    debias_dataset['t1'] = np.cos(np.pi*2.*debias_dataset['hms'].values/omega)
    debias_dataset['t2'] = np.sin(np.pi*2.*debias_dataset['hms'].values/omega)


    timestamps = debias_dataset.index.values
    latitude_values = debias_dataset.lat.values # degrees
    local_time_values = debias_dataset.lst.values # hours
    altitude_values = debias_dataset.alt.values/1000. # km
    lla_array = np.vstack((latitude_values, local_time_values, altitude_values)).T.reshape((-1, 3))


    _, _, ensemble_density, _ = rope_density_interpolator.interpolate(timestamps, lla_array)
    debias_dataset['debias_ratio'] = np.abs(debias_dataset.accelerometer_density/ensemble_density)
    return debias_dataset 

def build_output_data(start_date, end_date, interpolator, msis_version, dataset):
    
    subset = dataset.loc[start_date:end_date]
    
    timestamps = subset.index.values
    latitude_values = subset.lat.values # degrees
    local_time_values = subset.lst.values # hours
    longitude_values = subset.lon.values
    altitude_values = subset.alt.values/1000. # km
    lla_array = np.vstack((latitude_values, local_time_values, altitude_values)).T.reshape((-1, 3))

    all_models, dmd_density, ensemble_density, density_std = interpolator.interpolate(timestamps, lla_array)

    result = msis.calculate(
        timestamps,
        longitude_values,
        latitude_values,
        altitude_values, 
        geomagnetic_activity=-1,
        version = msis_version
    )

    msis_rho = result[:, 0]
    accelerometer_density = subset["accelerometer_density"].values

    t1 = interpolator.data.interval_hourly_drivers[1, :]
    t2 = interpolator.data.interval_hourly_drivers[2, :]
    t3 = interpolator.data.interval_hourly_drivers[3, :]
    t4 = interpolator.data.interval_hourly_drivers[4, :]
    kp = interpolator.data.interval_hourly_drivers[6, :]
    f10 = interpolator.data.interval_hourly_drivers[5, :]
    hourly_time_series = interpolator.data.hourly_date_series

    print(hourly_time_series.shape, kp.shape, f10.shape)
    print(hourly_time_series.min(), hourly_time_series.max())

    hourly_drivers_df = pd.DataFrame({
    'datetime': pd.to_datetime(hourly_time_series),
    'f10': f10,
    'kp': kp, 't1': t1, 't2': t2, 't3': t3, 't4': t4})
    hourly_drivers_df['datetime'] = pd.to_datetime(hourly_drivers_df['datetime'])

    outputs_df = pd.merge(
        pd.DataFrame(timestamps, columns=['datetime']),
        hourly_drivers_df,
        on='datetime',
        how='left'
    )
    print('date_series, interval_interpolated_drivers[5, :]')
    print(interpolator.data.date_series.shape, interpolator.data.interval_interpolated_drivers[5, :].shape)
    interpolated_outputs = pd.DataFrame({'datetime': interpolator.data.date_series, 
        'f10_prop': interpolator.data.interval_interpolated_drivers[5, :], 
        'kp_prop': interpolator.data.interval_interpolated_drivers[6, :]})
    
    outputs_df = pd.merge(
        outputs_df,
        interpolated_outputs,
        on='datetime',
        how='left'
    )

    print(interpolator.data.interval_interpolated_drivers.shape, interpolator.data.date_series.shape, outputs_df.shape)
    outputs_df.sort_values('datetime', inplace=True)
    outputs_df.ffill(inplace=True)
    outputs_df.reset_index(drop=True, inplace=True)
    outputs_df['lst'] = local_time_values
    outputs_df['lat'] = latitude_values
    outputs_df['ensemble_density'] = ensemble_density
    outputs_df['density_std'] = density_std
    outputs_df['accelerometer_density'] = accelerometer_density
    outputs_df['msis'] = msis_rho
    outputs_df['dmd'] = dmd_density
    

    return timestamps, accelerometer_density, ensemble_density, dmd_density, \
        density_std, msis_rho, outputs_df, latitude_values, local_time_values


def run_demo(start_date, end_date, rope_density, msis_version, dataset, bias_propagation):

    timestamps, accelerometer_density, ensemble_density, dmd_density, \
        density_std, msis_rho, outputs_df, latitudes, lst_values = \
            build_output_data(start_date, end_date, rope_density, msis_version, dataset)
    
    debias_dataset = bias_calculator(start_date, bias_propagation, dataset)
    
    debias_ratio_df = debias_dataset.groupby(['hour']).agg(mean_ratio = ('debias_ratio', 'mean')).reset_index().copy()
    debias_density_df = pd.DataFrame({'datetime': timestamps,
    'hour': pd.to_datetime(timestamps).hour,
        'hour_minute': pd.to_datetime(timestamps).hour + pd.to_datetime(timestamps).minute/60., 
            'hms': pd.to_datetime(timestamps).second + pd.to_datetime(timestamps).minute*100 + pd.to_datetime(timestamps).hour*1000,
                'den': ensemble_density})
    debias_density_df['t1'] = np.cos(np.pi*2.*debias_density_df['hms'].values/omega)
    debias_density_df['t2'] = np.sin(np.pi*2.*debias_density_df['hms'].values/omega)

    debias_density_df = pd.merge(debias_density_df, debias_ratio_df, on=['hour'], how = 'left')
    debias_density_df['debiased_density'] = debias_density_df['den'] * debias_density_df['mean_ratio']
    outputs_df['debiased_ensemble_density'] = debias_density_df.debiased_density.values   

    return outputs_df

# Latest Ridge parameters are [1., 1000, 10000, 100000]
for alpha_ridge in [1]:
    selected_bf_dict = {
            # 'poly': 10,
            # 'poly17': 500,
            # 'poly12': 1000,
            # 'poly13': 10,
            # 'poly135': 500, 
            'poly1357': 10, 
            # 'poly_sincos4': 100, 
            # 'poly_sincos7': alpha_ridge,
            # 'poly_exp1': alpha_ridge,
            # 'poly_exp2': alpha_ridge,
            # 'poly_exp12': alpha_ridge,
            # 'poly_exp22': alpha_ridge
        }
    # for basis in all_bf_dict.keys():
    #     for alpha_ridge in [1, 5, 10, 100, 1000, 10000, 100000]:

    # selected_bf_dict = {
    #         basis: alpha_ridge
    #     }

    lst_bias = 0. #-2.0 is good
    alt_bias = 0.#40.
    delta_rho_ic = 6

    lt_low = 0
    lt_high = 23.66666667

    lat_low = -87.5
    lat_high = 87.5

    alt_low = 100
    alt_high = 980

    #Check if start and end dates are compatible with forward_propagation


    # for bp in np.arange(1, 20):
    bias_propagation = 1


    start_date = "2023-01-01 12:00:00"
    end_date = "2023-01-05 18:00:00"
    forward_propagation = 5
    satellite_name = 'GRACE-FO'
    plot_name = '2023a'
    dataset = grace_all

    start_date = "2003-10-28 00:00:00"
    end_date = "2003-11-01 12:00:00"
    forward_propagation = 3
    satellite_name = 'CHAMP'
    plot_name = '2003'
    dataset = champ_all

    # start_date = "2023-05-05 00:00:00"
    # end_date = "2023-05-08 00:00:00"
    # forward_propagation = 3
    # satellite_name = 'GRACE-FO'
    # plot_name = '2023b'
    # dataset = grace_all

    # start_date = "2024-05-10 00:00:00"
    # end_date = "2024-05-15 00:00:00"
    # forward_propagation = 5
    # satellite_name = 'GRACE-FO'
    # plot_name = '2024'
    # dataset = grace_all

    sindy = rope.rope_propagator(selected_bf_dict = selected_bf_dict, delta_rho_ic = delta_rho_ic)
    sindy.propagate_models(init_date = start_date, forward_propagation = forward_propagation)
    rope_density = rope.rope_data_interpolator( data = sindy)

    outputs_df = run_demo(start_date, end_date, rope_density, msis_version, dataset, bias_propagation)
    plot_densities(outputs_df, satellite_name = satellite_name, plot_name = plot_name)

# +
msis_version = 2.1
omega = 45

def plot_densities(outputs_df, satellite_name = 'GRACE-FO', plot_name = '2023a'):
    fig, ax = plt.subplots(1, 1, figsize=(25, 10), sharex=False)

    # Subplot 1 — Density comparison
    ax.plot(outputs_df.datetime, outputs_df.accelerometer_density, label=f"{satellite_name} acceleromenter density", color='tab:blue', linewidth=2)
    ax.plot(outputs_df.datetime, outputs_df.msis, label="NRL-MSIS 2.1", color='green', linewidth=1.5)
    ax.plot(outputs_df.datetime, outputs_df.ensemble_density, label="Ensemble density", color='orange', linewidth=2)
    # ax.plot(outputs_df.datetime, outputs_df.debiased_ensemble_density, label="Debiased ensemble density", color='red', linewidth=1.)
    # ax.fill_between(outputs_df.datetime, outputs_df.debiased_ensemble_density - outputs_df.density_std, outputs_df.debiased_ensemble_density + outputs_df.density_std,
    #                    color='red', alpha=0.2, label="Ensemble confidence interval")
    # ax.plot(outputs_df.datetime, outputs_df.dmd, label="DMD", color='black', linewidth=1.0)


    # Labels and legend
    ax.set_ylabel(r"$\rho$ (kg/m$^3$)", fontsize=14)
    ax.set_title(f"Neutral density comparison – {satellite_name} vs ROPE vs NRL-MSIS 2.1", fontsize=16)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    

    ax12 = ax.twinx()
    ax12.plot(outputs_df.datetime, outputs_df.kp_prop, label="$K_p$", color='tab:red', linestyle='--', linewidth=1)
    ax12.set_ylabel(r"$K_p$", fontsize=14)

    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=90))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_rotation(25)
        label.set_fontweight('bold')
        label.set_fontsize(12)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=25)
    # ax.set_ylim(0.15e-12, 5.2e-11)
    plt.tight_layout()
    plt.savefig(f'./imgs/{satellite_name.lower()}_vs_rope_ensemble_{plot_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def bias_calculator(start_date, bias_propagation, dataset):

    start_debias = pd.to_datetime(start_date) - timedelta(hours= bias_propagation * 24 + delta_rho_ic + 1)
    end_debias = pd.to_datetime(start_date) - timedelta(hours= delta_rho_ic + 1)

    sindy = rope.rope_propagator(selected_bf_dict = selected_bf_dict, delta_rho_ic = delta_rho_ic)
    sindy.propagate_models(init_date = start_debias, forward_propagation = bias_propagation)
    rope_density_interpolator = rope.rope_data_interpolator( data = sindy )


    debias_dataset = dataset.loc[start_debias:end_debias]
    debias_dataset['hour'] = debias_dataset.index.hour
    debias_dataset['hour_minute'] = debias_dataset.index.minute/60 + debias_dataset['hour']
    debias_dataset['hms'] = debias_dataset.index.second + debias_dataset.index.minute*100 + debias_dataset['hour'] * 1000
    debias_dataset['t1'] = np.cos(np.pi*2.*debias_dataset['hms'].values/omega)
    debias_dataset['t2'] = np.sin(np.pi*2.*debias_dataset['hms'].values/omega)


    timestamps = debias_dataset.index.values
    latitude_values = debias_dataset.lat.values # degrees
    local_time_values = debias_dataset.lst.values # hours
    altitude_values = debias_dataset.alt.values/1000. # km
    lla_array = np.vstack((latitude_values, local_time_values, altitude_values)).T.reshape((-1, 3))


    _, _, ensemble_density, _ = rope_density_interpolator.interpolate(timestamps, lla_array)
    debias_dataset['debias_ratio'] = np.abs(debias_dataset.accelerometer_density/ensemble_density)
    return debias_dataset 

def build_output_data(start_date, end_date, interpolator, msis_version, dataset):
    
    subset = dataset.loc[start_date:end_date]
    
    timestamps = subset.index.values
    latitude_values = subset.lat.values # degrees
    local_time_values = subset.lst.values # hours
    longitude_values = subset.lon.values
    altitude_values = subset.alt.values/1000. # km
    lla_array = np.vstack((latitude_values, local_time_values, altitude_values)).T.reshape((-1, 3))

    all_models, dmd_density, ensemble_density, density_std = interpolator.interpolate(timestamps, lla_array)
    
    result = msis.calculate(
        timestamps,
        longitude_values,
        latitude_values,
        altitude_values, 
        geomagnetic_activity=-1,
        version = msis_version
    )

    msis_rho = result[:, 0]
    accelerometer_density = subset["accelerometer_density"].values

    t1 = interpolator.data.interval_hourly_drivers[1, :]
    t2 = interpolator.data.interval_hourly_drivers[2, :]
    t3 = interpolator.data.interval_hourly_drivers[3, :]
    t4 = interpolator.data.interval_hourly_drivers[4, :]
    kp = interpolator.data.interval_hourly_drivers[6, :]
    f10 = interpolator.data.interval_hourly_drivers[5, :]
    hourly_time_series = interpolator.data.hourly_date_series

    print(hourly_time_series.shape, kp.shape, f10.shape)
    print(hourly_time_series.min(), hourly_time_series.max())

    hourly_drivers_df = pd.DataFrame({
    'datetime': pd.to_datetime(hourly_time_series),
    'f10': f10,
    'kp': kp, 't1': t1, 't2': t2, 't3': t3, 't4': t4})
    hourly_drivers_df['datetime'] = pd.to_datetime(hourly_drivers_df['datetime'])

    outputs_df = pd.merge(
        pd.DataFrame(timestamps, columns=['datetime']),
        hourly_drivers_df,
        on='datetime',
        how='left'
    )
    print('date_series, interval_interpolated_drivers[5, :]')
    print(interpolator.data.date_series.shape, interpolator.data.interval_interpolated_drivers[5, :].shape)
    interpolated_outputs = pd.DataFrame({'datetime': interpolator.data.date_series, 
        'f10_prop': interpolator.data.interval_interpolated_drivers[5, :], 
        'kp_prop': interpolator.data.interval_interpolated_drivers[6, :]})
    
    outputs_df = pd.merge(
        outputs_df,
        interpolated_outputs,
        on='datetime',
        how='left'
    )

    print(interpolator.data.interval_interpolated_drivers.shape, interpolator.data.date_series.shape, outputs_df.shape)
    outputs_df.sort_values('datetime', inplace=True)
    outputs_df.ffill(inplace=True)
    outputs_df.reset_index(drop=True, inplace=True)
    outputs_df['lst'] = local_time_values
    outputs_df['lat'] = latitude_values
    outputs_df['ensemble_density'] = ensemble_density
    outputs_df['density_std'] = density_std
    outputs_df['accelerometer_density'] = accelerometer_density
    outputs_df['msis'] = msis_rho
    outputs_df['dmd'] = dmd_density

    return timestamps, accelerometer_density, ensemble_density, dmd_density, \
        density_std, msis_rho, outputs_df, latitude_values, local_time_values


def run_demo(start_date, end_date, rope_density, msis_version, dataset, bias_propagation):

    timestamps, accelerometer_density, ensemble_density, dmd_density, \
        density_std, msis_rho, outputs_df, latitudes, lst_values = \
            build_output_data(start_date, end_date, rope_density, msis_version, dataset)
    
    
    
    debias_dataset = bias_calculator(start_date, bias_propagation, dataset)
    
    debias_ratio_df = debias_dataset.groupby(['hour']).agg(mean_ratio = ('debias_ratio', 'mean')).reset_index().copy()
    debias_density_df = pd.DataFrame({'datetime': timestamps,
    'hour': pd.to_datetime(timestamps).hour,
        'hour_minute': pd.to_datetime(timestamps).hour + pd.to_datetime(timestamps).minute/60., 
            'hms': pd.to_datetime(timestamps).second + pd.to_datetime(timestamps).minute*100 + pd.to_datetime(timestamps).hour*1000,
                'den': ensemble_density})
    debias_density_df['t1'] = np.cos(np.pi*2.*debias_density_df['hms'].values/omega)
    debias_density_df['t2'] = np.sin(np.pi*2.*debias_density_df['hms'].values/omega)

    debias_density_df = pd.merge(debias_density_df, debias_ratio_df, on=['hour'], how = 'left')
    debias_density_df['debiased_density'] = debias_density_df['den'] * debias_density_df['mean_ratio']
    outputs_df['debiased_ensemble_density'] = debias_density_df.debiased_density.values   

    return outputs_df

# Latest Ridge parameters are [1., 1000, 10000, 100000]
for alpha_ridge in [7]:
    selected_bf_dict = {
            'poly': 100,
            # 'poly17': alpha_ridge,
            'poly12': 1000,
            # 'poly13': alpha_ridge,
            'poly135': alpha_ridge, 
            # 'poly_sincos4': 100, 
            # 'poly_sincos7': alpha_ridge,
            # 'poly_exp1': alpha_ridge,
            # 'poly_exp2': alpha_ridge,
            # 'poly_exp12': alpha_ridge,
            # 'poly_exp22': alpha_ridge
        }
    # for basis in all_bf_dict.keys():
    #     for alpha_ridge in [1, 5, 10, 100, 1000, 10000, 100000]:

    # selected_bf_dict = {
    #         basis: alpha_ridge
    #     }

    lst_bias = 0. #-2.0 is good
    alt_bias = 0.#40.
    delta_rho_ic = 6

    lt_low = 0
    lt_high = 23.66666667

    lat_low = -87.5
    lat_high = 87.5

    alt_low = 100
    alt_high = 980

    #Check if start and end dates are compatible with forward_propagation


    # for bp in np.arange(1, 20):
    bias_propagation = alpha_ridge


    # start_date = "2023-01-01 12:00:00"
    # end_date = "2023-01-05 18:00:00"
    # forward_propagation = 5
    # satellite_name = 'GRACE-FO'
    # plot_name = '2023a'
    # dataset = grace_all

    # start_date = "2003-10-28 00:00:00"
    # end_date = "2003-11-01 12:00:00"
    # forward_propagation = 5
    # satellite_name = 'CHAMP'
    # plot_name = '2003'
    # dataset = champ_all

    # start_date = "2023-05-05 00:00:00"
    # end_date = "2023-05-08 00:00:00"
    # forward_propagation = 3
    # satellite_name = 'GRACE-FO'
    # plot_name = '2023b'
    # dataset = grace_all

    start_date = "2024-05-09 12:00:00"
    end_date = "2024-05-15 00:00:00"
    forward_propagation = 5
    satellite_name = 'GRACE-FO'
    plot_name = '2024'
    dataset = grace_all

    sindy = rope.rope_propagator(selected_bf_dict = selected_bf_dict, delta_rho_ic = delta_rho_ic)
    sindy.propagate_models(init_date = start_date, forward_propagation = forward_propagation)
    rope_density = rope.rope_data_interpolator( data = sindy)

    outputs_df = run_demo(start_date, end_date, rope_density, msis_version, dataset, bias_propagation)
    plot_densities(outputs_df, satellite_name = satellite_name, plot_name = plot_name)
# -

plt.plot(sindy.z_results_lst[0][0, :2450])

# +

subset = dataset.loc[start_date:end_date].copy()

timestamps = subset.index.values
latitude_values = subset.lat.values # degrees
local_time_values = subset.lst.values # hours
longitude_values = subset.lon.values
altitude_values = subset.alt.values/1000. # km
lla_array = np.vstack((latitude_values, local_time_values, altitude_values)).T.reshape((-1, 3))


sindy = rope.rope_propagator(selected_bf_dict = selected_bf_dict)
sindy.propagate_models(init_date = start_date, forward_propagation = forward_propagation)


rope_density = rope.rope_data_interpolator( data = sindy )

all_models, dmd_density, ensemble_density, density_std = rope_density.interpolate(timestamps, lla_array)

ensemble_density
# -

plt.plot(ensemble_density)

sindy.z_results_lst[0].shape

# +
num_nans = np.isnan(sindy.z_results_lst[0][0, :]).sum()
print(f"Number of NaNs: {num_nans}")

# Get indices of NaNs
nan_indices = np.where(np.isnan(ensemble_density))[0]
print(f"Indices of NaNs: {nan_indices}")

# +
msis_version = 2.1
omega = 45

def plot_densities(outputs_df, satellite_name = 'GRACE-FO', plot_name = '2023a'):
    fig, ax = plt.subplots(1, 1, figsize=(25, 10), sharex=False)

    # Subplot 1 — Density comparison
    ax.plot(outputs_df.datetime, outputs_df.accelerometer_density, label=f"{satellite_name} acceleromenter density", color='tab:blue', linewidth=2)
    ax.plot(outputs_df.datetime, outputs_df.msis, label="NRL-MSIS 2.1", color='green', linewidth=1.5)
    ax.plot(outputs_df.datetime, outputs_df.ensemble_density, label="Ensemble density", color='orange', linewidth=2)
    # ax.plot(outputs_df.datetime, outputs_df.debiased_ensemble_density, label="Debiased ensemble density", color='red', linewidth=1.)
    # ax.fill_between(outputs_df.datetime, outputs_df.ensemble_density - outputs_df.density_std, outputs_df.ensemble_density + outputs_df.density_std,
    #                    color='orange', alpha=0.2, label="Ensemble confidence interval")
    ax.plot(outputs_df.datetime, outputs_df.dmd, label="DMD", color='black', linewidth=1.0)


    # Labels and legend
    ax.set_ylabel(r"$\rho$ (kg/m$^3$)", fontsize=14)
    ax.set_title(f"Neutral density comparison – {satellite_name} vs ROPE vs NRL-MSIS 2.1", fontsize=16)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    

    ax12 = ax.twinx()
    ax12.plot(outputs_df.datetime, outputs_df.kp_prop, label="$K_p$", color='tab:red', linestyle='--', linewidth=1)
    ax12.set_ylabel(r"$K_p$", fontsize=14)

    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=90))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_rotation(25)
        label.set_fontweight('bold')
        label.set_fontsize(12)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=25)
    # ax.set_ylim(0.15e-12, 5.2e-11)
    plt.tight_layout()
    # plt.savefig(f'./imgs/{satellite_name.lower()}_vs_rope_ensemble_{plot_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def bias_calculator(start_date, bias_propagation, dataset):

    start_debias = pd.to_datetime(start_date) - timedelta(hours= bias_propagation * 24 + delta_rho_ic + 1)
    end_debias = pd.to_datetime(start_date) - timedelta(hours= delta_rho_ic + 1)

    sindy = rope.rope_propagator(selected_bf_dict = selected_bf_dict, delta_rho_ic = delta_rho_ic)
    sindy.propagate_models(init_date = start_debias, forward_propagation = bias_propagation)
    rope_density_interpolator = rope.rope_data_interpolator( data = sindy )


    debias_dataset = dataset.loc[start_debias:end_debias]
    debias_dataset['hour'] = debias_dataset.index.hour
    debias_dataset['hour_minute'] = debias_dataset.index.minute/60 + debias_dataset['hour']
    debias_dataset['hms'] = debias_dataset.index.second + debias_dataset.index.minute*100 + debias_dataset['hour'] * 1000
    debias_dataset['t1'] = np.cos(np.pi*2.*debias_dataset['hms'].values/omega)
    debias_dataset['t2'] = np.sin(np.pi*2.*debias_dataset['hms'].values/omega)


    timestamps = debias_dataset.index.values
    latitude_values = debias_dataset.lat.values # degrees
    local_time_values = debias_dataset.lst.values # hours
    altitude_values = debias_dataset.alt.values/1000. # km
    lla_array = np.vstack((latitude_values, local_time_values, altitude_values)).T.reshape((-1, 3))


    _, _, ensemble_density, _ = rope_density_interpolator.interpolate(timestamps, lla_array)
    debias_dataset['debias_ratio'] = np.abs(debias_dataset.accelerometer_density/ensemble_density)
    return debias_dataset 

def build_output_data(start_date, end_date, interpolator, msis_version, dataset):
    
    subset = dataset.loc[start_date:end_date]
    
    timestamps = subset.index.values
    latitude_values = subset.lat.values # degrees
    local_time_values = subset.lst.values # hours
    longitude_values = subset.lon.values
    altitude_values = subset.alt.values/1000. # km
    lla_array = np.vstack((latitude_values, local_time_values, altitude_values)).T.reshape((-1, 3))

    all_models, dmd_density, ensemble_density, density_std = interpolator.interpolate(timestamps, lla_array)

    result = msis.calculate(
        timestamps,
        longitude_values,
        latitude_values,
        altitude_values, 
        geomagnetic_activity=-1,
        version = msis_version
    )

    msis_rho = result[:, 0]
    accelerometer_density = subset["accelerometer_density"].values

    t1 = interpolator.data.interval_hourly_drivers[1, :]
    t2 = interpolator.data.interval_hourly_drivers[2, :]
    t3 = interpolator.data.interval_hourly_drivers[3, :]
    t4 = interpolator.data.interval_hourly_drivers[4, :]
    kp = interpolator.data.interval_hourly_drivers[6, :]
    f10 = interpolator.data.interval_hourly_drivers[5, :]
    hourly_time_series = interpolator.data.hourly_date_series

    print(hourly_time_series.shape, kp.shape, f10.shape)
    print(hourly_time_series.min(), hourly_time_series.max())

    hourly_drivers_df = pd.DataFrame({
    'datetime': pd.to_datetime(hourly_time_series),
    'f10': f10,
    'kp': kp, 't1': t1, 't2': t2, 't3': t3, 't4': t4})
    hourly_drivers_df['datetime'] = pd.to_datetime(hourly_drivers_df['datetime'])

    outputs_df = pd.merge(
        pd.DataFrame(timestamps, columns=['datetime']),
        hourly_drivers_df,
        on='datetime',
        how='left'
    )
    print(interpolator.data.date_series.shape, interpolator.data.interval_interpolated_drivers[5, :].shape)
    interpolated_outputs = pd.DataFrame({'datetime': interpolator.data.date_series, 
        'f10_prop': interpolator.data.interval_interpolated_drivers[5, :], 
        'kp_prop': interpolator.data.interval_interpolated_drivers[6, :]})
    
    outputs_df = pd.merge(
        outputs_df,
        interpolated_outputs,
        on='datetime',
        how='left'
    )

    print(interpolator.data.interval_interpolated_drivers.shape, interpolator.data.date_series.shape, outputs_df.shape)
    outputs_df.sort_values('datetime', inplace=True)
    outputs_df.ffill(inplace=True)
    outputs_df.reset_index(drop=True, inplace=True)
    outputs_df['lst'] = local_time_values
    outputs_df['lat'] = latitude_values
    outputs_df['ensemble_density'] = ensemble_density
    outputs_df['density_std'] = density_std
    outputs_df['accelerometer_density'] = accelerometer_density
    outputs_df['msis'] = msis_rho
    outputs_df['dmd'] = dmd_density
    

    return timestamps, accelerometer_density, ensemble_density, dmd_density, \
        density_std, msis_rho, outputs_df, latitude_values, local_time_values


def run_demo(start_date, end_date, rope_density, msis_version, dataset, bias_propagation):

    timestamps, accelerometer_density, ensemble_density, dmd_density, \
        density_std, msis_rho, outputs_df, latitudes, lst_values = \
            build_output_data(start_date, end_date, rope_density, msis_version, dataset)
    
    debias_dataset = bias_calculator(start_date, bias_propagation, dataset)
    
    debias_ratio_df = debias_dataset.groupby(['hour']).agg(mean_ratio = ('debias_ratio', 'mean')).reset_index().copy()
    debias_density_df = pd.DataFrame({'datetime': timestamps,
    'hour': pd.to_datetime(timestamps).hour,
        'hour_minute': pd.to_datetime(timestamps).hour + pd.to_datetime(timestamps).minute/60., 
            'hms': pd.to_datetime(timestamps).second + pd.to_datetime(timestamps).minute*100 + pd.to_datetime(timestamps).hour*1000,
                'den': ensemble_density})
    debias_density_df['t1'] = np.cos(np.pi*2.*debias_density_df['hms'].values/omega)
    debias_density_df['t2'] = np.sin(np.pi*2.*debias_density_df['hms'].values/omega)

    debias_density_df = pd.merge(debias_density_df, debias_ratio_df, on=['hour'], how = 'left')
    debias_density_df['debiased_density'] = debias_density_df['den'] * debias_density_df['mean_ratio']
    outputs_df['debiased_ensemble_density'] = debias_density_df.debiased_density.values   

    return outputs_df

# Latest Ridge parameters are [1., 1000, 10000, 100000]
for alpha_ridge in [1]:
    selected_bf_dict = {
            'poly': alpha_ridge,
            # 'poly17': alpha_ridge,
            # 'poly12': alpha_ridge,
            # 'poly135': alpha_ridge, 
            # 'poly_sincos4': 1000, 
            # 'poly_sincos7': alpha_ridge,
            # 'poly_exp1': 1.0,
            # 'poly_exp2': 1,
            # 'poly_exp12': 1000,
            # 'poly_exp22': alpha_ridge
        }
    # for basis in all_bf_dict.keys():
    #     for alpha_ridge in [1, 5, 10, 100, 1000, 10000, 100000]:

    # selected_bf_dict = {
    #         basis: alpha_ridge
    #     }

    lst_bias = 0. #-2.0 is good
    alt_bias = 0.#40.
    delta_rho_ic = 0

    lt_low = 0
    lt_high = 23.66666667

    lat_low = -87.5
    lat_high = 87.5

    alt_low = 100
    alt_high = 980

    #Check if start and end dates are compatible with forward_propagation


    # for bp in np.arange(1, 20):
    bias_propagation = 1


    start_date = "2023-01-01 12:00:00"
    end_date = "2023-01-05 18:00:00"
    forward_propagation = 5
    satellite_name = 'GRACE-FO'
    plot_name = '2023a'
    dataset = grace_all

    start_date = "2003-10-28 00:00:00"
    end_date = "2003-11-01 12:00:00"
    forward_propagation = 5
    satellite_name = 'CHAMP'
    plot_name = '2003'
    dataset = champ_all

    # start_date = "2023-05-05 00:00:00"
    # end_date = "2023-05-08 00:00:00"
    # forward_propagation = 3
    # satellite_name = 'GRACE-FO'
    # plot_name = '2023b'
    # dataset = grace_all

    # start_date = "2024-05-10 00:00:00"
    # end_date = "2024-05-15 00:00:00"
    # forward_propagation = 5
    # satellite_name = 'GRACE-FO'
    # plot_name = '2024'
    # dataset = grace_all

    sindy = rope.rope_propagator(selected_bf_dict = selected_bf_dict, delta_rho_ic = delta_rho_ic)
    sindy.propagate_models(init_date = start_date, forward_propagation = forward_propagation)
    rope_density = rope.rope_data_interpolator( data = sindy)

    outputs_df = run_demo(start_date, end_date, rope_density, msis_version, dataset, bias_propagation)
    plot_densities(outputs_df, satellite_name = satellite_name, plot_name = plot_name)
# -















