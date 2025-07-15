import numpy as np
from datetime import datetime, timedelta
import orekit
import time
import pymsis
from pymsis import msis
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import hrd_20250608.utilities_ds as u
from orekit.pyhelpers import download_orekit_data_curdir, setup_orekit_curdir
from os import path
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.linalg import expm, logm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import hrd_20250608.rope_class_hrd as rope 


# download_orekit_data_curdir( 'hrd_20250608/orekit-data.zip' )  # Comment this out once this file has already been downloaded for repeated runs
vm = orekit.initVM()
setup_orekit_curdir( './hrd_20250608/' )

print ( 'Java version:', vm.java_version )
print ( 'Orekit version:', orekit.VERSION )

# User inputs
forward_propagation_mins = 1  # forward propagation time (mins)
init_date = pd.to_datetime('2003-05-10 00:00:00')

latitude_values = [89.]  # degrees
local_time_values = [8.]  # hours
altitude_values = [358.]  # km
lla_array = np.vstack((latitude_values, local_time_values, altitude_values)).T.reshape((-1, 3))

# Prepare to store results
all_dmd_outputs = []

# Loop over 60 minutes, propagating 1 min at a time
current_date = init_date
for i in range(60):
    sindy = rope.rope_propagator()
    sindy.propagate_models_mins(init_date=current_date, forward_propagation=forward_propagation_mins)
    all_dmd_outputs.append(sindy.z_dict['dmd'])
    current_date += pd.Timedelta(minutes=forward_propagation_mins)

# Concatenate all outputs along the time axis 
all_dmd_outputs = np.concatenate(all_dmd_outputs, axis=1)
print(all_dmd_outputs.shape)
df = pd.DataFrame(data=all_dmd_outputs)
                  
# Save the DataFrame to an Excel sheet
excel_file_path = 'dmd_outputs.xlsx'
df.to_excel(excel_file_path)
print(f'Data saved to {excel_file_path}')

# rope_density = rope.rope_data_interpolator( data = sindy )

# all_models, dmd_density, ensemble_density, density_std = rope_density.interpolate(timestamps, lla_array)

# ensemble_density, density_std


    
