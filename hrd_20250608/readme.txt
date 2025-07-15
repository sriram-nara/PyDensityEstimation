# TIE-GCM-ROPE usage instructions and brief description

-works at altitude between 100 km and 980 km, but can extrapolate outside the boundaries.
-Initial condition uses a simple database classified using kp and f10.7 bins 
(this will eventually be replaced with a nowcast achieved through data assimilation)
- Current version uses Celestrack drivers up to 2025. In the future, the most up to date Celestrack file 
will be downloaded automatically. The user can aklso provide his own drivers file.

Some backend information
The user has to enter initial date like

init_date = pd.to_datetime('2023-05-03 00:00:00'),

initial local solar time, latitude and altitude like

latitude_values = [-90., 89] # degrees
local_time_values = [20.7, 8] # hours
altitude_values = [570., 456] # km

then, by executing 

forward_propagation = 1
lla_array = np.vstack((latitude_values, local_time_values, altitude_values)).T.reshape((-1, 3))
timestamps = [pd.to_datetime('2023-05-03 00:00:00'), pd.to_datetime('2023-05-03 00:00:00')]
sindy = rope.rope_propagator()
sindy.propagate_models(init_date = init_date, forward_propagation = forward_propagation)

the system propagates over the required amount of days by the forward_propagation variable.
The user then calls the interpolator with

rope_density = rope.rope_data_interpolator( data = sindy )
all_models, dmd_density, ensemble_density, density_std = rope_density.interpolate_full_grid(timestamps, lla_array)

and the ensemble density is calculated and contained in the variable ensemble_density.


########################################################################################################################################

The initial conditions to the propagation are built using a classification for the initial vectors based on 
kp and f10.7 drivers table.
The propagation begins 6 hours before the date specified by the user to make sure that the system aligns with 
the external drivers by the time entered by the user.
The user then adds latitude (in degrees), local time (in hours) and altitudes (in km) at which to interpolate 
the thermospheric grid. 

##############################################Environment installation####################################################################
To install the environment to run the tool, you need to run from terminal the following command:

conda env create -f ./tie_gcm_rope_env.yml

after installing conda system-wide

##################################################################################################################
Before running everithing make sure that orekit-data.zip file is available in the current folder and is called 
through the code
vm = orekit.initVM()
setup_orekit_curdir( './' )
##################################################################################################################

The main body of the package comes in main.ipynb
In this file we perform comparison between the ensemble and the GRACE-FO accelerometer derived densities up to nowadays. 
The file contains plots and other analysis that can be customized. 

###############################################TIE-GCM-ROPE inputs###################################################################
latitude range : [-90, 90] deg
local solar time range: [0, 23.6667] hours
altitude range: [100, 980] km
timestamps is an array of strings representing date-time in the format 'YYYY-mm-dd HH:MM:SS'

In the following, we list a couple of ideas to let the user exercise with the tool.

The inputs that the user must provide to use the emulator are the datetime or datetimes inputs in the from

'2003-11-03 00:00:00'

or if there is more than 1 timestamp they must be provided as an array for instance as

timestamps = np.array(['2003-11-03 00:00:00', '2003-11-03 10:00:00'])

At the same time the set of latitudes, local times and altitudes must be provided as an (n, 3) numpy.array.
For instance you can provide.

init_date_str = '2003-11-03 00:00:00' #which always signals the initial propagation date
lla_array = np.array([[87., 23.7, 440.5]])
forward_propagation = 6

as they are specified in the example, and the outputs will be 

{'density': array([1.93864391e-12]), 'standard deviation': array([7.20897136e-14])}

Another example with more than one input is 

init_date_str = '2003-11-03 00:00:00' #which always signals the initial propagation date
timestamps = np.array(['2003-11-03 04:00:00', '2003-11-03 10:00:00'])
lla_array = np.array([[87., 23.7, 440.5], [87., 23.7, 440.5]])
forward_propagation = 6

whose output is 

{'density': array([2.01949620e-12, 1.81244489e-12]), 'standard deviation': array([8.14883348e-14, 1.10739710e-13])}

##################################################################################################################
The propagation occurs at the row 

sindy.propagate_models(init_date = init_date, forward_propagation = forward_propagation)

Following, the interpolator is defined at the row

rope_density = rope.rope_data_interpolator( data = sindy )

by feeding it with the sindy object containing the propagated variables.

The interpolated density is calculated at the row 

density_poly, density_poly_all, density_dmd, density, density_std = rope_density.interpolate(timestamps, lla_array)

where density and density_std are the ensemble mean and the ensemble uncertainty. The other outputs are specific
densities corresponding to particular basis functions or other models. If they are of no interest one can just 
cover those outputs by using 

_, _, _, density, density_std = rope_density.interpolate(timestamps, lla_array)

