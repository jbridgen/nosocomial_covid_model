# Manuscript code: A Bayesian approach to identifying the role of hospital structure and staff interactions in nosocomial transmission of SARS-CoV-2
Authors: Jessica R.E. Bridgen, Joseph M. Lewis, Stacy Todd, Miriam Taegtmeyer, Jonathan M. Read, Chris P. Jewell

Python implementation of a stochastic continuous time SEIR model for nosocomial transmission of SARS-CoV-2. A Metropolis-within-Gibbs algorithm is used to conduct inference on transmission rate parameters and unobserved SE and IR event times.

### Required data structures: 
| File name                | File Type                           | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|--------------------------|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| pat_pos_swab.csv         | csv                                 | **Data structure:** [num_individuals, 2]<br>**Columns:**<br>pid: string<br>Date_Time_Collected: %d/%m/%Y                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ward_colours.csv         | csv                                 | **Data structure:** [num_dates, num_wards]<br>**Columns:**<br>date: %d/%m/%Y <br>wardx: string<br><br>A column for each ward with the ward name as the header row. Subsequent entries list ward colours e.g. ‘red’                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| pids_saved               | Compressed pickled pandas DataFrame | **Data structure:** [num_individuals, 2]<br>**Columns:**<br>pid: string<br>pid_index: int64                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| wards_saved              | Compressed pickled numpy ndarray    | numpy.ndarray of ward names                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| connectivity_matrices.nc | NetCDF4 compressed xarray Dataset   | **Coordinates:**<br>pid: unique identifier - object<br>sWardLocation: ward name - object<br>time: datetime64<br>wards: ward name - object<br><br>**Data variables**<br>memb_mats (time, pid, sWardLocation) float64: 1 if individual is a member of ward at time t, 0 otherwise<br><br>adj_mats (time, sWardLocation, wards) float64: weighted ward connectivity matrix (C)<br><br>hospital_status_mats (time, pid) float64 : 1 if individual is admitted to the hospital at time t, 0 otherwise<br><br>study_status_mats (time, pid) int64: 1 if individual has been admitted to the hospital by time t and has therefore been initialised in the model, 0 otherwise<br><br>spatial_conn_mats (time, sWardLocation, wards) int64:weighted spatial adjacency matrix (W). |