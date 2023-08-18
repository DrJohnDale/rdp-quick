# rdp-quick

There are several implementations of the Ramer-Douglas-Peucker, however, they can be slow
for large datasets

In this library the Ramer-Douglas-Peucker is implemented using Numba (with eager compilation and cache) to speed up the algorithm

For very large sets starting we know that there will be sub-windows so we have functions to create more initial windows to speed up futher.
These include setting the number of windows, setting the number of points per window and computing the curvature of the route to determine starting windows

# Basic usage
## Setup
pip install rdp-quick

## Single Window

The simplest usage is to start with one window using the start and end point.

```python
from rdp_quick import rdp_single_initial_window
import numpy as np

num_osc = 20
points_per_osc = 200
epsilon = 0.01

print("building points")
n_pts = num_osc*points_per_osc
x = np.arange(n_pts)/n_pts*num_osc*2.0*np.pi
y = np.sin(x)
p = np.vstack([x, y]).astype(float).transpose()

down_sampled_p = rdp_single_initial_window(p, epsilon)

print(down_sampled_p.shape, p.shape)
```

## Using Curvature to Initialise
A more advanced method is to first compute the curvature and find peaks and use these peaks to define the initial windows

For large datasets this should be much quicker than the basic usage

This method uses numpy gradient and scipy.signal find_peaks. Both these methods can use a number of named arguments.  
If you want to set these use the input dictionaries ```gradient_nargs``` and ```find_peaks_nargs``` in ```rdp_windows_from_curvature```

```python
from rdp_quick import rdp_windows_from_curvature
import numpy as np

num_osc = 20
points_per_osc = 200
epsilon = 0.01

print("building points")
n_pts = num_osc*points_per_osc
x = np.arange(n_pts)/n_pts*num_osc*2.0*np.pi
y = np.sin(x)
p = np.vstack([x, y]).astype(float).transpose()

down_sampled_p = rdp_windows_from_curvature(p, epsilon)

print(down_sampled_p.shape, p.shape)
```

## Define Initial Number of Points Per Window
A simpler way of creating initial windows is to define the number of points per window.  The algorithm will then create 
windows of this length (it may shorten or extend to better match the number of points) to initialise the algorith with.
Note: this will be faster that the basic usage, but may not be as optimum

```python
from rdp_quick import rdp_points_per_window
import numpy as np

num_osc = 20
points_per_osc = 200
epsilon = 0.01

print("building points")
n_pts = num_osc*points_per_osc
x = np.arange(n_pts)/n_pts*num_osc*2.0*np.pi
y = np.sin(x)
p = np.vstack([x, y]).astype(float).transpose()

down_sampled_p = rdp_points_per_window(p, epsilon, points_per_osc)

print(down_sampled_p.shape, p.shape)
```

## Define Initial Number of Windows
A simpler way of creating initial windows is to define the number of inital windows.  The algorithm will then create 
this number of windows to initialise the algorith with.
Note: this will be faster that the basic usage, but may not be as optimum

```python
from rdp_quick import rdp_num_windows
import numpy as np

num_osc = 20
points_per_osc = 200
epsilon = 0.01

print("building points")
n_pts = num_osc*points_per_osc
x = np.arange(n_pts)/n_pts*num_osc*2.0*np.pi
y = np.sin(x)
p = np.vstack([x, y]).astype(float).transpose()

down_sampled_p = rdp_num_windows(p, epsilon, num_osc)

print(down_sampled_p.shape, p.shape)
```

## examples
https://github.com/DrJohnDale/rdp-quick/blob/main/example.py 
https://github.com/DrJohnDale/rdp-quick/blob/main/example_curvature.py

### Development
- Clone the repo
- Create virtual env
- pip install -r requirements.txt
- pip install -r requirements_test_and_dev.txt
- Use run_test.bat or run_text.sh to run the tests
- Run examples to see usage



