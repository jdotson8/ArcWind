# ArcWind
ArcWind is a 2D fluid simulation extension for ArcGIS Pro. 
The fluid solver is a modified version of Jos Stam's paper "Stable Fluids" with vorticity confinement.
Each stage of the solver is implemented as a seperate OpenCL kernel program.

Installation and Setup

1. Install the OpenCL runtime drivers and environment from Intel site.
2. Install Python 3.4.3
3. Using Pip command or by using WHL directly, install Scipy, Mingwpy, Numpy, and PyOpenCL on the Python 3.4.3 install in this order.
4. Open command line in Python 3.4.3 and check whether OpenCL is installed correctly. Use steps at: http://karthikhegde.blogspot.com/2013/09/hope-you-liked-previous-introductory.html
5. In python code, the import should explicitly point to the python 3.4.3 library locations.

Example:
sys.path.append('C:\Python34\Lib\site-packages')
import pyopencl as cl

