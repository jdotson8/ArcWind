import os
import sys
import arcpy
import numpy
import math
import datetime
from arcpy.sa import *
sys.path.append('C:\Python34\Lib\site-packages')
import pyopencl as cl

programSource = """
    __kernel void gaussSeidel(
        const int row,
        const int column,
        const float height,
        const float a,
        const float invC,
        __global float* x,
        __global float* x0,
        __global float* city
    ) {
        int index = get_global_id(0);
        int i = index % column;
        int j = index / column;
        for(int k = 0; k < 20; k++) {
            if( city[i + j * column] < height) {
                float left, right, up, down;

                if (i == 0) left = 0;
                else if ( city[(i - 1) + j * column] >= height) left = x[i + j * column];
                else left = x[(i - 1) + j * column];

                if (i == (column - 1)) right = 0;
                else if (city[(i + 1) + j * column] >= height) right = x[i + j * column];
                else right = x[(i + 1) + j * column];

                if (j == 0) up = 0;
                else if (city[i + (j - 1) * column] >= height) up = x[i + j * column];
                else up = x[i + (j - 1) * column];

                if(j == (row - 1)) down = 0;
                else if(city[i + (j + 1) * column] >= height) down = x[i + j * column];
                else down = x[i + (j + 1) * column];

                x[i + j * column] = invC * (x0[i + j * column] + a*(left + right + up + down));
            }
        }
    }

    __kernel void advect(
        const int row,
        const int column,
        const float height,
        const float dt,
        __global float* x,
        __global float* x0,
        __global float* u,
        __global float* v,
        __global float* city
    ) {
        int index = get_global_id(0);
        int i = index % column;
        int j = index / column;

        if (city[i + j * column] < height) {
            float posX, posY, s0, s1, t0, t1, x00, x10, x01, x11;
            int i0, i1, j0, j1;

            posX = clamp(i - dt * u[i + j * column], -0.5f, column - 0.5f);
            posY = clamp(j - dt * v[i + j * column], -0.5f, row - 0.5f);
            i0 = (int) (posX + 1) - 1;
            i1 = i0 + 1;
            j0 = (int) (posY + 1) - 1;
            j1 = j0 + 1;
            s1 = posX - i0;
            s0 = 1 - s1;
            t1 = posY - j0;
            t0 = 1 - t1;

            x00 = (i0 < 0 || j0 < 0) ? 0 : x0[i0 + j0 * column];
            x01 = (i0 < 0 || j1 >= row) ? 0 : x0[i0 + j1 * column];
            x10 = (i1 >= column || j0 < 0) ? 0 : x0[i1 + j0 * column];
            x11 = (i1 >= column || j1 >= row) ? 0 : x0[i1 + j1 * column];

            x[i + j * column] = s0 * (t0 * x00 + t1 * x01) + s1 * (t0 * x10 + t1 * x11); 
        }
    }

    __kernel void divergence(
        const int row,
        const int column,
        const float height,
        const float cellSize,
        __global float* div,
        __global float* u,
        __global float* v,
        __global float* city
    ) {
        int index = get_global_id(0);
        int i = index % column;
        int j = index / column;

        float left, right, up, down;

        if (i == 0) left = u[i + j * column];
        else if ( city[(i - 1) + j * column] >= height) left = 0;
        else left = u[(i - 1) + j * column];

        if (i == (column - 1)) right = u[i + j * column];
        else if (city[(i + 1) + j * column] >= height) right = 0;
        else right = u[(i + 1) + j * column];

        if (j == 0) up = v[i + j * column];
        else if (city[i + (j - 1) * column] >= height) up = 0;
        else up = v[i + (j - 1) * column];

        if(j == (row - 1)) down = v[i + j * column];
        else if(city[i + (j + 1) * column] >= height) down = 0;
        else down = v[i + (j + 1) * column];

        div[i + j  * column] = -0.5 * cellSize * (right - left + down - up);
    }

    __kernel void subPressureGrad(
        const int row,
        const int column,
        const float height,
        const float invCellSize,
        __global float* u,
        __global float* v,
        __global float* p,
        __global float* city
    ) {
        int index = get_global_id(0);
        int i = index % column;
        int j = index / column;

        if (city[i + j * column] < height) {
            float left, right, up, down, currentU, currentV;

            bool zeroU = city[(i - 1) + j * column] >= height || city[(i + 1) + j * column] >= height;
            bool zeroV = city[i + (j - 1) * column] >= height || city[i + (j + 1) * column] >= height;

            if (i == 0) left = p[i + j * column];
            else left = p[(i - 1) + j * column];

            if (i == (column - 1)) right = p[i + j * column];
            else right = p[(i + 1) + j * column];

            if (j == 0) up = p[i + j * column];
            else up = p[i + (j - 1) * column];

            if(j == (row - 1)) down = p[i + j * column];
            else down = p[i + (j + 1) * column];

            currentU = u[i + j * column];
            currentV = v[i + j * column];

            u[i + j * column] = (zeroU) ? 0 : currentU - 0.5f * invCellSize * (right - left);
            v[i + j * column] = (zeroV) ? 0 : currentV - 0.5f * invCellSize * (down - up);
        }

    }

    __kernel void vorticity(
        const int row,
        const int column,
        const float height,
        const float invCellSize,
        __global float* vort,
        __global float* u,
        __global float* v,
        __global float* city
    ) {
        int index = get_global_id(0);
        int i = index % column;
        int j = index / column;

        if (city[i + j * column] < height) {
            float left, right, up, down;

            if (i == 0) left = 0;
            else if ( city[(i - 1) + j * column] >= height) left = u[i + j * column];
            else left = u[(i - 1) + j * column];

            if (i == (column - 1)) right = 0;
            else if (city[(i + 1) + j * column] >= height) right = u[i + j * column];
            else right = u[(i + 1) + j * column];

            if (j == 0) up = 0;
            else if (city[i + (j - 1) * column] >= height) up = v[i + j * column];
            else up = v[i + (j - 1) * column];

            if(j == (row - 1)) down = 0;
            else if(city[i + (j + 1) * column] >= height) down = v[i + j * column];
            else down = v[i + (j + 1) * column];

            vort[i + j * column] = 0.5 * invCellSize * ((right - left) - (down - up));
        }
    }

    __kernel void addConfinementForce(
        const int row,
        const int column,
        const float height,
        const float cellSize,
        const float coeff,
        __global float* u,
        __global float* v,
        __global float* vort,
        __global float* city
    ) {
        int index = get_global_id(0);
        int i = index % column;
        int j = index / column;

        if (city[i + j * column] < height) {
            float center, left, right, up, down, nX, nY, norm, currentU, currentV;

            center = vort[i + j * column];

            if (i == 0) left = 0;
            else if ( city[(i - 1) + j * column] >= height) left = center;
            else left = vort[(i - 1) + j * column];

            if (i == (column - 1)) right = 0;
            else if (city[(i + 1) + j * column] >= height) right = center;
            else right = vort[(i + 1) + j * column];

            if (j == 0) up = 0;
            else if (city[i + (j - 1) * column] >= height) up = center;
            else up = vort[i + (j - 1) * column];

            if(j == (row - 1)) down = 0;
            else if(city[i + (j + 1) * column] >= height) down = center;
            else down = vort[i + (j + 1) * column];

            nX = 0.5 * (fabs(right) - fabs(left)) / cellSize;
            nY = 0.5 * (fabs(down) - fabs(up)) / cellSize;
            norm = sqrt(nX * nX + nY * nY);
            nX = (norm != 0) ? nX / norm : 0;
            nY = (norm != 0) ? nY / norm : 0;

            currentU = u[i + j * column];
            currentV = v[i + j * column]; 

            u[i + j * column] = currentU + coeff * cellSize * nY * center;
            v[i + j * column] = currentV + coeff * cellSize * -nX * center;
        }
    }

    __kernel void addSource(
        const int row,
        const int column,
        const float height,
        __global float* x,
        __global float* x0,
        __global float* city
    ) {
        int index = get_global_id(0);
        int i = index % column;
        int j = index / column;

        if (x0[i + j * column] != 0 && city[i + j * column] < height) {
            x[i + j * column] = x0[i + j * column];
        }
    }
    """


context = None
queue = None
gaussSeidel = None
advect = None
divergence = None
subPressureGrad = None
vorticity = None
addConfinementForce = None
addSource = None


city = None
densitySource = None
uSource = None
vSource = None
zeros = None
density = None
u = None
v = None

timeStep = 0
cellSize = 0
height = 0

def simulate(
    _boundaryFeatures,
    _heightField,
    _densitySourceFeatures,
    _densityField,
    _velocitySourceFeatures,
    _uField,
    _vField,
    _sliceHeight,
    _simulationSteps,
    _frameInterval,
    _cellSize,
    _extent,
    _outputGDB,
    _outputMosaic
):
    global city, densitySource, uSource, vSource, zeros, density, u, v, timeStep, cellSize, height, context, queue, gaussSeidel, advect, divergence, subPressureGrad, vorticity, addConfinementForce, addSource
    #Discretize Simulation Domain
    arcpy.AddMessage("Initializing Simulation Domain")
    arcpy.env.extent = _extent
    cityRaster = Raster(arcpy.PolygonToRaster_conversion(_boundaryFeatures, _heightField, os.path.join(arcpy.env.scratchGDB, "CityRaster"), "MAXIMUM_AREA", _heightField, _cellSize))
    densitySourceRaster = Raster(arcpy.FeatureToRaster_conversion(_densitySourceFeatures, _densityField, os.path.join(arcpy.env.scratchGDB, "DensityRaster"), _cellSize))
    uSourceRaster = Raster(arcpy.FeatureToRaster_conversion(_velocitySourceFeatures, _uField, os.path.join(arcpy.env.scratchGDB, "URaster"), _cellSize))
    vSourceRaster = Raster(arcpy.FeatureToRaster_conversion(_velocitySourceFeatures, _vField, os.path.join(arcpy.env.scratchGDB, "VRaster"), _cellSize))

    city = arcpy.RasterToNumPyArray(cityRaster, nodata_to_value = -1).astype(numpy.float32)
    densitySource = arcpy.RasterToNumPyArray(densitySourceRaster, nodata_to_value = 0).astype(numpy.float32)
    uSource = arcpy.RasterToNumPyArray(uSourceRaster, nodata_to_value = 0).astype(numpy.float32)
    vSource = arcpy.RasterToNumPyArray(vSourceRaster, nodata_to_value = 0).astype(numpy.float32)

    zeros = numpy.zeros(city.shape, dtype = numpy.float32)
    density = numpy.zeros(city.shape, dtype = numpy.float32)
    u = numpy.zeros(city.shape, dtype = numpy.float32)
    v = numpy.zeros(city.shape, dtype = numpy.float32)

    timeStep = 1
    cellSize = float(_cellSize)
    height = float(_sliceHeight)

    arcpy.AddMessage("Grid Dimensions:" + str(city.shape))

    #Setup OpenCL
    arcpy.AddMessage("Initializing OpenCL")
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    program = cl.Program(context, programSource).build()
    gaussSeidel = program.gaussSeidel
    gaussSeidel.set_scalar_arg_dtypes([int, int, numpy.float32, numpy.float32, numpy.float32, None, None, None])
    advect = program.advect
    advect.set_scalar_arg_dtypes([int, int, numpy.float32, numpy.float32, None, None, None, None, None])
    divergence = program.divergence
    divergence.set_scalar_arg_dtypes([int, int, numpy.float32, numpy.float32, None, None, None, None])
    subPressureGrad = program.subPressureGrad
    subPressureGrad.set_scalar_arg_dtypes([int, int, numpy.float32, numpy.float32, None, None, None, None])
    vorticity = program.vorticity
    vorticity.set_scalar_arg_dtypes([int, int, numpy.float32, numpy.float32, None, None, None, None])
    addConfinementForce = program.addConfinementForce
    addConfinementForce.set_scalar_arg_dtypes([int, int, numpy.float32, numpy.float32, numpy.float32, None, None, None, None])
    addSource = program.addSource
    addSource.set_scalar_arg_dtypes([int, int, numpy.float32, None, None, None])

    for i in range(int(_simulationSteps)):
        arcpy.AddMessage("Starting Step " + str(i))
        advectVelocity()
        confineVorticity(0.5)
        addVelocitySource()
        diffuse(u, 0)
        diffuse(v, 0)
        project()

        addDensitySource()
        diffuse(density, 0)
        advectDensity()

        if i % int(_frameInterval) == 0:
            arcpy.AddMessage("Converting to Raster")
            output = arcpy.NumPyArrayToRaster(density, arcpy.Point(cityRaster.extent.XMin, cityRaster.extent.YMin), cityRaster.meanCellWidth)
            output.save(os.path.join(_outputGDB, "Frame" + str(i // int(_frameInterval))))

    arcpy.AddMessage("Simulation Finished")
    arcpy.AddMessage("Creating Mosaic")
    arcpy.CreateMosaicDataset_management(_outputGDB, _outputMosaic, arcpy.Describe(cityRaster).spatialReference)

def getSpeed(u, v):
    speed = numpy.empty(city.shape)
    for j in range(city.shape[0]):
        for i in range(city.shape[1]):
            speed[j, i] = math.sqrt(u[j, i]**2 + v[j, i]**2)
    return speed

def diffuse(x, coeff):
    a = (timeStep * coeff) / (cellSize * cellSize)
    sourceBuffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = x)
    cityBuffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = city)
    resultBuffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = zeros)
    gaussSeidel(queue, (city.shape[0] * city.shape[1], 1), None, city.shape[0], city.shape[1], height, a, 1 / (1 + 4 * a), resultBuffer, sourceBuffer, cityBuffer)
    
    cl.enqueue_copy(queue, x, resultBuffer)

def advectDensity():
    dt = timeStep / cellSize

    sourceBuffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = density)
    uBuffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = u)
    vBuffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v)
    cityBuffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = city)
    resultBuffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, city.nbytes)

    advect(queue, (city.shape[0] * city.shape[1], 1), None, city.shape[0], city.shape[1], height, dt, resultBuffer, sourceBuffer, uBuffer, vBuffer, cityBuffer)
    cl.enqueue_copy(queue, density, resultBuffer)

def addDensitySource():
    sourceBuffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = densitySource)
    resultBuffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = density)
    cityBuffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = city)

    addSource(queue, (city.shape[0] * city.shape[1], 1), None,  city.shape[0], city.shape[1], height, resultBuffer, sourceBuffer, cityBuffer)
    cl.enqueue_copy(queue, density, resultBuffer)

def advectVelocity():
    dt = timeStep / cellSize

    uBuffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = u)
    vBuffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v)
    cityBuffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = city)
    resultBuffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = zeros)

    advect(queue, (city.shape[0] * city.shape[1], 1), None, city.shape[0], city.shape[1], height, dt, resultBuffer, uBuffer, uBuffer, vBuffer, cityBuffer)
    cl.enqueue_copy(queue, u, resultBuffer)
    advect(queue, (city.shape[0] * city.shape[1], 1), None, city.shape[0], city.shape[1], height, dt, resultBuffer, vBuffer, uBuffer, vBuffer, cityBuffer)
    cl.enqueue_copy(queue, v, resultBuffer)

def addVelocitySource():
    cityBuffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = city)

    sourceBuffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = uSource)
    resultBuffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = u)

    addSource(queue, (city.shape[0] * city.shape[1], 1), None,  city.shape[0], city.shape[1], height, resultBuffer, sourceBuffer, cityBuffer)
    cl.enqueue_copy(queue, u, resultBuffer)

    sourceBuffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = vSource)
    resultBuffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v)

    addSource(queue, (city.shape[0] * city.shape[1], 1), None,  city.shape[0], city.shape[1], height, resultBuffer, sourceBuffer, cityBuffer)
    cl.enqueue_copy(queue, v, resultBuffer)


def project():
    uBuffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = u)
    vBuffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = v)
    cityBuffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = city)
    divergenceBuffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = zeros)
    pressureBuffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = zeros)

    divergence(queue, (city.shape[0] * city.shape[1], 1), None, city.shape[0], city.shape[1], height, cellSize, divergenceBuffer, uBuffer, vBuffer, cityBuffer)
    gaussSeidel(queue, (city.shape[0] * city.shape[1], 1), None, city.shape[0], city.shape[1], height, 1.0, 0.25, pressureBuffer, divergenceBuffer, cityBuffer)
    subPressureGrad(queue, (city.shape[0] * city.shape[1], 1), None, city.shape[0], city.shape[1], height, 1 / cellSize, uBuffer, vBuffer, pressureBuffer, cityBuffer)
    cl.enqueue_copy(queue, u, uBuffer)
    cl.enqueue_copy(queue, v, vBuffer)

def confineVorticity(coeff):
    uBuffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = u)
    vBuffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = v)
    cityBuffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = city)
    vorticityBuffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = zeros)

    vorticity(queue, (city.shape[0] * city.shape[1], 1), None, city.shape[0], city.shape[1], height, 1 / cellSize, vorticityBuffer, uBuffer, vBuffer, cityBuffer)
    addConfinementForce(queue, (city.shape[0] * city.shape[1], 1), None, city.shape[0], city.shape[1], height, cellSize, timeStep * coeff, uBuffer, vBuffer, vorticityBuffer, cityBuffer)

if __name__ == '__main__':
    argv = tuple(arcpy.GetParameterAsText(i)
        for i in range(arcpy.GetArgumentCount()))
    simulate(*argv)

