import sys
import os 
import arcpy
from arcpy import env
from arcpy.sa import *
import re

getTime = """import re
def getTime(name):
    return int(re.sub("Frame", "", name))
"""

def mosaicCreator(
    _sourceGDB,
    _mosaic,
    _simulationSteps,
    _frameInterval
):
    #arcpy.CreateMosaicDataset_management(_sourceGDB, _mosaic, "PROJCS['WGS_1984_Web_Mercator_Auxiliary_Sphere',GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Mercator_Auxiliary_Sphere'],PARAMETER['False_Easting',0.0],PARAMETER['False_Northing',0.0],PARAMETER['Central_Meridian',0.0],PARAMETER['Standard_Parallel_1',0.0],PARAMETER['Auxiliary_Sphere_Type',0.0],UNIT['Meter',1.0]]", None, None, "NONE", None)

    mosaicPath = os.path.join(_sourceGDB, _mosaic)
    rasters = ''
    for i in range(int(_simulationSteps) // int(_frameInterval)):
        filePath = os.path.join(_sourceGDB, "Frame"+ str(i))
        rasters = rasters + ';' + filePath

    arcpy.AddMessage("Adding Rasters to Mosaic")
    arcpy.AddRastersToMosaicDataset_management(mosaicPath, "Raster Dataset", rasters, calculate_statistics = "CALCULATE_STATISTICS", estimate_statistics = "ESTIMATE_STATISTICS")
    #arcpy.management.AddRastersToMosaicDataset(os.path.join(_mosaicPath,'Mosaic'), "Raster Dataset", rasters, "UPDATE_CELL_SIZES", "UPDATE_BOUNDARY", "NO_OVERVIEWS", None, 0, 1500, None, None, "SUBFOLDERS", "ALLOW_DUPLICATES", "NO_PYRAMIDS", "CALCULATE_STATISTICS", "NO_THUMBNAILS", None, "NO_FORCE_SPATIAL_REFERENCE", "ESTIMATE_STATISTICS", None)

    arcpy.AddMessage("Adding Time Stamps")
    arcpy.AddField_management(mosaicPath, "TimeStamp", "LONG")
    arcpy.CalculateField_management(mosaicPath, "TimeStamp", "getTime(!Name!)", code_block = getTime)

if __name__ == '__main__':
    argv = tuple(arcpy.GetParameterAsText(i)
        for i in range(arcpy.GetArgumentCount()))
    mosaicCreator(*argv)