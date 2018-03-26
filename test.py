from __future__ import print_function
from tifffile import TiffFile

import hashlib
import io
import zipfile

import sys
from abc import ABCMeta, abstractmethod
from multiprocessing import Process, Manager
from pprint import pprint
import numpy as np
import random
import re
import sys
from operator import add
import os
from pyspark import SparkContext, SparkConf


def getOrthoTif(zfBytes):
 #given a zipfile as bytes (i.e. from reading from a binary file),
 # return a np array of rgbx values for each pixel
 bytesio = io.BytesIO(zfBytes[1])
 zfiles = zipfile.ZipFile(bytesio, "r")
 #find tif:
 for fn in zfiles.namelist():
  if fn[-4:] == '.tif':#found it, turn into array:
   tif = TiffFile(io.BytesIO(zfiles.open(fn).read()))
   return (zfBytes[0].split('/')[len(zfBytes[0].split('/'))-1],tif.asarray())

def f(x):
    avg = sum(x[0:3])/3.0
    infrared = x[3]/100.0
    return int(avg*infrared)

def changeIntensity(subarray):

    rows = subarray.shape[0];
    cols = subarray.shape[1];

    transformedArray = np.ones((rows,cols))

    #f = np.vectorize(f)

    transformedArray = np.apply_along_axis(f,2,subarray)
    '''
    for i in range(subarray.shape[0]):
        for j in range(subarray.shape[1]):
            avgval = sum(subarray[i,j,0:3])/3
            infrared = subarray[i,j,3]/100
            transformedArray[i,j] = int(avgval*infrared)
    '''
    return transformedArray




def getAvg(array):  
    return sum(map(sum,array))

def reduceResol(array,factor):
    
    rows = array.shape[0]
    cols = array.shape[1]

    resol = factor*factor
    reducearray = np.zeros((rows//factor,cols//factor))

    for i in range(0,rows,factor):
        for j in range(0,cols,factor):
            reducearray[i//factor,j//factor] = getAvg(array[i:i+factor,j:j+factor])/resol
            
    return reducearray

def transformdiscrete(array):

    rows = array.shape[0]
    cols = array.shape[1]

    array = np.reshape(array,(1,rows*cols))

    array[np.logical_and((array>=-1),(array<=1)) ] = 0

    array[array < -1] = -1
    array[array > 1 ] = 1

    return array 

def minhash(array,factor):
    features = array.shape[0]
    #size_38 = 92
    #size_39 = 36
    minhash_feature = np.zeros(128)
    
    chuncks = np.array_split(array,128)
    for i in range(0,128):
        hashval = hashlib.md5(chuncks[i])
        hashbit = bin(int('0x'+hashval.hexdigest(),16))[-1]
        minhash_feature[i] = hashbit             
    return minhash_feature



def lshband(rddrow):
    filename = rddrow[0]
    feature  = rddrow[1]
    
    bands = 8
    lencols = 128//bands
    bucket = 1000
    
    result = []
    chuncks = np.array_split(feature,bands)    
    for i in range(0,bands):
        hashval = hashlib.md5(chuncks[i])
        result.append((filename,int('0x'+hashval.hexdigest(),16)%bucket + i*bucket))
    return result


def findsimilarImages(imagelist):
    images = ['3677454_2025195.zip-20','3677454_2025195.zip-1','3677454_2025195.zip-18','3677454_2025195.zip-19']
    result = []
    for image in images:
        if image in imagelist:
            result.append((image,imagelist))
    return result


def findsimilarity(PCAfeatures,imageset):
    
    result = []
    
    featdict = PCAfeatures
    
    for row in imageset:
        temp = []
        for image in row[1]:
            temp.append((image,findeuclidean(featdict[row[0]],featdict[image])))
        temp = sorted(temp, key=lambda x:x[1])# sorting based on distance
        result.append((row[0],temp))
    return result

def findeuclidean(row1,row2):
    return np.sqrt(np.sum((row1-row2)**2))    


def PCAonpartition(list_of_images):
    res = []
    features = []
    
    images = []
    
    for image in list_of_images:
        features.append(image[1])
        images.append(image[0])
    
    featuresmean = np.mean(features,axis=0)
    featurestd  = np.std(features,axis =0)
    
    features = (features - featuresmean)/featurestd
    
    U , D, Vt = linalg.svd(features)
    
    res = []
    i=0
    for image in images:
        res.append((image,U[i,0:10]))
        i = i+1
    
    return iter(res)


# In[2]:

def findEigenVectors(list_of_images):
    res = []
    features = []
    
    images = []
    
    for image in list_of_images:
        features.append(image[1])
        images.append(image[0])
    
    featuresmean = np.mean(features,axis=0)
    featurestd  = np.std(features,axis =0)
    
    features = (features - featuresmean)/featurestd
   
    features = np.nan_to_num(features)
 
    U , D, Vt = linalg.svd(features)
   
    #features = np.nan_to_num(features)
 
    return np.transpose(Vt[0:10,:])


from scipy import linalg

def concatenatefeatures(array):

    rowdiff = np.diff(array,axis=0)

    coldiff = np.diff(array,axis=1)

    rows = rowdiff.shape[0]
    cols = rowdiff.shape[1]    
    
    rowdiff = rowdiff.flatten()

    coldiff = coldiff.flatten()

    result = np.concatenate((rowdiff,coldiff))

    result[np.logical_and((result>=-1),(result<=1)) ] = 0

    result[result < -1] = -1
    result[result > 1 ] = 1


    return result

def RDDfunc(tiffRDDIntensityRDD,factor):

    tiffredresolRDD = tiffRDDIntensityRDD.map(lambda x:(x[0], reduceResol(x[1],factor))) 
 
    tiffRdiff3 = tiffredresolRDD.map(lambda x: (x[0],concatenatefeatures(x[1])))

    tiffdifffilter = tiffRdiff3.filter(lambda x: x[0] == "3677454_2025195.zip-1" or x[0] == "3677454_2025195.zip-18"  ).map(lambda x:(x[0],x[1]))   

    if factor == 10:    
        print(tiffdifffilter.collect())
        print("\n\n\n")

    minhashRDD = tiffRdiff3.map(lambda x: (x[0],minhash(x[1],factor)))

    lshbandRDD = minhashRDD.flatMap(lambda x: lshband(x))

    #print(lshbandRDD.take(1))

    bucketRDD = lshbandRDD.map(lambda x: (x[1],x[0])).groupByKey().mapValues(list) #.filter(lambda x: '3677454_2025195.zip-14' in x[1])

    similarimageRDD = bucketRDD.flatMap(lambda x: findsimilarImages(x[1])).filter(lambda x:x).reduceByKey(lambda x,y:list(set(x+y)))

    sampledrows = tiffRdiff3.takeSample(False,20)

    eigenvalues = sc.broadcast(findEigenVectors(sampledrows))

    similarimages = similarimageRDD.filter(lambda x: x[0] == '3677454_2025195.zip-1' or x[0] =='3677454_2025195.zip-18').collect()    
   
    if factor ==10:
        print(similarimages)
        print("\n\n\n")
 
    PCAfeatures = tiffRdiff3.filter(lambda x: x[0] in similarimages[0][1] or x[0] in similarimages[1][1]).map(lambda x: (x[0],np.matmul(x[1],np.array(eigenvalues.value))))

    filterPCAfeatures = PCAfeatures.collectAsMap()

    res = findsimilarity(filterPCAfeatures,similarimages)

    print(res)



if __name__ == "__main__":

    file_path = "hdfs:/data/large_sample/"

    conf = SparkConf().setAppName("SatelliteImageAnalysis")
    sc = SparkContext(conf=conf)

    ImageRDD = sc.binaryFiles(file_path)
    FileRDD  = ImageRDD.map(lambda x: x[0])

    FileNameImageRDD = FileRDD.map(lambda FileRDD: FileRDD.split('/')[len(FileRDD.split('/'))-1])

    tiffRDD = ImageRDD.map(getOrthoTif)


    tiffRDDreplica = tiffRDD.flatMap(lambda x: [(x[0]+"-"+str(i),x[1][(i//5)*500:(i//5)*500+500,(i%5)*500:(i%5)*500+500,:]) for i in range(0,25,1)])


    tiffIntensityfilter = tiffRDDreplica.filter(lambda x: x[0] == "3677454_2025195.zip-0" or x[0] == "3677454_2025195.zip-1" or x[0] == "3677454_2025195.zip-18" or x[0] == "3677454_2025195.zip-19" ) # -- 1.e

    tiffIntensityfilter = tiffIntensityfilter.map(lambda x:(x[0],x[1][0,0,:]))

    print(tiffIntensityfilter.collect())

    tiffRDDIntensityRDD = tiffRDDreplica.map(lambda x:(x[0], changeIntensity(x[1])))

    tiffRDDIntensityRDD.persist()

    RDDfunc(tiffRDDIntensityRDD,10)

    print("\n\n\n\n")

    RDDfunc(tiffRDDIntensityRDD,5)
    
    sc.stop()