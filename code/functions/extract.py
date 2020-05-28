import os
import re
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import firstorder, glcm, imageoperations, shape, glrlm, glszm, shape2D, ngtdm, gldm
import six

######
import auxFunc
import displayFunc

def radiomicsExtract(image, mask, 
                     file_name, #User Control
                     dimension, binCount, features="all"):
    def extractType(func, type_name):
        name = []
        values = []
        
        feat = func(image, mask, binCount=binCount, force2D=True)
        feat.enableAllFeatures()  
        feat.execute()
        for (key,val) in six.iteritems(feat.featureValues):
            name.append(key+f'_{type_name}')
            ################## ESSA LINHA AQUI TEM QUE CHECAR ... COLOQUEI O FLOAT PARA SALVAR EM PARQUET
            values.append(float(val)) ################################################
            ##################
        return pd.DataFrame([values], columns=name)
    
    if  dimension == 3:
        features_array = np.array(["FO", "S3D", "GLCM", "GLSZM", "GLRLM", "NGTDM", "GLDM"])
        features_func = np.array([firstorder.RadiomicsFirstOrder, shape.RadiomicsShape, glcm.RadiomicsGLCM,
                                  glszm.RadiomicsGLSZM, glrlm.RadiomicsGLRLM, ngtdm.RadiomicsNGTDM, 
                                  gldm.RadiomicsGLDM])
    elif dimension == 2:
        features_array = np.array(["FO", "S2D", "GLCM", "GLSZM", "GLRLM", "NGTDM", "GLDM"])
        features_func = np.array([firstorder.RadiomicsFirstOrder, shape2D.RadiomicsShape2D, glcm.RadiomicsGLCM,
                                  glszm.RadiomicsGLSZM, glrlm.RadiomicsGLRLM, ngtdm.RadiomicsNGTDM, 
                                  gldm.RadiomicsGLDM])
    else:
        return None

    if features != "all":
        if features is str:
            print("Type wrong. Returning None.")
            return None
        index = pd.Index(features_array).isin(features)
        features_array = features_array[index]
        features_func = features_func[index]

    list_feat = list(map(lambda i: extractType(features_func[i], features_array[i]), np.arange(len(features_array))))
    return pd.concat([pd.DataFrame([file_name], columns=["File Name"])] + list_feat, axis=1)

def extractFeatures(image_object, mask_object, #User Control
                    file_name,
                    binCount= 32, features = "all", reduce = True):

    shape_mask = mask_object.GetSize()
    dimension = sum([True if i != 1 else False for i in shape_mask])

    if reduce:
        #Só precisa passar a imagem para array se for reduzir
        image_np = sitk.GetArrayFromImage(image_object)
        mask_np = sitk.GetArrayFromImage(mask_object)
        #Retorna
        image_object, mask_object = auxFunc.reduceSize(image_np, mask_np)

    return radiomicsExtract(image_object, mask_object, file_name, dimension, binCount = binCount, features=features)