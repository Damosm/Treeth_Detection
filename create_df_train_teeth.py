# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 07:52:50 2020

@author: MonOrdiPro
"""
import cv2
import pandas as pd


from sklearn.preprocessing import StandardScaler
import glob
import numpy as np





path_M_HG = glob.glob(r"C:\Users\utilisateur\OneDrive\Documents\Deeplearning\Altran\Treeth_Detection\images_svm\mol_HG\*jpg")
path_PM_HG = glob.glob(r"C:\Users\utilisateur\OneDrive\Documents\Deeplearning\Altran\Treeth_Detection\images_svm\pm_HG\*jpg")
path_C_HG = glob.glob(r"C:\Users\utilisateur\OneDrive\Documents\Deeplearning\Altran\Treeth_Detection\images_svm\can_HG\*jpg")
path_I_HG = glob.glob(r"C:\Users\utilisateur\OneDrive\Documents\Deeplearning\Altran\Treeth_Detection\images_svm\inci_HG\*jpg")
path_I_HD = glob.glob(r"C:\Users\utilisateur\OneDrive\Documents\Deeplearning\Altran\Treeth_Detection\images_svm\inci_HD\*jpg")
path_C_HD = glob.glob(r"C:\Users\utilisateur\OneDrive\Documents\Deeplearning\Altran\Treeth_Detection\images_svm\can_HD\*jpg")
path_PM_HD = glob.glob(r"C:\Users\utilisateur\OneDrive\Documents\Deeplearning\Altran\Treeth_Detection\images_svm\pm_HD\*jpg")
path_M_HD = glob.glob(r"C:\Users\utilisateur\OneDrive\Documents\Deeplearning\Altran\Treeth_Detection\images_svm\mol_HD\*jpg")

path_M_BG = glob.glob(r"C:\Users\utilisateur\OneDrive\Documents\Deeplearning\Altran\Treeth_Detection\images_svm\mol_BG\*jpg")
path_PM_BG = glob.glob(r"C:\Users\utilisateur\OneDrive\Documents\Deeplearning\Altran\Treeth_Detection\images_svms\pm_BG\*jpg")
path_C_BG = glob.glob(r"C:\Users\utilisateur\OneDrive\Documents\Deeplearning\Altran\Treeth_Detection\images_svm\can_BG\*jpg")
path_I_BG = glob.glob(r"C:\Users\utilisateur\OneDrive\Documents\Deeplearning\Altran\Treeth_Detection\images_svm\inci_BG\*jpg")
path_I_BD = glob.glob(r"C:\Users\utilisateur\OneDrive\Documents\Deeplearning\Altran\Treeth_Detection\images_svm\inci_BD\*jpg")
path_C_BD = glob.glob(r"C:\Users\utilisateur\OneDrive\Documents\Deeplearning\Altran\Treeth_Detection\images_svm\can_BD\*jpg")
path_PM_BD = glob.glob(r"C:\Users\utilisateur\OneDrive\Documents\Deeplearning\Altran\Treeth_Detection\images_svm\pm_BD\*jpg")
path_M_BD = glob.glob(r"C:\Users\utilisateur\OneDrive\Documents\Deeplearning\Altran\Treeth_Detection\images_svm\mol_BD\*jpg")

h = 60
l = 60

def load_img(path) :
    df = pd.DataFrame(columns=[n for n in range (0,h*l)])
    i=0
    for img_path in path :
        print('in '+img_path)
        img = cv2.imread(img_path,0)
        img = cv2.resize(img, (l,h))
        
        scaler = StandardScaler()
        scaler.fit(img)
        img = scaler.transform(img)
        img = img.flatten()
        df_temp = pd.Series(img)
        df.loc[i] = df_temp
        i+=1
    return df

df = load_img(path_M_HG)
df['type'] = 0
df = df.append(load_img(path_PM_HG))
df = df.fillna(1)  
df = df.append(load_img(path_C_HG))
df = df.fillna(2)
df = df.append(load_img(path_I_HG))
df = df.fillna(3)
df = df.append(load_img(path_M_HD))
df = df.fillna(4)
df = df.append(load_img(path_PM_HD))
df = df.fillna(5)  
df = df.append(load_img(path_C_HD))
df = df.fillna(6)
df = df.append(load_img(path_I_HD))
df = df.fillna(7)

df = df.append(load_img(path_M_BG))
df = df.fillna(8) 
df = df.append(load_img(path_PM_BG))
df = df.fillna(9)  
df = df.append(load_img(path_C_BG))
df = df.fillna(10)
df = df.append(load_img(path_I_BG))
df = df.fillna(11)
df = df.append(load_img(path_M_BD))
df = df.fillna(12)
df = df.append(load_img(path_PM_BD))
df = df.fillna(13)  
df = df.append(load_img(path_C_BD))
df = df.fillna(14)
df = df.append(load_img(path_I_BD))
df = df.fillna(15)


df = df.reset_index(drop=True)



df.to_csv('tooth_train_df.csv', index = False)