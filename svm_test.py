# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 07:52:50 2020

@author: MonOrdiPro
"""
import cv2
from sklearn.preprocessing import StandardScaler
from joblib import load



#%%

clf = load('svm.joblib') 

img = cv2.imread(r'C:\Users\MonOrdiPro\Desktop\Altran\Images\cropped_test\capture.JPG', 0)

img = cv2.resize(img, (60,60))
cv2.imshow('ii', img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()
scaler = StandardScaler()
scaler.fit(img)
img = scaler.transform(img)
img = img.flatten()


print(clf.predict(img.reshape(1, -1)))






