import sys
import os
import cv2
import numpy as np
from sklearn.svm import SVC
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import time
import pickle as cPickle

SHAPE = (50, 50)

def read_files(directory):
   print "Reading files..."
   images_list = list()
   files = os.listdir(directory)
   for filename in files:
      images_list.append(filename)
   return images_list 
   

def extract_feature(image_file):
   img = cv2.imread(image_file)
   img = cv2.resize(img, SHAPE, interpolation = cv2.INTER_CUBIC)
   sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
   abs_sobel64f = np.absolute(sobelx64f)
   img= np.uint8(abs_sobel64f)
   img = img.flatten()
   return img

if __name__ == "__main__":
   if len(sys.argv) < 2:
      print  "Usage: python extract_features.py [image_folder]"
      exit()

   # Directory containing subfolders with images in them.
   directory = sys.argv[1]

   # generating two numpy arrays for features and labels
   images= read_files(directory)
   


#load saved model
with open('svm_model.pkl' , 'rb') as f :
    lr = cPickle.load(f)
    

for i in images :
   
   path = './testes/%s'%i
   img = extract_feature(path) 
   img = img.reshape(1, -1)
   print "make a prediction"
   w=lr.predict(img)
   print w
   print i,'',w 

