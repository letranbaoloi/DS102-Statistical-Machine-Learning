import cv2
from joblib import dump, load #Use save model
import matplotlib.pyplot as plt
import pickle
from skimage import feature

with open('model_lr.pkl', 'rb') as file:
    model = pickle.load(file)

def TinhHog(pathfilename):
  img = cv2.imread(pathfilename)
  print(img.shape)
  img = cv2.resize(img, (100, 100))
  print(img.shape)


x_test = TinhHog('IM-0001-0001.jpeg')
