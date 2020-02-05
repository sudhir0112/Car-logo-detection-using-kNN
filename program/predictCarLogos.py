from sklearn.neighbors import KNeighborsClassifier
import joblib                                    #For storing ML model 
from skimage import feature
from skimage import exposure
import cv2
import os
import time 
from progress.bar import IncrementalBar

inputPath='../input/'
datasetPath='../dataset/'
outputPath='../output/'
modelPath = '../model/'
testImageName='test.jpg'
modelName = 'carLogoModel.pkl'

imageSize = (200,100)

def LoadTrainModel():
    print ('[INFO] Loading ML Model')
    initialTime = time.time()
    model=joblib.load(modelPath+modelName)
    print('[INFO] ML Model Loaded ,Time Taken: ' + str(round((time.time() - initialTime),3)) + ' sec')
    print('[INFO] ML Model Name: ' + str(modelName))
    modelSize = round(os.stat(modelPath+modelName).st_size / (1024 * 1024),3)
    print('[INFO] ML Model size: ' + str(modelSize) + ' MB')
    return model

model = LoadTrainModel()
'''Now loading the test image and predicting the results'''
image=cv2.imread(inputPath+testImageName)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.resize(gray, imageSize) 

(H, hogImage)=feature.hog(gray,orientations=9,pixels_per_cell=(10,10),cells_per_block=(2,2),
              transform_sqrt=True,visualize=True)
pred = model.predict(H.reshape(1,-1))[0]
print ('[INFO] Pridicted Result is: '+str(pred))

# visualize the HOG image
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
cv2.imshow("HOG Image", hogImage)

# draw the prediction on the test image and display it
cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 2)
cv2.imshow("Test Image", image)
cv2.waitKey(0)