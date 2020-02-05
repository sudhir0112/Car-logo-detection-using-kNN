from sklearn.neighbors import KNeighborsClassifier
from skimage import feature
from skimage import exposure
import cv2
import os

inputPath='../input/'
datasetPath='../dataset/'
outputPath='../output/'
testImageName='test.jpg'

imageSize = (200,100)

print('[INFO] Extracting features...')
data=[]
labels=[]

for logos in os.listdir(datasetPath):
    if os.path.isdir(datasetPath+logos):
        label = logos 
        for imageName in os.listdir(datasetPath+logos):

            #Load image from the path
            imagePath = datasetPath+logos+ '/' + imageName 
            image=cv2.imread(imagePath)
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(gray,100,255)

            #im2, contours, hierarchy=cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy=cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            maxCnt=max(contours,key=cv2.contourArea)

            # extract the logo of the car and resize it to a canonical width
            # and height
            x,y,w,h=cv2.boundingRect(maxCnt)
            logo = gray[y:y + h, x:x + w]
            logo = cv2.resize(logo, imageSize)
            H = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
                cells_per_block=(2, 2), transform_sqrt=True)

            # update the data and labels
            data.append(H)
            labels.append(label)


# "train" the nearest neighbors classifier
print ("[INFO] training classifier...")
model = KNeighborsClassifier(n_neighbors=1)
model.fit(data, labels)
print ("[INFO] evaluating...")

'''Now loading the test image and predicting the results'''
image=cv2.imread(inputPath+testImageName)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.resize(gray, imageSize) 

(H, hogImage)=feature.hog(gray,orientations=9,pixels_per_cell=(10,10),cells_per_block=(2,2),
              transform_sqrt=True,visualize=True)
pred = model.predict(H.reshape(1,-1))[0]
print ('Pridicted Result is: '+str(pred))

# visualize the HOG image
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
cv2.imshow("HOG Image", hogImage)

# draw the prediction on the test image and display it
cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 2)
cv2.imshow("Test Image", image)
cv2.waitKey(0)



