import sys
import math
#importing Image library to read and display the image. 
from PIL import Image
#Library to show the images  
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
#Library for matrix calculations
import numpy as np
#Library to calculate eigen values and eigen vectors
from numpy import linalg as LA

#==================================Implementing EigenFaces Training Functions===================================================#
#For each training image, the rows are stacked together to form a column vector Ri of dimension width*heght
#here all the images are stacked in the list imagesN2vector
def convertToN2vector(width, height):
    imagesN2vector = []
    for images in range(len(Imageobjects)):
        k = 0
        individualImages = np.zeros((width*height,1),dtype = np.int16)
        for i in range(height):
            for j in range(width):
                #storing the pixel value in the form of increasing rows with single column 
                individualImages[k,0] = Imageobjects[images].getpixel((j,i))
                k = k + 1
        #appending single image to stack all the images
        imagesN2vector.append(individualImages)
    return imagesN2vector


#The mean face m(meanFace) is computed by taking the average of the M training face images
#providing the input as obtained in the above method
def averageFaceVector(imagesN2vector):
    meanFace = np.zeros((width*height,1),dtype = np.int16)
    M = len(imagesN2vector)
    for i in range(len(imagesN2vector[0])):
        sum = 0
        #taking the sum of all pixel in one row and dividing by the number of images
        for images in range(len(imagesN2vector)):
            sum = sum + imagesN2vector[images][i][0]
        sum = sum/M
        meanFace[i][0] = sum
    return meanFace

#subtracting the mean face m from each training face
#providing the input as obtained in the above method to subtract
def subtractMeanFace(imagesN2vector, meanFace):
    subtractmeanface = []
    for images in range(len(Imageobjects)):
        #using the subtract function of the numpy library
        individualImages = np.subtract(imagesN2vector[images],meanFace)
        #appending the individual subtracted image stacked in one list
        subtractmeanface.append(individualImages)
    return subtractmeanface

#All training faces into a single matrix A of dimension [width*height, no_of_test_images]
#providing the input as obtained in the above method
def matrixA(R):
    A = np.zeros((width*height,len(R)),dtype = np.int16)
    for i in range(width*height):
        for j in range(len(R)):
            A[i,j] = R[j][i][0]
    return A

#since calculation of eigen values and eigen vectors from co-variance matrix will require large computational effort as matrix will be large
#Implementing the alternate method to calculate the eigenvalues
#AT is the transpose of matrix A obtained above
#taking the dot product of AT and A to get the matrix L
# calculating the eigenvalue and eigen vector
# w is the eigenvalue, V is the eigenvector
def alternateToCovariance(A):
    AT = np.transpose(A)
    L = np.dot(AT, A)
    w, V = LA.eig(L)
    return V

#Eigen vectors of C can be found by U = AV
# U is the eigenspace, face spave or eigenfaces
def covariance(A, V):
    U = np.dot(A, V)
    return U

#printing all the eigen faces based on the dataset of the images provided.
# For 8 trained image we will get 8 eigenFaces
def printEigenFaces(U):
    for i in range (len(U[0])):
        plt.title('Eigen face'+ str(i))
        plt.imshow((U[:,i].reshape(231,195)),cmap='gray')
        plt.show()

#Each training face can then be projected on the face space
#UT is the transpose of U as obtained above
#projectedfacespace = (UT)(Ri)
def projectedFaceSpace(U, R):
    UT = np.transpose(U)
    rows, column = UT.shape
    projectedfacespace = []
    for images in range(len(Imageobjects)):
        projectedfacespace.append(np.dot(UT, R[images])) 
    return projectedfacespace

#printing all the PCA coefficients from the projected face space obtained in above method
#8 training image will have 8 set of PCA coeffients
def printPCACoefficients(projectedfacespace):
    for i in range(len(projectedfacespace)):
        print "PCA Coefficient for training Image", i
        print projectedfacespace[i]


#==================================Implementing EigenFaces Recognition Functions===================================================#
#reading the test image of which the face needs to be recognized
#the function will take single test image at a time
def getTestImage(width, height):
    Testimage = np.zeros((width*height,1),dtype = np.int16)
    k = 0
    for i in range(height):
        for j in range(width):
            Testimage[k,0] = test_image_object.getpixel((j,i))
            k = k + 1
    return Testimage

#subtracting the mean face m from each test face
#Mean face m was obtained in the above training methods
#providing the input as obtained in the above method to subtract
def subtractTestFace(testimage, meanFace):
    subtracttestface = np.subtract(testimage,meanFace)
    return subtracttestface

#computing its projection onto the face space
#UT is the transpose of U as obtained above in the training method
#projectionface = (UT)(I)
#I is the subtracted image as obtained above after subtracting
def projectiononFace(U, I):
    projectionface = np.dot(np.transpose(U), I)
    return projectionface

#Reconstruct input face image from the eigenfaces
#reconstructedimage = (U)(projectionface)
#where U is the eigenface and the projectionFace of the test image is obtained above
def reconstruct(U, projectionface):
    reconstructedimage = np.dot(U, projectionface)
    return reconstructedimage

#Computing the distance between the input face image and the reconstruction of the image
#Subtracting the value pixel by pixel and then passing the complete vector to get the euclidean distance
def findEuclideanDistance(reconstructedimage, I):
    subtractedform = np.subtract(reconstructedimage, I)
    subtractedform = LA.norm(subtractedform)
    return subtractedform

#Compute distance between input face image and training images in the face space
#projectionface is the projected test face
#projectedfacespace[i] is the individual traing images
#Test image is subtracted from all the training images then the image with the minimum distance is taken with the image in the dataset
def computeDistance(projectionface, projectedfacespace):
    Di = []
    for i in range(len(projectedfacespace)):
        di = LA.norm(np.subtract(projectionface, projectedfacespace[i]))
        Di.append(di)
    return Di

def displayFinalResult(testpath,imagePath):
    plt.title('Input Test Image')
    plt.imshow(mpimg.imread(testpath),cmap='gray')
    plt.figure()
    plt.imshow(mpimg.imread(imagePath),cmap='gray')
    plt.title('Resulting image')
    plt.show()

#==================================================================================================================================
#EigenFaces Training
#Training Dataset
#TrainingImages list contains the set images as training dataset
#change the path in the 'imagePath' variable where all the images are stored
#change to the path of images stored on the machine
storedimagepath = "Dataset\\"
TrainingImages = ['subject01.normal.jpg', 'subject02.normal.jpg', 'subject03.normal.jpg', 'subject07.normal.jpg', 'subject10.normal.jpg', 'subject11.normal.jpg', 'subject14.normal.jpg', 'subject15.normal.jpg']
Imageobjects = []
for i in range(len(TrainingImages)):
    imagePath = storedimagepath + TrainingImages[i]
    image_object = Image.open(imagePath)
    Imageobjects.append(image_object)
#All the images are of same size
#Calculating the width and height
width, height = Imageobjects[0].size

imagesN2vector = convertToN2vector(width, height)

meanFace = averageFaceVector(imagesN2vector)
plt.title("Mean Face m")
plt.imshow(meanFace.reshape(231,195), cmap = "gray")
plt.show()

R = subtractMeanFace(imagesN2vector, meanFace)

A = matrixA(R)

V = alternateToCovariance(A)

U = covariance(A, V)

printEigenFaces(U)

projectedfacespace = projectedFaceSpace(U, R)

printPCACoefficients(projectedfacespace)

#=========================================================================================================================================
#Testing
#EigenFaces Recognition
print "Training on Images is completed..."
testImage = raw_input("Provide the test image to be detected: ")
testimagePath = "Dataset\\" + testImage
test_image_object = Image.open(testimagePath)
width, height = test_image_object.size

testimage = getTestImage(width, height)

I = subtractTestFace(testimage, meanFace)
plt.title("The image after subtracting from the mean face")
plt.imshow(I.reshape(231,195), cmap = "gray")
plt.show()

projectionface = projectiononFace(U, I)
print "For each test image: the image after subtracting the mean face, its PCA coefficients"
print projectionface

reconstructedimage = reconstruct(U, projectionface)
plt.title("The reconstructed face Image")
plt.imshow(reconstructedimage.reshape(231,195), cmap = "gray")
plt.show()

subtractedform = findEuclideanDistance(reconstructedimage, I)
print "Distance D0 is",subtractedform
#Choosing the threshold
#T0 is used to identify whether the image is face or non-face
T0 = 7000000000000
#T1 is used to identify whether the face is present in the dataset or not
T1 = 89000000
if (subtractedform > T0):
    print ""
    print "Classification: non-face"

else:
    Di = computeDistance(projectionface, projectedfacespace)
    print "Distance D of Test Image to Train Image is",min(Di)
    if (min(Di) > T1):
        print ""
        print "Classification: unknown face"
    else:
        print ""
        print "Classification: identify of face which is similar to", TrainingImages[Di.index(min(Di))]
        displayFinalResult(testimagePath,storedimagepath + TrainingImages[Di.index(min(Di))])
