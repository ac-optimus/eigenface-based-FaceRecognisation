#plot eigenfaces
from faceDetectionEigen import * 
import matplotlib.pylab as plt

#dataset path
PATH_TRAIN = "train/"
PATH_TEST = "test/"

#model hyperparameters
K = 20
IMAGE_SIZE = (112,92)

print ("traning start!")
rec = FaceRecognonisation()
faces = rec.fit(PATH_TRAIN, K)
p = rec.eigen_vec
k = p[:,1].reshape(IMAGE_SIZE)  #change 1 to some other value to see other eigenface
print ("see the eigenface")
plt.imshow(k, cmap='Greys')
plt.show()