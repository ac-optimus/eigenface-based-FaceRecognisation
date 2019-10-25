#train and test the face recognisation system
from faceDetectionEigen import * 

#dataset path
PATH_TRAIN = "train/"
PATH_TEST = "test/"

#model hyperparameters
K = 20
THRESHOLD = 30000


print ("traning start!")
rec = FaceRecognonisation()
faces = rec.fit(PATH_TRAIN, K)
print ("training done!")
print("starting test-->")
y_hat = rec.test(PATH_TEST, K, THRESHOLD)
acc = accuracy_score(np.arange(40), y_hat)
print ("the accuracy:", acc)