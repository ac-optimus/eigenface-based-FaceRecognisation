import numpy as np
import os
import matplotlib.pylab as plt
from numpy import linalg 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize


class UTILS:
    """utility class for helper methods"""
    def __init__(self,):
        pass
    def getImg(self,loc):
        """load image from , files--> image-matrix """
        list_dir = os.listdir(loc)
        lst_images = []
        for person_i in list_dir:
            images = os.listdir(loc+person_i)
            for img_i in images:
                img_i = plt.imread(loc+person_i+"/"+img_i)
                lst_images.append(img_i.reshape(1,img_i.size))
        A = np.concatenate(lst_images, axis=0)
        A = A.transpose()
        return A #number of dimmesnion X number of images

    def m_shift(self,A):
        """mean sift:
            input : image matrix
            outut :mean image and mean shifted image
        """
        mean = np.mean(A, axis=1).reshape(A.shape[0],1)
        A = A - mean
        return mean, A 

    def cov(self, A):
        """compute covarience matrix"""
        return np.matmul( A.transpose(), A)

    def EigenFaces(self, A, k):
        """compute eigenvales and eigenvectors (A.transpose)A matrix"""
        mat = self.cov( A)  #MXM covarience matrix
        w, v = linalg.eig(mat) #v[:i] is ith eigenvector
        u = np.matmul(A, v)#rows are eigenvectors
        ui = normalize(u, axis=1, norm='l2')#l2 normalization
        indexK = np.argsort(w)[::-1][:k]#top k indexes

        ui = ui.transpose()[indexK].transpose()
        eigenFaces = np.matmul(ui.transpose(), A)  #corresponding eigenvectors
        eigen_vec = ui
        return eigenFaces, eigen_vec

class FaceRecognonisation(UTILS):
    def __init__(self):
        UTILS.__init__(self)

    def fit(self, trainLoc, K):
        """train method for face recogonisation"""
        A = self.getImg(trainLoc)
        mean_face, A = self.m_shift(A)
        faces, eigen_vec = self.EigenFaces(A, K)
        #save the eigenfaces, eigen vec and mean face for testing
        self.faces = faces
        self.eigen_vec = eigen_vec
        self.meanFace = mean_face
        return 0#ALL THE EIGENFACES

    def test(self, testLOC, K, threshold):
        """test method for face recognisation"""
        A = self.getImg(testLOC)
        A = A - self.meanFace  #using the same mean face
        OMEGA = self.faces.transpose() 
        OMEGA_test = np.matmul(self.eigen_vec.transpose(), A).transpose()
        lables = []

        for image_i in OMEGA_test:
            l =OMEGA - image_i.reshape(1,image_i.shape[0])
            p = linalg.norm(l, axis=1)
            index, val = np.argmin(p), np.min(p)
            if val <threshold:  #see if threshold is crossed
                lables.append((index//9))
            else:
                lables.append(-1)
        return lables


