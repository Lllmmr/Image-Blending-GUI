import cv2
import numpy as np
import PasteOnePicToAnother

def LaplacePyramid(src, mask, back, resultaddr, posX=0 , posY=0):
    if mask is None:
        A_raw = src
    else:
        A_raw = PasteOnePicToAnother.Pyramid_CreateNewSrc(src, mask, back, posX, posY)
    #cv2.imshow("lalala",A)
    B_raw = back
    #cv2.imshow("lala",B)

    #进行裁剪
    H = A_raw.shape[0]
    W = A_raw.shape[1]
    A = A_raw[0:(int)(H/16)*16, 0:(int)(W/16)*16]
    H = B_raw.shape[0]
    W = B_raw.shape[1]
    B = B_raw[0:(int)(H / 16) * 16, 0:(int)(W / 16) * 16]
    #print(A.shape, B.shape)
    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in np.arange(4):
        G = cv2.pyrDown(G)
        gpA.append(G)

    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in np.arange(4):
        G = cv2.pyrDown(G)
        gpB.append(G)

    # generate Laplacian Pyramid for A
    lpA = [gpA[3]]
    for i in np.arange(3, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)

    # generate Laplacian Pyramid for B
    lpB = [gpB[3]]
    for i in np.arange(3, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)

    # Now add left and right halves of images in each level
    # numpy.hstack(tup)
    # Take a sequence of arrays and stack them horizontally
    # to make a single array.
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        #ls = np.hstack((lb[:, 0:posY],la[:, posY:] ))  # 将两个图像的矩阵拼接到一起
        ls = cv2.addWeighted(la,0.4, lb,0.6,0)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]  # 这里LS[0]为高斯金字塔的最小图片
    for i in range(1, 4):  # 第一次循环的图像为高斯金字塔的最小图片，依次通过拉普拉斯金字塔恢复到大图像
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])  # 采用金字塔拼接方法的图像
        #ls_ = cv2.addWeighted(ls_,1,LS[i],1,0)
    cv2.imwrite(resultaddr, ls_)
    cv2.imshow("Result",ls_)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

def try1():
    src = cv2.imread('Resource/VioletEvergarden/src_used.png')
    mask = cv2.imread('Resource/VioletEvergarden/mask.png')
    back = cv2.imread('Resource/VioletEvergarden/dst.jpg')
    resultaddr = 'Results/Pyramid_Violet_Outline.png'
    LaplacePyramid(src, mask, back, resultaddr, 350, 850)

def try2():
    src = cv2.imread('Resource/Scenery/src.jpg')
    mask = cv2.imread('Resource/Scenery/mask.png')
    back = cv2.imread('Resource/Scenery/dst.jpg')
    resultaddr = 'Results/Pyramid_Scenery.png'
    LaplacePyramid(src, mask, back, resultaddr)

def try3():
    src = cv2.imread("Resource/Multi-focus-Image-Fusion-Dataset/MF_Dataset/5fused.bmp")
    back = cv2.imread('Resource/Multi-focus-Image-Fusion-Dataset/MF_Dataset/5fused1.bmp', )
    resultaddr = 'Results/MultiFocus_Pyramid_bottle.png'
    LaplacePyramid(src, None, back, resultaddr)

#try1()
#try2()
try3()