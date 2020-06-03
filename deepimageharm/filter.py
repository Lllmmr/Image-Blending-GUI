import cv2

blur = cv2.imread("blur.jpg")

src = cv2.imread("com.jpg")

result = cv2.ximgproc.guidedFilter(src,blur,20,2)

cv2.imwrite("r.jpg",result)
