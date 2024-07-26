import cv2

img=cv2.imread("assets/images.jpg")

cv2.imshow("image_op_op",img)
cv2.waitKey(0)
cv2.destroyAllWindows()