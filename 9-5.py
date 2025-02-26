# pixellib를 이용한 정지 영상에 대한 의미 분할
# pixeelib 모듈과 정확한 동작은 tensorflow==2.5 이하에서 구현 잘 됨
# numpy는 1.23 버전으로 작업하면 좋음
from pixellib.semantic import semantic_segmentation
import cv2 as cv


seg = semantic_segmentation()
seg.load_ade20k_model('./data/deeplabv3_xception65_ade20k.h5')

img_fname = './data/busy_street.jpg'
seg.segmentAsAde20k(img_fname, output_image_name='image_new1.jpg')       # 파일로 저장
seg.segmentAsAde20k(img_fname, overlay=True, output_image_name='image_new2.jpg')       # 파일로 저장
info1, img_segmented1 = seg.segmentAsAde20k(img_fname)
info2, img_segmented2 = seg.segmentAsAde20k(img_fname, overlay=True)

cv.imshow('Image original', cv.imread(img_fname))
cv.imshow('Image segmentation', img_segmented1)
cv.imshow('Image segmentation overlayed', img_segmented2)

cv.waitKey()
cv.destroyAllWindows()
