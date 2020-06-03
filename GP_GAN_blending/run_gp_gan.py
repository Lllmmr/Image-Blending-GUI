import os
import uuid
import numpy as np

from chainer import serializers

from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.color import rgb2gray

import cv2

from GP_GAN_blending.model import EncoderDecoder

from GP_GAN_blending.gp_gan import gp_gan, ndarray_resize



"""
    Note: source image, destination image and mask image have the same size.
"""
# def blending(args):
#     # load image
#     obj  = img_as_float(imread(args.src)[:,:,:3])
#     bg   = img_as_float(imread(args.dst)[:,:,:3])
#     mask = img_as_float(imread(args.mask))
#     if len(mask.shape) == 3:
#         mask = mask[:, :, 0]
#
#     src_h, src_w, _ = obj.shape
#     src_h, src_w = int(src_h*args.ratio), int(src_w*args.ratio)
#     obj = ndarray_resize(obj, (src_h, src_w))
#     mask = ndarray_resize(mask, (src_h, src_w), order=0)
#
#     x, y = args.x, args.y
#     dst_h, dst_w, _ = bg.shape
#
#     left, top = max(0, -x), max(0, -y)
#     right, bottom = min(dst_w, x + src_w) - x, min(dst_h, y + src_h) - y
#     x, y = max(0, x), max(0, y)
#
#     new_obj = np.zeros_like(bg)
#     new_obj[y:y+bottom-top, x:x+right-left] = obj[top:bottom, left:right]
#
#     new_mask = np.zeros((dst_h, dst_w), bg.dtype)
#     new_mask[y:y+bottom-top, x:x+right-left] = mask[top:bottom, left:right]
#
#     blended_im = gp_gan(new_obj, bg, new_mask, G, 64, color_weight=args.color_weight)
#
#     name = '{}.png'.format(uuid.uuid4())
#     path = os.path.join('/tmp/imgs', name)
#     imsave(path, blended_im)
#
#     return {'path': os.path.join('images', name), 'status': 'success'}

# def MaskAndMove(src,dst,m,pos):
#     shape = dst.shape
#     newSrc = dst.copy()
#     newmask = np.zeros((shape[0],shape[1]))
#
#     y,x,w,h =cv2.boundingRect(m)
#
#     if(h % 2 ==1):
#         h+=1
#     if(w %2 == 1):
#         w +=1
#
#     newSrc[pos[1]-h//2-1:pos[1]+h//2+1,pos[0]-w//2-1:pos[0]+w//2+1] = src[x-1:x+h+1,y-1:y+w+1]
#     newmask[pos[1] - h//2:pos[1] + h//2, pos[0] - w//2:pos[0] + w//2] = m[x:x+h, y:y+w]
#
#     return newSrc,newmask
def MaskAndMove(src,dst,m,pos):
    shape = dst.shape
    newSrc = np.zeros_like(dst)
    newmask = np.zeros((shape[0],shape[1]))

    x,y,w,h =cv2.boundingRect(m)

    h,w = h//2,w//2
    left = min(x,pos[0]-w)
    right = min(src.shape[1]-x-w*2,shape[1]-pos[0]-w)
    top = min(y,pos[1]-h)
    bottom = min(src.shape[0]-y-h*2,shape[0]-pos[1]-h)

    newSrc[pos[1]-h-top:pos[1]+h+bottom,pos[0]-w-left:pos[0]+w+right] = src[y-top:y+h*2+bottom,x-left:x+w*2+right]
    #cv2.imshow()
    newmask[pos[1] - h:pos[1] + h, pos[0] - w:pos[0] + w] = m[y:y+h*2, x:x+w*2]

    return newSrc,newmask

def gpganblending(src,dst,mask,pos,color_weight=1):

    G = EncoderDecoder(64, 64, 3, 4000, image_size=64)
    serializers.load_npz('./GP_GAN_blending/blending_gan.npz', G)
    #mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    if len(mask.shape) == 3:
        mask = rgb2gray(mask)
    mask = mask.astype(src.dtype)
    obj, mask = MaskAndMove(src, dst, mask, pos)

    obj = img_as_float(obj)
    bg = img_as_float(dst)
    mask = img_as_float(mask)
    # load image
    # obj  = img_as_float(src)
    # bg   = img_as_float(dst)
    # mask = img_as_float(mask)
    # if len(mask.shape) == 3:
    #     mask = mask[:, :, 0]
    #
    # src_h, src_w, _ = obj.shape
    # #src_h, src_w = int(src_h*args.ratio), int(src_w*args.ratio)
    # #obj = ndarray_resize(obj, (src_h, src_w))
    # #mask = ndarray_resize(mask, (src_h, src_w), order=0)
    #
    # x = pos[0]
    # y = pos[1]
    #
    # dst_h, dst_w, _ = bg.shape
    #
    #
    #
    # left, top = max(0, -x), max(0, -y)
    # right, bottom = min(dst_w, x + src_w) - x, min(dst_h, y + src_h) - y
    # x, y = max(0, x), max(0, y)
    #
    #
    # new_obj = np.zeros_like(bg)
    # new_obj[y:y+bottom-top, x:x+right-left] = obj[top:bottom, left:right]
    #
    # new_mask = np.zeros((dst_h, dst_w), bg.dtype)
    # new_mask[y:y+bottom-top, x:x+right-left] = mask[top:bottom, left:right]


    blended_im = gp_gan(obj, bg, mask, G, 64, color_weight=color_weight)

    return blended_im

if __name__ == '__main__':
    src = cv2.imread("./source2.png")
    dst = cv2.imread("./dst2.jpg")
    mask = cv2.imread("./mask2.png")
#    mask[mask != 0] =255

    b = gpganblending(src, dst, mask, (200, 200))

    cv2.imwrite("b.jpg", b)