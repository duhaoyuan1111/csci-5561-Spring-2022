
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import math


def Sift_Feature_Extraction():
    #!pip3 install opencv-python==3.4.2.17
    #!pip3 install opencv-contrib-python==3.4.2.17
    #import numpy as np
    #import cv2 as cv
    #img = cv.imread('Hyun_Soo_target1.jpg')
    #gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #sift = cv.xfeatures2d.SIFT_create()
    #kp = sift.detect(gray,None)
    #img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv.imwrite('sift_keypoints.jpg',img)
    return 0

def find_match(img1, img2):
    #keypoints
    x1 = []
    x2 = []
    s1 = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = s1.detectAndCompute(img1,None)
    #print(descriptors_1)
    
    s2 = cv2.xfeatures2d.SIFT_create()
    keypoints_2, descriptors_2 = s2.detectAndCompute(img2,None)
    #d1/d2 < 0.7
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(descriptors_2)
    dist, ind = neigh.kneighbors(descriptors_1)
    #http://amroamroamro.github.io/mexopencv/matlab/cv.SIFT.detectAndCompute.html
    #(array([[0.5],[1]]), array([[2],[1]]))

    count = 0
    for i in range(len(dist)):
        if dist[i, 0] / dist[i, 1] < 0.7: #unique
            x1.append(keypoints_1[i].pt)
            idx = ind[i,0]
            x2.append(keypoints_2[idx].pt)
            count += 1
    x1 = np.array(x1).reshape(count,2)
    x2 = np.array(x2).reshape(count,2)
    return x1, x2

def align_image_using_feature(x1, x2, ransac_thr = 8, ransac_iter = 1100):
    # To do
    #affine a11 ~ a23. 6/2 =3 points to locate
    size = len(x2) # 9
    inliermax = 0
    A = None
    #print(x1)
    for i in range(ransac_iter):
        rng = np.random.default_rng()
        rand = rng.choice(size,3,replace=False)
        #print(rand)
        rand = np.array(rand)
        x1sam = x1[rand,:]
        x1sam = np.append(x1sam, [[1],[1],[1]],axis=1) # 3*3
        x2sam = x2[rand,:] # 2*3
        if np.linalg.matrix_rank(x1sam)<len(x1sam):
            continue
        X = np.matmul(np.linalg.inv(x1sam),x2sam)
        X3by3 = X.T # [a11,a12,a13][a21,a22,a23] 2*3
        X3by3 = np.append(X3by3, [[0],[0],[1]]).reshape(3,3) # 3*3
        #print(X3by3)
        #RANSAC
        x1p = np.concatenate((x1,np.ones((size,1))),axis=1) # 9*3
        x2p = np.concatenate((x2,np.ones((size,1))),axis=1) # 9*3
        #print(x2p)
        count = 0
        for i in range(x1.shape[0]):
            targetHat = np.matmul(X3by3,x1p[i])
            eucli = np.linalg.norm(targetHat - x2p[i],2)
            #print(eucli)
            if eucli < ransac_thr:
                count+=1
        #print(count)
        if count > inliermax:
            inliermax = count
            A = X3by3
    #print(A)
    return A

def warp_image(img, A, output_size):
    
    h,w = output_size
    img_warped = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            tp = np.matmul(A[:2,:],np.array([j,i,1]))#horizontal first
            tp_x = tp[0]
            tp_y = tp[1]
            xinbox = np.array([math.ceil(tp_x)-tp_x,tp_x%1])#1*2
            yinbox = np.array([math.ceil(tp_y)-tp_y,tp_y%1]).reshape(2,1)#2*1
            lefttop = img[math.floor(tp_y),math.floor(tp_x)]
            rightbottom = img[math.ceil(tp_y),math.ceil(tp_x)]
            box = np.array([[lefttop,lefttop],[rightbottom,rightbottom]])
            img_warped[i,j] = np.matmul(np.matmul(xinbox,box),yinbox)#1*1

    return img_warped


def get_differential_filter():
    # convolution filter
    filter_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    filter_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    return filter_x, filter_y


def filter_image(im, filter):
    # each pixel has only one value instead of r,g,b. Because it's grey scale
    height = im.shape[0] # before padding
    width = im.shape[1]
    im_filtered = np.zeros((height, width))
    # pad 0 (up,down) (left,right)
    im = np.pad(im, ((1,1),(1,1)))
    for i in range(1, height):
        for j in range(1, width):
            temp = np.multiply(im[i-1:i+2, j-1:j+2], filter) # i-1 i i+1
            im_filtered[i,j] = np.sum(temp)
    return im_filtered

def align_image(template, target, A):
    # To do
    m,n = np.shape(template)
    filx,fily = get_differential_filter()
    im_dx = filter_image(template,filx)
    #print(im_dx)
    im_dy = filter_image(template,fily)
    sdi = np.zeros((m,n,6))
    partial = np.zeros((m,n,2))
    for i in range(m):
        for j in range(n):
            partial[i,j] = [im_dx[i,j],im_dy[i,j]]
            sdi[i,j] = np.matmul(partial[i,j], np.array([[i,j,1,0,0,0],[0,0,0,i,j,1]])) #1*6
    hess = np.zeros((6,6))
    for i in range(m):
        for j in range(n):
            hess += np.matmul(sdi[i,j].reshape(6,1), sdi[i,j].reshape(1,6))
    #print(hess)
    err_norm_array = []
    A_refined = A
    prev = 0
    for u in range(110):
        warped = warp_image(target, A_refined, template.shape)
        error = warped-template
        f = np.zeros((6,1))
        for i in range(m):
            for j in range(n):
                f +=  ((sdi[i, j].T) * error[i, j]).reshape(6,1)
                
        delta_p = np.matmul(np.linalg.inv(hess), f)
        #print(delta_p)
        affine_delp = np.array([[delta_p[0][0]+1,delta_p[1][0],delta_p[2][0]],[delta_p[3][0],delta_p[4][0]+1,delta_p[5][0]],[0,0,1]])
        A_refined = np.matmul(A_refined, np.linalg.inv(affine_delp))
        err_norm = np.linalg.norm(error,2)
        err_norm_array.append(err_norm)
        #print(u, " --- " , abs(err_norm - prev))
        if abs(err_norm - prev) < 1:
            break
        prev = err_norm
    err_norm_array = np.array(err_norm_array)
    return A_refined,err_norm_array


def track_multi_frames(template, img_list):
    # To do
    temp = template
    A_align_list = []
    for i in range(len(img_list)):
        cur_tar = img_list[i]
        x,y = find_match(temp, cur_tar)
        A = align_image_using_feature(x, y, 8, 1100)
        A_refined,err_norm_array = align_image(temp, cur_tar, A)
        A_align_list.append(A_refined)
        #update
        temp = np.array(warp_image(cur_tar, A_refined, temp.shape),dtype='uint8')
    return np.array(A_align_list)

    

def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    A = align_image_using_feature(x1, x2, ransac_thr = 8, ransac_iter = 1100)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[0], A)
    visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)


