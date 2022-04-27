import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def get_gradient(im_dx, im_dy):
    # lec4 p12,13 dx = u, dy = v
    grad_mag = np.sqrt(np.power(im_dx, 2) + np.power(im_dy, 2)) # ||I||
    grad_angle = np.arctan2(im_dy, im_dx)
    # angle [0,pi)
    grad_angle = np.where(grad_angle<0, grad_angle+np.pi, grad_angle)
    grad_angle = grad_angle * 180 / np.pi # degree
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size = 8):
    m = grad_mag.shape[0] // cell_size
    n = grad_mag.shape[1] // cell_size
    ori_histo = np.zeros((m, n, 6))
    for i in range(m):
        for j in range(n):
            for k in range(cell_size): # loop smaller squares
                for l in range(cell_size):
                    whichbin = 0
                    temp = grad_angle[i*cell_size + k][j*cell_size + l] 
                    if temp < 15 or temp >= 165:
                        whichbin = 0
                    elif temp >= 15 and temp < 45:
                        whichbin = 1
                    elif temp >= 45 and temp < 75:
                        whichbin = 2
                    elif temp >= 75 and temp < 105:
                        whichbin = 3
                    elif temp >= 105 and temp < 135:
                        whichbin = 4
                    elif temp >= 135 and temp < 165:
                        whichbin = 5
                    ori_histo[i][j][whichbin] += grad_mag[i*cell_size + k][j*cell_size + l] 

    return ori_histo


def get_block_descriptor(ori_histo, block_size = 2):
    # To do
    e = 0.001
    x = ori_histo.shape[0] - (block_size-1)
    y = ori_histo.shape[1] - (block_size-1)
    ori_histo_normalized = np.zeros((x, y, 6*block_size*block_size))
    for i in range(x):
        for j in range(y):
            den = np.sqrt(np.sum(np.power(ori_histo[i:i+block_size+1, j:j+block_size+1, :],2) + np.power(e, 2)))
            num = ori_histo[i:i+block_size, j:j+block_size, :]
            ori_histo_normalized[i,j] = (num/den).reshape(-1) # 24 to 1

    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    
    fil_x, fil_y = get_differential_filter()
    im_dx = filter_image(im, fil_x)
    im_dy = filter_image(im, fil_y)
    mag, angle = get_gradient(im_dx, im_dy)
    ori_histo = build_histogram(mag, angle, 8)
    ori_histo_normalized = get_block_descriptor(ori_histo, 2)

    # visualize to verify
    #visualize_hog(im, ori_histo_normalized, 8, 2)

    return ori_histo_normalized


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()

    
def face_recognition(I_target, I_template):
    hog_template = extract_hog(I_template)
    tar_m, tar_n = np.shape(I_target) # larger
    tem_m, tem_n = np.shape(I_template)
    #lec6 44
    
    bounding_boxes = []
    for i in range(tar_m - tem_m):
        for j in range(tar_n - tem_n):
            hog_target = extract_hog(I_target[i:i+tem_m,j:j+tem_n])
            hogtar_m, hogtar_n, hogtar_vec = np.shape(hog_target)
            tarmean = np.mean(hog_target)
            temmean = np.mean(hog_template)
            num = 0.0
            a = 0.0
            b = 0.0
            den = 0.0
            for k in range(hogtar_m):
                for l in range(hogtar_n):
                    tarnorm = hog_target[k,l] - tarmean
                    temnorm = hog_template[k,l] - temmean
                    num += np.dot(tarnorm,temnorm)
                    a += np.sum(np.power(tarnorm, 2))
                    b += np.sum(np.power(temnorm, 2))

            den = np.sqrt(a*b)
            ncc = num / den
            if ncc > 0.48:
                bounding_boxes.append([j,i,ncc])
    ans = np.array(bounding_boxes).reshape(len(bounding_boxes), 3)
    size = np.shape(ans)[0]
    p1 = 0
    p2 = 1
    while True:
        ou = iou(ans[p1],ans[p2],tem_m)
        if ou > 0.5:
            if ans[p1, 2] > ans[p2, 2]:
                ans = np.delete(ans, p2, 0)
                size -= 1
            else:
                ans[[p1,p2],:] = ans[[p2,p1],:]
                ans = np.delete(ans, p2, 0)
                size -= 1
        else:
            p2 += 1
        if p2 == size:
            p1 += 1
            p2 = p1 + 1
        if p1 >= size - 1:
            break
    return ans

def iou(a, b, size):
    ax1 = a[0]
    ax2 = a[0]+size
    ay1 = a[1]
    ay2 = a[1]+size
    bx1 = b[0]
    bx2 = b[0]+size
    by1 = b[1]
    by2 = b[1]+size
    iou = 0.0
    inter = 0.0
    union = 0.0
    if ax1<=bx1 and ax2>=bx1 and ay1>=by1 and ay1<=by2:
      inter = (ax2-bx1)*(by2-ay1)
      union = size*size*2 - inter
      iou = inter / union
    elif ax1<=bx1 and ax2>=bx1 and ay1<=by1 and ay2>=by1:
      inter = (ax2-bx1)*(ay2-by1)
      union = size*size*2 - inter
      iou = inter / union
    elif ax1>=bx1 and bx2>=ax1 and by1<=ay1 and ay1<=by2:
      inter = (bx2-ax1)*(by2-ay1)
      union = size*size*2 - inter
      iou = inter / union
    elif ax1>=bx1 and bx2>=ax1 and by1>=ay1 and ay2>=by1:
      inter = (bx2-ax1)*(ay2-by1)
      union = size*size*2 - inter
      iou = inter / union
    return iou


def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()



if __name__=='__main__':
    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im)

    I_target= cv2.imread('target.png', 0)
    
    #I_target= cv2.imread('cameraman.tif', 0)
    #MxN image

    I_template = cv2.imread('template.png', 0)
    #mxn  face template
    #hog2 = extract_hog(I_target)
    bounding_boxes=face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    #I_target_c= cv2.imread('cameraman.tif')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    #this is visualization code.
    
