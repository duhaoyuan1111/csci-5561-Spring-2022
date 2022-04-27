import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

#!pip3 install opencv-python==3.4.2.17
#!pip3 install opencv-contrib-python==3.4.2.17
# Need this version, otherwise bugs happen
# same as hw2 :(

def find_match(img1, img2):
    # same as hw2
    s1 = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = s1.detectAndCompute(img1,None)
    #print(descriptors_1)
    s2 = cv2.xfeatures2d.SIFT_create()
    keypoints_2, descriptors_2 = s2.detectAndCompute(img2,None)

    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(descriptors_2)
    dist, ind = neigh.kneighbors(descriptors_1)
    #http://amroamroamro.github.io/mexopencv/matlab/cv.SIFT.detectAndCompute.html
    arr1 = []
    arr2 = []
    count = 0
    for i in range(len(dist)):
        if dist[i,0] < 0.65 * dist[i,1]: #unique
            arr1.append(keypoints_1[i].pt)
            idx = ind[i,0]
            arr2.append(keypoints_2[idx].pt)
            count +=1
    arr1 = np.array(arr1).reshape(count,2)
    arr2 = np.array(arr2).reshape(count,2)

    neigh_2 = NearestNeighbors(n_neighbors=2)
    neigh_2.fit(descriptors_1)
    dist_2, ind_2 = neigh_2.kneighbors(descriptors_2)
    #http://amroamroamro.github.io/mexopencv/matlab/cv.SIFT.detectAndCompute.html
    aRR1 = []
    aRR2 = []
    con = 0
    for i in range(len(dist_2)):
        if dist_2[i,0] < 0.65 * dist_2[i,1]: #unique
            idx = ind_2[i,0]
            aRR1.append(keypoints_1[idx].pt)
            aRR2.append(keypoints_2[i].pt)
            con += 1
    aRR1 = np.array(aRR1).reshape(con,2)
    aRR2 = np.array(aRR2).reshape(con,2)

    temp1 = []
    temp2 = []
    for j in range(len(arr1)):
        for k in range(len(aRR1)):
            if arr1[j,0] == aRR1[k,0] and arr1[j,1] == aRR1[k,1]:
                temp1.append(arr1[k,:])
                temp2.append(arr2[k,:])
                break
    temp1 = np.array(temp1).reshape(len(temp1),2)
    temp2 = np.array(temp2).reshape(len(temp2),2)

    return temp1, temp2


def compute_F(pts1, pts2):
    # lec 30 p49
    old = 0
    ransac_iter = 3000
    size = pts1.shape[0]

    for i in range(ransac_iter):
        A = []
        rng = np.random.default_rng()
        rand = rng.choice(size,8,replace=False)
        #print(rand)
        rand = np.array(rand)
        p1 = pts1[rand,:]
        p2 = pts2[rand,:]
        for j in range(len(rand)):
            p1_pad = p1[j]
            p1_pad = np.append(p1_pad,1)
            p2_pad = p2[j]
            p2_pad = np.append(p2_pad,1)
            product = np.outer(p2_pad,p1_pad)
            product = product.reshape(-1)
            A.append(product)
        #print(np.array(A))
        u,s,vh = np.linalg.svd(np.array(A),full_matrices=True)
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
        vh = vh.T
        r_col = len(vh[0])-1
        fundamental = vh[:,r_col]
        uu,ss,vvhh = np.linalg.svd(fundamental.reshape((3,3)),full_matrices=True)
        ss[len(ss)-1] = 0
        F = np.matmul(uu,np.diag(ss))
        F = np.matmul(F,vvhh).reshape(np.size(F),-1)
        ans = []
        for k in range(pts1.shape[0]):
            p1_pad = pts1[k]
            p1_pad = np.append(p1_pad,1)
            p2_pad = pts2[k]
            p2_pad = np.append(p2_pad,1)
            product = np.outer(p2_pad,p1_pad)
            product = product.reshape(-1)
            ans.append(product)
        ans = np.array(ans)
        ans = ans.reshape(size,-1)
        result = np.absolute(np.matmul(ans,F))
        howMany = np.sum(np.where(result > 0.02, 0, 1))
        if howMany > old:
            old = howMany
            F_true = F

    # print(F_true)
    return F_true.reshape(3,3)



def triangulation(P1, P2, pts1, pts2):
    pts3D = []
    for i in range(pts1.shape[0]):
        p1_pad = pts1[i,:]
        p1_pad = np.append(p1_pad,1)
        skew_p1 = [[0,-1*p1_pad[2],p1_pad[1]],[p1_pad[2],0,-1*p1_pad[0]],[-1*p1_pad[1],p1_pad[0],0]]
        p2_pad = pts2[i,:]
        p2_pad = np.append(p2_pad,1)
        skew_p2 = [[0,-1*p2_pad[2],p2_pad[1]],[p2_pad[2],0,-1*p2_pad[0]],[-1*p2_pad[1],p2_pad[0],0]]
        skew_p1 = np.array(skew_p1).reshape(3,3)
        skew_p2 = np.array(skew_p2).reshape(3,3)
        # print(skew_p1)
        top = np.matmul(skew_p1,P1)
        bottom = np.matmul(skew_p2,P2)
        matr = np.stack((top,bottom)).reshape((6,4))
        # print(matr)
        u,s,vh = np.linalg.svd(matr,full_matrices=True)
        vh = vh.T
        r_col = len(vh[0])-1
        temp = vh[:,r_col]
        # print(temp) #4
        pts = temp / temp[3]
        pts3D.append(pts[:3])

    pts3D = np.array(pts3D)
    return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):
    
    n = len(Rs)
    old = 0
    best = 0
    for i in range(n):
        temp = np.array(Rs[i])
        rotation = temp[2,:]
        for j in range(len(pts3Ds[i])):
            count = 0
            p3d = pts3Ds[i][j].reshape(3,1)
            cam1 = np.matmul(rotation,(p3d-Cs[i]))
            cam2 = pts3Ds[i][j][2]
            if cam1 > 0 and cam2 > 0:
                count+=1
        if count > old:
            old = count
            best = i
    pts3D = pts3Ds[best]
    R = Rs[best]
    C = Cs[best]
    return R, C, pts3D


def compute_rectification(K, R, C):
    inverse_K = np.linalg.inv(K)
    rectification_rotation = []
    C_norm = np.linalg.norm(C,2)
    x = (C/C_norm).reshape(-1)
    rectification_rotation.append(x)
    temp = x[2]
    z = temp*x.T
    z[0] = -1*z[0]
    z[1] = -1*z[1]
    z[2] = 1-z[2]
    z = z/np.linalg.norm(z)
    z = z.T
    # y is perpendicular 
    y = np.cross(z,x)
    rectification_rotation.append(y.reshape(-1))
    rectification_rotation.append(z.reshape(-1))
    rr = np.array(rectification_rotation)
    H1 = np.matmul(np.matmul(K,rr),inverse_K)
    H2 = np.matmul(np.matmul(np.matmul(K,rr),R.T),inverse_K)
    return H1, H2


def dense_match(img1, img2):
    
    h,w = img1.shape
    s1 = cv2.xfeatures2d.SIFT_create()
    kp1 = []
    for i in range(h):
        for j in range(w):
            kp1.append(cv2.KeyPoint(j,i,4))
    kp1 = np.array(kp1)
    H,W = img2.shape
    s2 = cv2.xfeatures2d.SIFT_create()
    kp2 = []
    for i in range(H):
        for j in range(W):
            kp2.append(cv2.KeyPoint(j,i,4))
    kp2 = np.array(kp2)
    keypoints1,descriptors1 = s1.compute(img1,kp1)
    keypoints2,descriptors2 = s2.compute(img2,kp2)
    descriptors1 = descriptors1.reshape(h,w,128)
    descriptors2 = descriptors2.reshape(H,W,128) # h=H
    disparity = np.zeros((h,w))
    for m in range(h):
        for n in range(w):
            if img1[m,n] != 0:
                img1_d = descriptors1[m,n,:]
                res = []
                for p in range(n+1):
                    img2_d = descriptors2[m,p]
                    norm = np.linalg.norm((img1_d-img2_d),2)
                    res.append(norm)
                res = np.array(res)
                disparity[m,n] = np.argmin(res)-n
    # print(disparity)
    disparity = np.absolute(disparity)

    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, -el[2] / el[1]), (img.shape[1], (-img_width * el[0] - el[2]) / el[1])
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    #img_left = cv2.imread('/content/drive/MyDrive/hw5/left.bmp', 1)
    #img_right = cv2.imread('/content/drive/MyDrive/hw5/right.bmp', 1)
    visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 4: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 7: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    disparity = dense_match(img_left_w, img_right_w)
    visualize_disparity_map(disparity)

    # save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
