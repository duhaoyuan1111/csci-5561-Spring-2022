
import numpy as np
import math
import main_functions as main


##from cnn import get_mini_batch, fc, relu, conv, pool2x2, flattening
##from cnn import train_slp_linear, train_slp, train_mlp, train_cnn
#mnist_train = sio.loadmat('/content/drive/MyDrive/hw4/mnist_train.mat')
#mnist_test = sio.loadmat('/content/drive/MyDrive/hw4/mnist_test.mat')
#if __name__ == '__main__':
    # main_slp_linear()
    # main_slp()
    # main_mlp()
    #main_cnn()

def get_mini_batch(im_train, label_train, batch_size):
    # batch_size: 32
    n_img, size = np.shape(im_train)
    total_batch = math.floor(size/batch_size)
    mini_batch_x = []
    mini_batch_y = []

    categories = 10
    for i in range(total_batch):
        random_batch = np.random.permutation(size) # generate random list, no repeat
        end_at = 0
        start_at = i*batch_size
        if (i+1)*batch_size > size:
            end_at = size
        else:
            end_at = (i+1)*batch_size
        cand_list = np.arange(start_at, end_at)
        result = random_batch[cand_list]
        # print(result)
        mini_batch_x.append(im_train[:,result].T)
        temp = np.identity(categories) # all possible one hot
        label = label_train[0][result]
        label = np.array([label]).reshape(-1)

        mini_batch_y.append(temp[label])

    mini_batch_x = np.array(mini_batch_x)
    mini_batch_y = np.array(mini_batch_y)
    
    return mini_batch_x, mini_batch_y #196¡Ábatch_size 10¡Ábatch_size


def fc(x, w, b):
    # n*m  **  m*1   + n*1
    # not always 196
    y = np.add(np.matmul(w,x),b)
    return y #n*1 10*1


def fc_backward(dl_dy, x, w, b, y):
    #dl_dy 1*n , w n*m
    dl_dx = np.matmul(dl_dy,w)
    dl_dw = np.multiply(dl_dy,x)
    # print(dl_dw.shape)
    dl_db = dl_dy
    return dl_dx, dl_dw, dl_db #1*m 1*n*m 1*n

def loss_euclidean(y_tilde, y):
    # euclidean
    y_tilde = np.array(y_tilde).reshape(-1)
    # print(y_tilde)
    # print(y.shape)
    l = np.linalg.norm(y_tilde-y)
    l = pow(l,2)
    dl_dy = 2*(y_tilde-y)

    return l, dl_dy #1*10


def loss_cross_entropy_softmax(x, y):
    # x m*1
    num = np.exp(x)
    den = np.sum(num)
    ytil = num/den
    ytil = np.array(ytil)
    l = np.sum(y*np.log(ytil))
    dl_dy = (ytil-y).reshape(1,-1)
    # print(y.shape)
    # print(ytil.shape)
    return l, dl_dy


def relu(x):
    # relu: x>0 -> x, x<0 -> 0
    y = np.array([np.size(x)])
    y = np.where(x<=0,x*0.00001,x)

    return y


def relu_backward(dl_dy, x, y):
    # dl_dy 1*z
    # print(y.shape)
    dydx = np.where(y>=0,1,0)
    dl_dx = np.multiply(dydx,dl_dy)
    return dl_dx


def conv(x, w_conv, b_conv):
    # TO DO
    # print(b_conv.shape)
    b_conv = np.array(b_conv)
    x = x.reshape(14,14,1)
    # b_conv.reshape((3,1))
    # print(b_conv.shape)
    H,W,c1 = np.shape(x)
    h,w,c1_copy,c2 = np.shape(w_conv)
    y = np.zeros((H,W,c2)) # H*W*C2
    for i in range(c1):
      pad_x = np.pad(x[:,:,i],((1,1),(1,1)))
      colim = []
      newH, newW = pad_x.shape
      for ii in range(newH+1-h):
        for jj in range(newW+1-h):
          stride = pad_x[ii:ii+h,jj:jj+h].reshape(-1)
          colim.append(stride)
      colim = np.array(colim)
      # colim = colim.T
      for j in range(c2):
        temp = w_conv[:,:,i,j].reshape((-1,1))
        product = np.matmul(colim,temp)
        product = product.reshape(H,W)
        # print(product.shape)
        update = product + b_conv[j,:]
        y[:,:,j] = y[:,:,j] + update

    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    x = x.reshape(14,14,1)
    H,W,c1 = x.shape
    h,w,c1_copy,c2 = w_conv.shape
    w_conv_shape = []
    b_conv_shape = []
    w_conv_shape = w_conv.shape
    b_conv_shape = b_conv.shape
    dl_dw = np.zeros(w_conv_shape)
    dl_db = np.zeros(b_conv_shape)
    #w
    for i in range(c2):
      for j in range(c1):
        tempdldy = dl_dy[:,:,i].reshape(-1)
        pad_x = np.pad(x[:,:,j],((1,1),(1,1)))
        colim = []
        newH, newW = pad_x.shape
        for ii in range(newH+1-h):
          for jj in range(newW+1-h):
            stride = pad_x[ii:ii+h,jj:jj+h].reshape(-1)
            colim.append(stride)
        colim = np.array(colim)
        # print(len(tempdldy))
        ans = np.matmul(tempdldy,colim)
        dl_dw[:,:,j,i] = ans.reshape(h,w)
    #b
    for k in range(c2):
      dl_db[k,:] = np.sum(dl_dy[:,:,k])

    return dl_dw, dl_db

def pool2x2(x):
    # H*W*C
    H,W,C = x.shape
    halfH = math.floor(H/2)
    halfW = math.floor(W/2)
    y = np.zeros((halfH,halfW,C))
    for i in range(C):
      pool = x[:,:,i]
      for j in range(halfH):
        for k in range(halfW):
          looppool = pool[2*j:2*j+2,2*k:2*k+2]
          maxima = np.amax(looppool)
          # print(maxima)
          y[j,k,i] = maxima

    return y

def pool2x2_backward(dl_dy, x, y):
    H,W,C = x.shape
    dl_dx = np.zeros((H,W,C))
    for i in range(C):
      for j in range(0,H,2):
        for k in range(0,W,2):
          stride = 2
          looppool = x[j:j+stride,k:k+stride,i]
          max_value_among_2by2 = np.argmax(looppool)
          halfJ = math.floor(j/stride)
          halfK = math.floor(k/stride)
          originaldldy = dl_dy[halfJ,halfK,i]
          height = math.floor(max_value_among_2by2/stride)+j
          width = math.floor(max_value_among_2by2%stride)+k
          dl_dx[height,width,i] = originaldldy
    dl_dx = dl_dx.reshape((14,14,3))

    return dl_dx #1*z


def flattening(x):
    h,w,col = x.shape
    y = x.reshape((h*w*col,1))
    return y


def flattening_backward(dl_dy, x, y):
    # TO DO
    h,w,col = x.shape
    dl_dx = dl_dy.reshape((h,w,col))
    return dl_dx

def train_slp_linear(mini_batch_x, mini_batch_y):

    lrn_rate = 0.02
    decay_rate = 0.95
    #Gaussian random
    w = np.random.normal(0, 0.5, size=(10, 196))
    b = np.random.normal(0, 0.5, size=(10, 1))
    k=0
    num_minibatch,size,temp = mini_batch_x.shape
    for i in range(2000):
        if i%1000 == 0:
            lrn_rate = lrn_rate*decay_rate
        dldwUpdate = np.zeros((10,196))
        dldbUpdate = np.zeros((10,1))
        for j in range(size):
            x = mini_batch_x[k,j,:].reshape(-1,1)
            # print(x.shape)
            y = mini_batch_y[k,j,:].reshape((1,10)) #1*10
            y_tilde = fc(x,w,b)
            L,dldy = loss_euclidean(y_tilde,y)
            dl_dx, dl_dw, dl_db = fc_backward(dldy, x, w, b, y)
            dldwUpdate += dl_dw.T
            dldbUpdate += dl_db.T
        k+=1
        if k>=num_minibatch:
            k = 0
        w -= lrn_rate/size* dldwUpdate
        b -= lrn_rate/size* dldbUpdate

    return w, b

def train_slp(mini_batch_x, mini_batch_y):

    lrn_rate = 0.1
    decay_rate = 0.95
    #Gaussian random
    w = np.random.normal(0, 0.5, size=(10, 196))
    b = np.random.normal(0, 0.5, size=(10, 1))
    k=0
    num_minibatch,size,temp = mini_batch_x.shape
    for i in range(2000):
        if i%1000 == 0:
            lrn_rate = lrn_rate*decay_rate
        dldwUpdate = np.zeros((10,196))
        dldbUpdate = np.zeros((10,1))
        for j in range(size):
            x = mini_batch_x[k,j,:].reshape(-1,1)
            # print(x.shape)
            y = mini_batch_y[k,j,:].reshape((1,10)) #1*10
            y_tilde = fc(x,w,b)
            L,dldy = loss_cross_entropy_softmax(y_tilde.reshape(-1),y)
            dl_dx, dl_dw, dl_db = fc_backward(dldy, x, w, b, y)
            dldwUpdate += dl_dw.T
            dldbUpdate += dl_db.T
        k+=1
        if k>=num_minibatch:
            k = 0
        w -= lrn_rate/size* dldwUpdate
        b -= lrn_rate/size* dldbUpdate

    return w, b

def train_mlp(mini_batch_x, mini_batch_y):

    lrn_rate = 0.22
    decay_rate = 0.92
    #Gaussian random. NEED avoid float overflow
    w1 = np.random.normal(0, 0.02, size=(30, 196))
    w2 = np.random.normal(0, 0.02, size=(10, 30))
    b1 = np.random.normal(0, 2, size=(30, 1))
    b2 = np.random.normal(0, 2, size=(10, 1))

    k=0
    num_minibatch,size,temp = mini_batch_x.shape
    for i in range(20000): # 18000-20000 0.912, 0.909, 0.899, 0.890, 0.918
        if i%1000 == 0:
            lrn_rate = lrn_rate*decay_rate
        dldw1Update = np.zeros((30,196))
        dldw2Update = np.zeros((10,30))
        dldb1Update = np.zeros((30,1))
        dldb2Update = np.zeros((10,1))
        for j in range(size):
            x = mini_batch_x[k,j,:].reshape(-1,1)
            layer1_a = fc(x, w1, b1)
            layer1_a_flat = layer1_a.reshape(-1)
            f = relu(layer1_a)
            f_flat = f.reshape(-1)
            # print(f.shape)
            # print(w2.shape)
            layer2_a = fc(f, w2, b2)
            layer2_a_flat = layer2_a.reshape(-1)
            y = mini_batch_y[k,j,:]
            l,dldlayer2_a = loss_cross_entropy_softmax(layer2_a.reshape(-1),y)
            dl_df,dl_dw2,dl_db2 = fc_backward(dldlayer2_a,f,w2,b2,y)
            # print(dl_dw2.shape)
            # print("second back ----")

            dldlayer1_a = relu_backward(dl_df,layer1_a_flat,f_flat)
            dl_dx1,dldw1,dldb1 = fc_backward(dldlayer1_a, x, w1, b1, layer1_a)
            dldw1Update += dldw1.T
            dldw2Update += dl_dw2.T
            dldb1Update += dldb1.T
            dldb2Update += dl_db2.T
        k+=1
        if k>=num_minibatch:
            k = 0
        # print(w1.shape)
        w1 -= lrn_rate/size* dldw1Update
        w2 -= lrn_rate/size* dldw2Update
        b1 -= lrn_rate/size* dldb1Update
        b2 -= lrn_rate/size* dldb2Update

    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    lrn_rate = 0.22
    decay_rate = 0.9
    #Gaussian random
    w_conv = np.random.normal(0, 0.5, size=(3,3,1,3))
    w_fc = np.random.normal(0, 0.5, size=(10,147))
    b_conv = np.random.normal(0, 1.5, size=(3,1))
    b_fc = np.random.normal(0, 1.5, size=(10,1))
    k=0
    num_minibatch,size,temp = mini_batch_x.shape
    for i in range(11000): #9500 [19£º07 0.933, 18:33 0.925, 18:24 0.908],10000 [20:16, 0.915]
        # 11000 [24:35 0.936 31:22 0.934 38:09 0.930 33:28 0.920]
        if i%1000 == 0:
            lrn_rate = lrn_rate*decay_rate
        dldw_convUpdate = np.zeros((3,3,1,3))
        dldw_fcUpdate = np.zeros((10,147))
        dldb_convUpdate = np.zeros((3,1))
        dldb_fcUpdate = np.zeros((10,1))
        for j in range(size):
            x = mini_batch_x[k,j,:].reshape((14,14,1))
            x = x.T
            conva = conv(x,w_conv,b_conv)
            reluy = relu(conva)
            pool2by2 = pool2x2(reluy)
            flat = flattening(pool2by2)
            layer_two_a = fc(flat,w_fc,b_fc)

            y = mini_batch_y[k,j,:].reshape((1,10)) #1*10
            L,dldlayer_two_a = loss_cross_entropy_softmax(layer_two_a.reshape(-1),y)
            # print(dl_dx.shape)
            dl_dflat, dl_dwfc, dl_dbfc = fc_backward(dldlayer_two_a,flat,w_fc,b_fc,layer_two_a)
            dl_dflat = dl_dflat.reshape(147,1)
            # chain rule
            dl_dpool2by2 = flattening_backward(dl_dflat,pool2by2,flat)
            # print(dl_dpool2by2.shape)
            dl_dreluy = pool2x2_backward(dl_dpool2by2,reluy,pool2by2)
            # print(dl_dreluy.shape)
            dl_dconva = relu_backward(dl_dreluy,conva,reluy)
            # print(dl_dconva.shape)
            dl_dwconv,dl_dbconv = conv_backward(dl_dconva,x,w_conv,b_conv,conva)

            # print(dl_dwconv.shape)
            dldw_convUpdate += dl_dwconv
            dldw_fcUpdate += dl_dwfc.T
            dldb_convUpdate += dl_dbconv
            dldb_fcUpdate += dl_dbfc.T

        k+=1
        if k>=num_minibatch:
            k = 0
        # print(dldw_convUpdate.shape)
        # print(dldw_fcUpdate.shape)
        # print(dldb_convUpdate.shape)
        # print(dldb_fcUpdate.shape)
        w_conv -= lrn_rate/size* dldw_convUpdate
        w_fc -= lrn_rate/size* dldw_fcUpdate
        b_conv -= lrn_rate/size* dldb_convUpdate
        b_fc -= lrn_rate/size* dldb_fcUpdate

    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()
