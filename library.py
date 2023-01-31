import numpy as np 

def readYUV420(name: str, resolution: tuple, upsampleUV: bool = False):
    height = resolution[0]
    width = resolution[1]
    bytesY = int(height * width)
    bytesUV = int(bytesY/4)
    Y = []
    U = []
    V = []
    with open(name,"rb") as yuvFile:
        while (chunkBytes := yuvFile.read(bytesY + 2*bytesUV)):
            Y.append(np.reshape(np.frombuffer(chunkBytes, dtype=np.uint8, count=bytesY, offset = 0), (width, height)))
            U.append(np.reshape(np.frombuffer(chunkBytes, dtype=np.uint8, count=bytesUV, offset = bytesY),  (width//2, height//2)))
            V.append(np.reshape(np.frombuffer(chunkBytes, dtype=np.uint8, count=bytesUV, offset = bytesY + bytesUV), (width//2, height//2)))
    Y = np.stack(Y)
    U = np.stack(U)
    V = np.stack(V)
    if upsampleUV:
        U = U.repeat(2, axis=1).repeat(2, axis=2)
        V = V.repeat(2, axis=1).repeat(2, axis=2)
    return Y, U, V
    
def writeYUV420(name: str, Y, U, V, downsample=True):
    towrite = bytearray()
    if downsample:
        U = U[:, ::2, ::2]
        V = V[:, ::2, ::2]
    for i in range(Y.shape[0]):
        towrite.extend(Y[i].tobytes())
        towrite.extend(U[i].tobytes())
        towrite.extend(V[i].tobytes())
    with open(name, "wb") as destination:
        destination.write(towrite)

def readYUV420Range(name: str, resolution: tuple, range: tuple, upsampleUV: bool = False):
    height = resolution[0]
    width = resolution[1]
    bytesY = int(height * width)
    bytesUV = int(bytesY/4)
    Y = []
    U = []
    V = []
    with open(name,"rb") as yuvFile:
        startLocation = range[0]
        endLocation = range[1] + 1
        startLocationBytes = startLocation * (bytesY + 2*bytesUV)
        endLocationBytes = endLocation * (bytesY + 2*bytesUV)
        data = np.fromfile(yuvFile, np.uint8, endLocationBytes-startLocationBytes, offset=startLocationBytes).reshape(-1,bytesY + 2*bytesUV)
        Y = np.reshape(data[:, :bytesY], (-1, width, height))
        U = np.reshape(data[:, bytesY:bytesY+bytesUV], (-1, width//2, height//2))
        V = np.reshape(data[:, bytesY+bytesUV:bytesY+2*bytesUV], (-1, width//2, height//2))
    if upsampleUV:
        U = U.repeat(2, axis=1).repeat(2, axis=2)
        V = V.repeat(2, axis=1).repeat(2, axis=2)
    return Y, U, V

def YUV2RGB( yuv ):
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
    
    rgb = np.dot(yuv,m)
    rgb[:,:,:,0]-=179.45477266423404
    rgb[:,:,:,1]+=135.45870971679688
    rgb[:,:,:,2]-=226.8183044444304
    return rgb

def RGB2YUV( rgb ):
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
     
    yuv = np.dot(rgb,m)
    yuv[:,:,:,1:]+=128.0
    return yuv