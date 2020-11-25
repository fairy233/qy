import cv2
import numpy as np
# import numba as nb

# from HDR.Methods import Read_Image
import random
from struct import pack

# from HDR.Methods import find_n

def rgbe2float(rgbe:np.ndarray)->np.ndarray:
    res = np.zeros((rgbe.shape[0], rgbe.shape[1], 3))
    p = rgbe[:, :, 3] > 0
    m = 2.0 ** (rgbe[:, :, 3][p] - 136.0)
    res[:, :, 0][p] = rgbe[:, :, 0][p] * m
    res[:, :, 1][p] = rgbe[:, :, 1][p] * m
    res[:, :, 2][p] = rgbe[:, :, 2][p] * m
    return np.array(res,dtype=np.float32)

# def float2rgbe(RGB:np.ndarray)->np.ndarray:
#     '''
#     从RGB浮点数转换为rgbe表示
#     :param RGB: RGB浮点数组，范围应当已经被规范到(0,1)
#     :return:
#     '''
#     rgbe=np.zeros([RGB.shape[0],RGB.shape[1],4],dtype=float)
#     p=np.max(RGB,axis=2)
#     find_n_v=np.vectorize(find_n)
#     p=find_n_v(p)
#     p=np.expand_dims(p,2)
#     p=np.array(p,dtype=float)
#     rgbe[:,:,:3]=RGB*256/(2**p)
#     rgbe[:,:,3:4]=p+128
#     # for i in range(RGB.shape[0]):
#     #     for j in range(RGB.shape[1]):
#     #         _,n=find_mn(np.max(RGB[i,j,:]))
#     #         rgbe[i,j,:3]=RGB[i,j,:]*256/(2**n)
#     #         rgbe[i,j,3]=128+n
#
#     return rgbe

'''获取HDR图像的四通道表示'''
def readHdr(fileName:str)->np.ndarray:
    fileinfo = {}
    with open(fileName, 'rb') as fd:
        tline = fd.readline().strip()
        if len(tline) < 3 or tline[:2] != b'#?':
            print('invalid header')
            return
        fileinfo['identifier'] = tline[2:]

        # while(tline[:1]==b'#'):
        tline=fd.readline().strip()

        if(tline[:1]==b'#'):
            tline = fd.readline().strip()
        while tline:
            n = tline.find(b'=')
            if n > 0:
                fileinfo[tline[:n].strip()] = tline[n + 1:].strip()
            tline = fd.readline().strip()

        tline = fd.readline().strip().split(b' ')
        fileinfo['Ysign'] = tline[0][0]
        fileinfo['height'] = int(tline[1])
        fileinfo['Xsign'] = tline[2][0]
        fileinfo['width'] = int(tline[3])

        data = [d for d in fd.read()]
        height, width = fileinfo['height'], fileinfo['width']
        if width < 8 or width > 32767:
            data.resize((height, width, 4))
            print("error")
            return rgbe2float(data)

        img = np.zeros((height, width, 4))
        dp = 0
        # c=0
        for h in range(height):
            if data[dp] != 2 or data[dp + 1] != 2:
                print('this file is not run length encoded')
                print(data[dp:dp + 4])
                return
            if data[dp + 2] * 256 + data[dp + 3] != width:
                print('wrong scanline width')
                return
            dp += 4
            for i in range(4):
                ptr = 0
                while (ptr < width):
                    if data[dp] > 128:
                        count = data[dp] - 128
                        if count == 0 or count > width - ptr:
                            print('bad scanline data')
                        img[h, ptr:ptr + count, i] = data[dp + 1]
                        ptr += count
                        dp += 2
                    else:
                        # if(data[dp]==127):
                            # c=c+1
                        count = data[dp]
                        dp += 1
                        if count == 0 or count > width - ptr:
                            print('bad scanline data')
                        img[h, ptr:ptr + count, i] = data[dp: dp + count]
                        ptr += count
                        dp += count
        # return rgbe2float(img)
        # return img,c
        return img

def saveHdr(filename:str,rgbe:np.ndarray)->bool:
    '''
    直接将rgbe格式的数据保存为"*.hdr"文件
    这样保存会导致文件大小和opencv等标准库保存的大小不同（即便数据完全不变）,这个问题暂未解决
    :param filename:
    :param rgbe:
    :return:
    '''
    if(rgbe.shape[1]<8 or rgbe.shape[1]>32767):
        print("The width of the hdr image must be in range(8,32767)")
        return False

    rgbe=rgbe.astype(int)

    with open(filename,'wb') as fw:
        fw.write(b'#?RGBE')
        fw.write(b'\n')
        fw.write(b'FORMAT=32-bit_rle_rgbe')
        fw.write(b'\n')
        fw.write(b'\n')

        fw.write(b'-Y ')
        fw.write(bytes(str(rgbe.shape[0]),'ansi'))
        fw.write(b' +X ')
        fw.write(bytes(str(rgbe.shape[1]),'ansi'))
        fw.write(b'\n')

        for j in range(rgbe.shape[0]):
            fw.write(pack('B', 2))
            fw.write(pack('B', 2))
            fw.write(pack('B', int(rgbe.shape[1] / 256)))
            fw.write(pack('B', int(rgbe.shape[1] % 256)))

            for i in range(4):
                value = rgbe[j, 0, i]
                same_length = 1
                dif_list=[]
                dif_list.append(rgbe[j,0,i])
                for k in range(1, rgbe.shape[1]):
                    if (rgbe[j, k, i] == value):
                        if (len(dif_list) > 1):
                            dif_list.pop(-1)
                            fw.write(pack('B',len(dif_list)))
                            for _,d in enumerate(dif_list):
                                fw.write(pack('B',d))
                            dif_list.clear()
                            dif_list.append(value)

                        if (same_length < 127):
                            same_length = same_length + 1
                        else:
                            fw.write(pack('B',255))
                            fw.write(pack('B',value))
                            same_length = 1
                    elif (rgbe[j, k, i] != value and same_length == 1):
                        value = rgbe[j, k, i]
                        if (len(dif_list) < 127):
                            dif_list.append(rgbe[j,k,i])
                        else:
                            fw.write(pack('B',127))
                            for _,d in enumerate(dif_list):
                                fw.write(pack('B',d))
                            dif_list.clear()
                            dif_list.append(value)
                    elif (rgbe[j, k, i] != value and same_length > 1):
                        fw.write(pack('B',128+same_length))
                        fw.write(pack('B',value))
                        value = rgbe[j, k, i]
                        same_length=1
                        dif_list=[value]


                if(len(dif_list)>1):
                    fw.write(pack('B',len(dif_list)))
                    for _,d in enumerate(dif_list):
                        fw.write(pack('B',d))
                elif(same_length>1):
                    fw.write(pack('B',128+same_length))
                    fw.write(pack('B',value))
                else:
                    fw.write(pack('B',1))
                    fw.write(pack('B',value))
    fw.close()
    return True
