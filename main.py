import os
import cv2
import numpy as np
from numpy import linalg
# from cv2 import cv2


def hog(img_gray, cell_size=8, block_size=2, bins=9):
    img = img_gray#ảnh xám
    h, w = img.shape#kích thước chiều cao chiều rộng của ảnh

    # gradient
    xkernel = np.array([[-1, 0, 1]])#bộ lọc ngang
    ykernel = np.array([[-1], [0], [1]])#bộ lọc dọc
    dx = cv2.filter2D(img, cv2.CV_32F, xkernel)#tính gradient theo hướng x của ảnh img, kqua là ma trận dx
    dy = cv2.filter2D(img, cv2.CV_32F, ykernel)#tính gradient theo hướng y của ảnh img

    # histogram
    magnitude = np.sqrt(np.square(dx) + np.square(dy))#độ lớn gradient gx gy là căn bạc 2 tổng bình phuong
    orientation = np.arctan(np.divide(dy, dx + 0.00001))  # phương gradient(radian), cộng 0.00001 vì mẫu rất nhỏ nên để tránh chia cho 0 thì cộng them vào
    orientation = np.degrees(orientation) # chuyển radian sang độ, giá trị gốc là -90 -> 90
    orientation += 90  # cộng thêm 90 để góc chạy từ 0 đến 180 độ

    num_cell_x = w // cell_size  # sô lượng ô theo chiều ngang 248/8 =31
    num_cell_y = h // cell_size  # số lượng ô theo chiều dọc 338/8=42
    # print(num_cell_x)
    # print(num_cell_y)
    hist_tensor = np.zeros([num_cell_y, num_cell_x, bins])  # 42 x 31 x 9 khởi tạo ma trận hist_tensor với kích thước [num_cell_y, num_cell_x, bins], trong đó bins là số lượng bin trong histogram.
    for cx in range(num_cell_x):
        for cy in range(num_cell_y):
            ori = orientation[cy * cell_size : cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]#góc
            mag = magnitude[cy * cell_size : cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]#độ lớn
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
            hist, _ = np.histogram(ori, bins=bins, range=(0, 180), weights=mag)  # tính histogram, kqua là 1-D vector, 9 elements = 9bins
            hist_tensor[cy, cx, :] = hist #Gán giá trị histogram (hist) cho vị trí tương ứng trong hist_tensor để lưu trữ histogram của mỗi ô
        pass
    pass

    # normalization
    redundant_cell = block_size - 1#1
    feature_tensor = np.zeros(
        [num_cell_y - redundant_cell, num_cell_x - redundant_cell, block_size * block_size * bins])#kích thước 41,30,36
    for bx in range(num_cell_x - redundant_cell):  # 30
        for by in range(num_cell_y - redundant_cell):  # 41
            by_from = by
            by_to = by + block_size
            bx_from = bx
            bx_to = bx + block_size
            v = hist_tensor[by_from:by_to, bx_from:bx_to, :].flatten()  # to 1-D array (vector) trải phẳng
            feature_tensor[by, bx, :] = v / linalg.norm(v, 2)
            # avoid NaN:
            if np.isnan(feature_tensor[by, bx, :]).any():  # avoid NaN (zero division)
                feature_tensor[by, bx, :] = v#chuẩn hóa

    return feature_tensor.flatten()  # 44280 features = 36x30x41

def extract(img_path, filename):
    print(img_path) #in đường dẫn của ảnh đang được xử lý
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)#ảnh được đọc dưới dạng ảnh xám
    f = hog(img)#trích xuất đặc trung từ ảnh xám
    print('Extracted feature vector of %s. Shape:' % img_path)#in đường dẫn của ảnh
    print('Feature size:', f.shape)#in kích thước của đặc trưng
    folderFeature = "./feature/"  + "/"#nơi lưu trữ tệp tin đặc trưng
    np.save(folderFeature + filename + "_feature.npy", f)#định dạng tệp tin đặc trưng
    print("DONE" + img_path)#thông báo DONE
    pass

def extract1(img_path, filename):
    print(img_path) #in đường dẫn của ảnh đang được xử lý
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)#ảnh được đọc dưới dạng ảnh xám
    f1 = hog(img)#trích xuất đặc trung từ ảnh xám
    print('Extracted feature vector of %s. Shape:' % img_path)#in đường dẫn của ảnh
    print('Feature size:', f1.shape)#in kích thước của đặc trưng
    folderFeature1 = "./ift/"  + "/"#nơi lưu trữ tệp tin đặc trưng
    np.save(folderFeature1 + filename + "_feature.npy", f1)#định dạng tệp tin đặc trưng
    print("DONE" + img_path)#thông báo DONE
    pass
def cosine_similarity(a, b):
    return  np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))#dot(a,b) là tích vô hướng 2 vector a b, norm là độ dài 2 vector a b
#độ tương đồng cosine được trả về từ hàm là một giá trị từ 0 đến 1, trong đó giá trị gần 1 cho thấy hai vector có độ tương đồng cao








if __name__ == '__main__':
    imgFolder = "./db/PNG/"
    inp = "./db/in/"
    np.seterr(divide='ignore', invalid='ignore')
    # for character in range(ord('1_1.png'), ord('16_8.png')+1):
    #   print(chr(character))

    # for filename in os.listdir(imgFolder):
    #     extract(os.path.join(imgFolder, filename), os.path.splitext(filename)[0])
    list_file = []
    folderFeature = "./feature/"
    features = np.array([[None] * 44280])
    for filename in os.listdir(folderFeature):
        f = np.load(folderFeature + '/' + filename)
        features = np.append(features, [f], axis=0)
        list_file.append(filename)
    print(features)
    print(features.shape)

    for filename in os.listdir(inp):
        extract1(os.path.join(inp, filename), os.path.splitext(filename)[0])
    folderFeature1 = "./ift/"
    features1 = np.array([[None] * 44280])
    for filename in os.listdir(folderFeature1):
        f1 = np.load(folderFeature1 + '/' + filename)
        features1 = np.append(features1, [f1], axis=0)
    print(features1)
    print(features1.shape)

    # print(cosine_similarity(features[1], features1[1]))#57=7x8+1= ảnh 1_1 vì vector xếp ảnh 10_1 dtien
    # print(features[i])
    # print(features1[1])

    max = 0
    ans = 0
    for i in range(128):
        if i == 0:
            continue
        tmp = cosine_similarity(features[i], features1[1])#57=7x8+1= ảnh 1_1 vì vector xếp ảnh 10_1 dtien
        if tmp > max:
            max = tmp
            ans = i

    print("abc")
    print(list_file[ans-1])
    print("max")
    print(max)
    print(features[ans])
    print(features1[1])


