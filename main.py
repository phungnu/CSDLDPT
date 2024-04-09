import os
import cv2
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

# from cv2 import cv2
#chuan hoa hinh anh về trung bin 0 và độc lệch chuẩn 1
def nomalise (img):
    normed = (img-np.mean(img))/(np.std(img))
    return normed


def hog(img_gray, cell_size=8, block_size=4, bins=9):
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

    num_cell_x = w // cell_size  # sô lượng ô theo chiều ngang 150/8 =18
    num_cell_y = h // cell_size  # số lượng ô theo chiều dọc 150/8=18
    # print(num_cell_x)
    # print(num_cell_y)
    hist_tensor = np.zeros([num_cell_y, num_cell_x, bins])  # 18 x 18 x 9 khởi tạo ma trận hist_tensor với kích thước [num_cell_y, num_cell_x, bins], trong đó bins là số lượng bin trong histogram.
    for cx in range(num_cell_x):
        for cy in range(num_cell_y):
            ori = orientation[cy * cell_size : cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]#góc
            mag = magnitude[cy * cell_size : cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]#độ lớn
            # https://d...content-available-to-author-only...y.org/doc/numpy/reference/generated/numpy.histogram.html
            hist, _ = np.histogram(ori, bins=bins, range=(0, 180), weights=mag)  # tính histogram, kqua là 1-D vector, 9 elements = 9bins
            hist_tensor[cy, cx, :] = hist #Gán giá trị histogram (hist) cho vị trí tương ứng trong hist_tensor để lưu trữ histogram của mỗi ô
        pass
    pass

    # normalization
    redundant_cell = block_size - 1#1
    feature_tensor = np.zeros(
        [num_cell_y - redundant_cell, num_cell_x - redundant_cell, block_size * block_size * bins])#kích thước 17,17,36
    for bx in range(num_cell_x - redundant_cell):  # 17
        for by in range(num_cell_y - redundant_cell):  # 17
            by_from = by
            by_to = by + block_size
            bx_from = bx
            bx_to = bx + block_size
            v = hist_tensor[by_from:by_to, bx_from:bx_to, :].flatten()  # to 1-D array (vector) trải phẳng
            feature_tensor[by, bx, :] = v / linalg.norm(v, 2)
            # avoid NaN:
            if np.isnan(feature_tensor[by, bx, :]).any():  # avoid NaN (zero division)
                feature_tensor[by, bx, :] = v#chuẩn hóa

    return feature_tensor.flatten()

def extract(img_path, filename):
    print(img_path) #in đường dẫn của ảnh đang được xử lý
    img = cv2.imread(img_path)  # doc anh bang open cv
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # doc anh bang open cv
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # đổi thành ảnh xám
    # nomalise_img = nomalise(gray_img)
    _, binary_img = cv2.threshold(gray_img, 127, 255,cv2.THRESH_BINARY)  # nhị phân hóa, giá trị ngưỡng là 127, giá trị cực đại đưuọc ử dụng để nhị phân hóa ảnh là 255



    f = hog(binary_img)#trích xuất đặc trung từ ảnh xám
    print('Extracted feature vector of %s. Shape:' % img_path)#in đường dẫn của ảnh
    print('Feature size:', f.shape)#in kích thước của đặc trưng
    folderFeature = "./feature/"  + "/"#nơi lưu trữ tệp tin đặc trưng
    np.save(folderFeature + filename + "_feature.npy", f)#định dạng tệp tin đặc trưng
    print("DONE" + img_path)#thông báo DONE
    pass

def extract1(img_path, filename):
    print(img_path) #in đường dẫn của ảnh đang được xử lý
    img = cv2.imread(img_path)  # doc anh bang open cv
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # doc anh bang open cv
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # đổi thành ảnh xám
    # nomalise_img = nomalise(gray_img)
    _, binary_img = cv2.threshold(gray_img, 127, 255,
                                  cv2.THRESH_BINARY)  # nhị phân hóa, giá trị ngưỡng là 127, giá trị cực đại đưuọc ử dụng để nhị phân hóa ảnh là 255

    f1 = hog(binary_img)#trích xuất đặc trung từ ảnh xám
    print('Extracted feature vector of %s. Shape:' % img_path)#in đường dẫn của ảnh
    print('Feature size:', f1.shape)#in kích thước của đặc trưng
    folderFeature1 = "./ift/"  + "/"#nơi lưu trữ tệp tin đặc trưng
    np.save(folderFeature1 + filename + "_feature.npy", f1)#định dạng tệp tin đặc trưng
    print("DONE" + img_path)#thông báo DONE
    pass


def cosine_similarity(a, b):
    return np.dot(a.reshape(-1), b.reshape(-1)) / (np.linalg.norm(a) * np.linalg.norm(b))#dot(a,b) là tích vô hướng 2 vector a b, norm là độ dài 2 vector a b
#độ tương đồng cosine được trả về từ hàm là một giá trị từ 0 đến 1, trong đó giá trị gần 1 cho thấy hai vector có độ tương đồng cao



def clear_ift():
    folder_path = "./ift/"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Không thể xóa {file_path}: {e}")



if __name__ == '__main__':
    imgFolder = "./db/PNG/"
    inp = "./db/in/"
    np.seterr(divide='ignore', invalid='ignore')

    # for filename in os.listdir(imgFolder):
    #     extract(os.path.join(imgFolder, filename), os.path.splitext(filename)[0])

    folderFeature = "./feature/"
    features = np.array([[None] * 32400])
    for i in range(1, 190):
        filename = 'child-' + str(i) + '_feature.npy'
        f = np.load(folderFeature + '/' + filename, allow_pickle=True)
        features = np.append(features, [f.flatten()], axis=0)
        print('append feature image: %s' % filename)

    for filename in os.listdir(inp):
        extract1(os.path.join(inp, filename), os.path.splitext(filename)[0])
    folderFeature1 = "./ift/"
    features1 = np.array([[None] * 32400])
    for filename in os.listdir(folderFeature1):
        f1 = np.load(folderFeature1 + '/' + filename)
        features1 = np.append(features1, [f1.flatten()], axis=0)

    compare = []
    for i in range(1,len(features)):
        tmp = cosine_similarity(features[i], features1[1])
        compare.append([tmp,i])

    # #chuyển sang numpy
    comparenp = np.array(compare)

    # Xác định chỉ mục của vị trí đầu tiên trong mỗi hàng
    first_col_idx = np.argsort(comparenp[:, 0])

    # Sắp xếp lại các hàng dựa trên chỉ mục đó

    sorted_arr = comparenp[first_col_idx]

    pos1 = int(sorted_arr[-1][1])#vị trí ảnh giống nhất trong dataset
    pos2 = int(sorted_arr[-2][1])
    pos3 = int(sorted_arr[-3][1])

    res1 = 'child-' + str(pos1)
    res2 = 'child-' + str(pos2)
    res3 = 'child-' + str(pos3)

    print("anh giống với input nhất là ")
    print( res1 + ".png với độ tương đồng là "+str(float(sorted_arr[-1][0]) * 100))#in vị trí
    print( res2 + ".png với độ tương đồng là "+str(float(sorted_arr[-2][0]) * 100))
    print( res3 + ".png với độ tương đồng là "+str(float(sorted_arr[-3][0]) * 100))

    clear_ift()

    res_img = cv2.imread("./db/PNG/" + res1 + '.png')
    plt.imshow(res_img)
    plt.axis('off')  # xoa truc x y
    plt.show()

    res_img2 = cv2.imread("./db/PNG/" + res2 + '.png')
    plt.imshow(res_img2)
    plt.axis('off')  # xoa truc x y
    plt.show()

    res_img3 = cv2.imread("./db/PNG/" + res3 + '.png')
    plt.imshow(res_img3)
    plt.axis('off')  # xoa truc x y
    plt.show()