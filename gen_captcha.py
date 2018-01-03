# coding:utf-8
# from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
# from PIL import Image
import os, random, base64, cv2
from urllib import parse,request
from PIL import Image
from io import BytesIO

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

image_height = 30  # 图像高度
image_width = 70  # 图像宽度
image_text_len = 4  # 图像最大字符串长度
image_name_start_i = -19
image_name_end_i = -15
#total_text = number + alphabet + ALPHABET + ['_']  # 所有包含数字
total_text = number
total_text_len = len(total_text)  # 所有长度


# 把字符串生成hex编码的数据用于计算训练误差
def text_to_vector(text):
    # text = text.lower()#大写转小写
    vector = np.zeros(total_text_len * image_text_len)
    for i in range(image_text_len):
        vector[total_text.index(text[i]) + i * total_text_len] = 1  # 数据稀疏编码 设置某个位置为1
    return vector

# 向量转回文本 根据编码生成对应的字符串
def vector_to_text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i in range(image_text_len):
        text.append(total_text[char_pos[i] - i * total_text_len])
    return "".join(text)

# 扫描文件
def scan_files(directory, prefix=None, postfix=None):
    files_list = []
    for root, sub_dirs, files in os.walk(directory):
        for special_file in files:
            if postfix:
                if special_file.endswith(postfix):
                    files_list.append(os.path.join(root, special_file))
            elif prefix:
                if special_file.startswith(prefix):
                    files_list.append(os.path.join(root, special_file))
            else:
                files_list.append(os.path.join(root, special_file))
    return files_list


text_list_len = 0  # 图像文件个数
text_list = []  # 文件列表
text_image = []  # 内存词典
text_vector = []  # 内存与向量对象

#dispaly_img = True
dispaly_img = False

def image_conversion(image_file,use_cv2=False):
    if use_cv2:
        image_cut = image_file[2:image_file.shape[0] - 2, 2:image_file.shape[1] - 2]  # 截取图像
        image_cut = cv2.copyMakeBorder(image_cut, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))  # 扩充
        image_cut = cv2.cvtColor(image_cut, cv2.COLOR_BGR2GRAY)  # 灰度化图像
        '''
        image_cut = cv2.GaussianBlur(image_cut, (3, 3), 0)#图像滤波
        #image_cut = cv2.adaptiveThreshold(image_cut, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31,20)  # 二化值图像
        image_cut = cv2.GaussianBlur(image_cut, (3, 3), 0)  # 图像滤波
        #image_cut = cv2.erode(image_cut,np.ones((2,2),np.uint8),iterations = 1)#腐蚀
        image_cut = cv2.dilate(image_cut,np.ones((3,3),np.uint8),iterations = 1)#膨胀
        #image_cut = cv2.morphologyEx(image_cut, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))  # 闭运算
        image_cut = cv2.morphologyEx(image_cut, cv2.MORPH_OPEN, np.ones((2, 1), np.uint8))#开运算
        image_cut = cv2.adaptiveThreshold(image_cut, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7,10)  # 二化值图像
        '''
        # ret,image_cut = cv2.threshold(image_cut,235,255,cv2.THRESH_BINARY)
        # print(image_cut.shape)
        image_cut = image_cut.flatten() / 255
        # image_at_corrosion = cv2.Canny(image_at_corrosion,30,1250)
        if (dispaly_img):
            cv2.imshow('img', cv2.resize(np.array(image_cut).reshape(50, 130), (260, 100), interpolation=cv2.INTER_CUBIC))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        '''
        #old
        image_file = cv2.imread(image_name)
        image_gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
        image_at_mean = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)
        image_norm = image_at_mean.flatten() / 255
        '''
        return image_cut
    else:
        image_file = np.array(image_file)
        image_out = image_file.flatten() / 255
        return image_out

def read_image_name(url):
    #image_file = cv2.imread(url)  # 读取图像
    image_file = Image.open(url).convert('L')
    return image_conversion(image_file)

# 从目录加载图像
def load_images(directory):
    global text_list_len, text_list, text_image, text_vector
    print("load image\n")
    # print(directory.split(','))
    for dir_em in directory.split(','):
        images_list = scan_files(dir_em, postfix=".jpg")  # 加载图像列表
        if len(images_list) != 0:
            text_list_len += len(images_list)  # 获取总图像个数
            for image_name in images_list:
                text = image_name[image_name_start_i:image_name_end_i]  # 截取文件名称
                image_norm = read_image_name(image_name)
                text_list.append(text)  # 生成列表
                text_image.append(image_norm)  # 添加文本与图像字典
                text_vector.append(text_to_vector(text))  # 添加文本树向量字典
            print("load '%s' complete" % (dir_em))
    print("all load complete")
    print("total text count:", len(text_list))
    print("total images count:", len(text_image))
    print("total vector count:", len(text_vector))
    return text_list, text_image, text_vector

text_list_index = 0  # 获取图像的位置
# 读取图像和文本
def read_text_image_vector(rd=False):
    global text_list_len, text_list_index
    index = 0
    if text_list_index >= text_list_len:
        text_list_index = 0
    if rd == False:  # 不是随机
        index = text_list_index  # 先增加再获取 顺序获取数据
        text_list_index += 1  # 移动到下一位置
    else:
        index = random.randint(0, text_list_len - 1)

    return text_list[index], text_image[index], text_vector[index]

# 生成一个训练batch 加载数据
def get_next_batch(rd=False, batch_size=128):
    batch_x = np.zeros([batch_size, image_height * image_width])  # 生成输入向量
    batch_y = np.zeros([batch_size, image_text_len * total_text_len])  # 生成输出向量
    for i in range(batch_size):
        text, image, vector = read_text_image_vector(rd)
        batch_x[i, :] = image  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = vector
    return batch_x, batch_y

def download_image_vector(num,url):
    global text_list_len, text_list, text_image, text_vector
    header_dict = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko'}
    for i in range(num):
        text = str(random.randint(0000, 9999)).zfill(4)
        new_url = url + "?name=" + text
        req = request.Request(url=new_url, headers=header_dict)
        res = request.urlopen(req)
        data = res.read()

        image_file = Image.open(BytesIO(base64.b64decode(data))).convert('L')
        image_cut = image_conversion(np.array(image_file))#image_file = cv2.cvtColor(np.array(image_file), cv2.COLOR_BGR2RGB)

        text_list.append(text)
        text_image.append(image_cut)
        text_vector.append(text_to_vector(text))
        text_list_len = i
        print("load '%s' \n" % (i))
    return text_list, text_image, text_vector

#text_list, text_image, text_vector = download_image_vector(1000, "http://robot.xmxing.net/get_code/Vcode0.php")
# text_list,text_image,text_vector = load_images("image/,test/,image1/,image2/,image3/")

'''
text_list,text_image,text_vector = load_images("image/")
for i in range(10):
    text, image, vector = read_text_image_vector(rd=True)
    #print(text, image, vector)
    print(text)
    cv2.imshow('img',cv2.resize(np.array(image).reshape(50,130),(260,100),interpolation=cv2.INTER_CUBIC))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
'''
image_file = cv2.imread("image/4yNP.jpg")#读取图像
image_gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
image_at_mean = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)#最后参数设置块大小与
image_total  = image_at_mean.flatten() / 255
cv2.imshow('img',cv2.resize(image_at_mean,(1300,500),interpolation=cv2.INTER_CUBIC))
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
cv2.imshow('img', cv2.resize(np.array(read_image_name("image/0008_4000000152.jpg")).reshape(30, 70), (140, 60), interpolation=cv2.INTER_CUBIC))
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
