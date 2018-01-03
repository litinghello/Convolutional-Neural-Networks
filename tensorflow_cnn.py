#coding:utf-8
from gen_captcha import read_text_image_vector
from gen_captcha import image_conversion
from gen_captcha import get_next_batch
from gen_captcha import vector_to_text
from gen_captcha import load_images

from gen_captcha import image_height
from gen_captcha import image_width
from gen_captcha import image_text_len
from gen_captcha import total_text_len

import argparse,http.server,base64,socket,uuid
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

X = tf.placeholder(tf.float32, [None, image_height*image_width])
Y = tf.placeholder(tf.float32, [None, image_text_len*total_text_len])
keep_prob = tf.placeholder(tf.float32)

cnn_layer1_kernel = [3, 3, 1, 16]
cnn_layer1_pool   = [1, 2, 2, 1]
cnn_layer2_kernel = [3, 3, 16, 32]
cnn_layer2_pool   = [1, 2, 2, 1]
cnn_layer3_kernel = [3, 3, 32, 64]
cnn_layer3_pool   = [1, 2, 2, 1]
cnn_layer_full    = 4096

def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, image_height, image_width, 1])
    w_c1 = tf.Variable(w_alpha*tf.random_normal(cnn_layer1_kernel))
    b_c1 = tf.Variable(b_alpha*tf.random_normal([cnn_layer1_kernel[3]]))
    conv1 = tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    pool1 = tf.nn.avg_pool(conv1, ksize=cnn_layer1_pool, strides=[1, 2, 2, 1], padding='SAME')
    layer1 = tf.nn.dropout(pool1, keep_prob)
 
    w_c2 = tf.Variable(w_alpha*tf.random_normal(cnn_layer2_kernel))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([cnn_layer2_kernel[3]]))
    conv2 = tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d(layer1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    pool2 = tf.nn.avg_pool(conv2, ksize=cnn_layer2_pool, strides=[1, 2, 2, 1], padding='SAME')
    layer2 = tf.nn.dropout(pool2, keep_prob)

    w_c3 = tf.Variable(w_alpha*tf.random_normal(cnn_layer3_kernel))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([cnn_layer3_kernel[3]]))
    conv3 = tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d(layer2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    pool3 = tf.nn.avg_pool(conv3, ksize=cnn_layer3_pool, strides=[1, 2, 2, 1], padding='SAME')
    layer3 = tf.nn.dropout(pool3, keep_prob)
    layer3_size = int (layer3.shape[1]*layer3.shape[2]*layer3.shape[3])

    w_d = tf.Variable(w_alpha*tf.random_normal([layer3_size, cnn_layer_full]))
    b_d = tf.Variable(b_alpha*tf.random_normal([cnn_layer_full]))
    dense = tf.reshape(layer3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.tanh(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha*tf.random_normal([cnn_layer_full, image_text_len*total_text_len]))
    b_out = tf.Variable(b_alpha*tf.random_normal([image_text_len*total_text_len]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    #out = tf.nn.softmax(out)
    return out

def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, image_text_len, total_text_len])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, image_text_len, total_text_len]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    Saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            batch_x, batch_y = get_next_batch(rd=True,batch_size=128)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1})
            print(step, loss_)
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(rd=True,batch_size=100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print("测试结果：error:%0.2f%%"%((1-acc)*100))
                if (acc > 0.98) and (loss_ < 0.00001):
                #if acc > 0.98:
                    Saver.save(sess, "./model/crack_capcha.model", global_step=step)
                    break
            if step % 5001 == 0:#五千次保存一次模型
                Saver.save(sess, "./model/crack_capcha.model", global_step=step)
            step += 1
def cnn_load_model():
    output = crack_captcha_cnn()
    Saver = tf.train.Saver()
    session = tf.Session()
    Saver.restore(session, tf.train.latest_checkpoint("./model"))
    predict = tf.argmax(tf.reshape(output, [-1, image_text_len, total_text_len]), 2)
    return session,predict

def cnn_crack_image(session,predict,image):
    text_list = session.run(predict, feed_dict={X: [image], keep_prob: 1})
    text = text_list[0].tolist()
    vector = np.zeros(image_text_len*total_text_len)
    i = 0
    for n in text:
        vector[i*total_text_len + n] = 1
        i += 1
    return vector_to_text(vector)    

cnn_session = 0
cnn_predict = 0

def save_discern_image(base64_str):
    image_file = Image.open(BytesIO(base64.b64decode(base64_str))).convert('L')
    image_value = image_conversion(np.array(image_file))# image_file = cv2.cvtColor(np.array(image_file), cv2.COLOR_BGR2RGB)
    if image_value.shape[0] != image_height * image_width:
        return ("please updata "+str(image_height)+"*"+str(image_width)+"pix image base64")
    else:
        return cnn_crack_image(cnn_session,cnn_predict,image_value)#识别图像

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        #path = self.path
        send_str = "Please use POST to upload.Content is base64 image."
        # Send response status code
        self.send_response(200)
        # Send headers
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(send_str)))
        self.end_headers()
        # Send message back to client
        #message = "Hello world!"
        self.wfile.write(bytes(send_str, "utf8"))
        return
    
    def do_POST(self):
        re_data = self.rfile.read(int(self.headers['content-length']))  
        re_str = str(re_data, encoding = "utf-8") 
        result = save_discern_image(re_str)
        # Send response status code
        self.send_response(200)
        # Send headers
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(result)))
        self.end_headers()
        # Send message back to client
        #message = "Hello world!"
        self.wfile.write(bytes(result, "utf8"))
        return
    
if __name__ == '__main__':
    #global cnn_session,cnn_predict
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",action="store_true",default=False,help="开启网络识别模式'-n'，不设置将进入训练或者测试模式")
    parser.add_argument("-t",action="store_true",default=False,help="开启训练请添加：'-t'，不设置将进入测试模式")
    parser.add_argument("-f",type=str,default="image",help="设置训练目录,默认'image/' 例如：'image/'")
    args = parser.parse_args()
    
    is_server = args.n
    is_tarin = args.t
    tarin_folder = args.f

    host_name = socket.getfqdn(socket.gethostname())
    host_ip = socket.gethostbyname(host_name)
    host_ip_list = socket.gethostbyname_ex(host_name)
    print(host_name,uuid.UUID(int = uuid.getnode()).hex[-12:],host_ip)
    print(host_ip_list)

    if is_server == True:
        print("start http server")
        cnn_session,cnn_predict = cnn_load_model()
        serverAddress = ('', 8080)
        server = http.server.HTTPServer(serverAddress, RequestHandler)
        server.serve_forever()
    else:
        text_list,text_image,text_vector = load_images(tarin_folder)
        if is_tarin == True :
            train_crack_captcha_cnn()
        else :
            error=0
            cnn_session,cnn_predict = cnn_load_model()
            for i in range(len(text_list)):
                text, image, vector = read_text_image_vector()
                predict_text = cnn_crack_image(cnn_session,cnn_predict,image)
                if text != predict_text:
                    #print("error")
                    print("error:{}<->{}".format(text, predict_text))
                    error += 1
                #else:
                    #print("success")
            print("error:%0.3f%%"%(error/len(text_list)*100))
