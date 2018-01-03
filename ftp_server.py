# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:36:44 2017

@author: first
"""

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

import socket,uuid

host_name = socket.getfqdn(socket.gethostname())
host_ip = socket.gethostbyname(host_name)
host_ip_list = socket.gethostbyname_ex(host_name)
print(host_name,uuid.UUID(int = uuid.getnode()).hex[-12:],host_ip)
print(host_ip_list)


#实例化虚拟用户，这是FTP验证首要条件
authorizer = DummyAuthorizer()

#添加用户权限和路径，括号内的参数是(用户名， 密码， 用户目录， 权限)
authorizer.add_user('user', '12345', './image', perm='elradfmw')

#添加匿名用户 只需要路径
authorizer.add_anonymous('./image')

#下载上传速度设置
#dtp_handler = ThrottledDTPHandler
#dtp_handler.read_limit = settings.max_download
#dtp_handler.write_limit = settings.max_upload

#初始化ftp句柄
handler = FTPHandler
handler.authorizer = authorizer
#日志记录
#logging.basicConfig(filename=settings.loging_name, level=logging.INFO)
#添加被动端口范围
#handler.passive_ports = range(2000, 2333)
#欢迎信息
#handler.banner = settings.welcome_msg
#监听ip 和 端口,因为linux里非root用户无法使用21端口，所以我使用了2121端口
server = FTPServer((host_ip, 21), handler)
#最大连接数
#server.max_cons = settings.max_cons
#server.max_cons_per_ip = settings.max_per_ip
#开始服务
#server.serve_forever()
