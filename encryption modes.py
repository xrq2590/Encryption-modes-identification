# coding=utf-8
from Crypto.Cipher import AES,DES,DES3
from Crypto import Random
import binascii
import random as rand
'''
key = ''.join([rand.choice('0123456789abcdef')for jx in range(32)])   #秘钥，必须是16、24或32字节长度
iv = Random.new().read(16) #随机向量，必须是16字节长度
data = ''.join([rand.choice('0123456789abcdef')for jx in range(16)])
cipher1 = AES.new(key.encode(),AES.MODE_CFB,iv)  #密文生成器,MODE_CFB为加密模式

encrypt_msg =cipher1.encrypt(data.encode())  #附加上iv值是为了在解密时找到在加密时用到的随机iv
print(binascii.b2a_hex(encrypt_msg))  #将二进制密文转换为16机制显示
'''
'''
cipher2 = AES.new(key,AES.MODE_CFB,iv) #解密时必须重新创建新的密文生成器
decrypt_msg = cipher2.decrypt(encrypt_msg[16:]) #后十六位是真正的密文
'''

# coding=utf-8 3DES加密

for i in range(10000):

    key = ''.join([rand.choice('0123456789abcdef')for jx in range(16)])
    data =''.join([rand.choice('0123456789abcdef')for jy in range(16)])
    iv = Random.new().read(8)  #iv值必须是8位
    cipher1 = DES3.new(key.encode(),DES3.MODE_CBC,iv)  #密文生成器，采用MODE_OFB加密模式
    encrypt_msg =  iv + cipher1.encrypt(data.encode())
#附加上iv值是为了在解密时找到在加密时用到的随机iv,加密的密文必须是八字节的整数倍，最后部分
#不足八字节的，需要补位
    print(binascii.b2a_hex(encrypt_msg))   #将二进制密文转换为16进制显示
