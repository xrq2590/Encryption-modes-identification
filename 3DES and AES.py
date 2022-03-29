# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:38:22 2019

@author: zhhandsome
"""
import random

'''
import random as rand

def hex2bin(text):
    ret = ''
    for i in text:
        tmp = bin(int(i, 16))[2:]
        ret += '0'*(4-len(tmp)) + tmp
    return ret

def bin2hex(text):
    ret = ''
    for i in range(0, len(text), 4):
        tmp = text[i:i+4]
        ret += hex(int(tmp, 2))[2:]
    return ret

class DES_ECB:
    def __init__(self, text, key, mode):
        self.original_text = text
        self.text = text
        self.key = key
        self.mode = mode
        self.ring_shift_left = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]
        self.ring_shift_right = [0,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]
        self.PC_2 = [14,17,11,24,1,5,3,28,15,6,21,10,
                     23,19,12,4,26,8,16,7,27,20,13,2,
                     41,52,31,37,47,55,30,40,51,45,33,48,
                     44,49,39,56,34,53,46,42,50,36,29,32]
        self.S_box = [[] for i in range(8)]
        self.S_box[0] = [14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7,
                        0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8,
                        4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0,
                        15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13]
        self.S_box[1] = [15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10,
                        3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5,
                        0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15,
                        13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9]
        self.S_box[2] = [10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8,
                        13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1,
                        13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7,
                        1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12]
        self.S_box[3] = [7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15,
                        13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9,
                        10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4,
                        3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14]
        self.S_box[4] = [2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9,
                        14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6,
                        4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14,
                        11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3]
        self.S_box[5] = [12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11,
                        10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8,
                        9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6,
                        4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13]
        self.S_box[6] = [4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1,
                        13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6,
                        1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2,
                        6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12]
        self.S_box[7] = [13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7,
                        1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2,
                        7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8,
                        2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11]
        self.P_box = [16,7,20,21,29,12,28,17,1,15,23,26,5,18,31,10,
                      2,8,24,14,32,27,3,9,19,13,30,6,22,11,4,25]

    #初始置换，将64位的明文按照特定顺序置换
    def text_initial_permutation(self):
        index = 0
        D = ['' for i in range(64)]
        for i in range(58, 65, 2):
            for j in range(i, i-57, -8):
                D[index] = self.text[j-1]
                D[index+32] = self.text[j-2]
                index += 1
        self.text = ''.join(D)

    #密钥置换，64位密钥变成56位后置换
    def key_initial_permutation(self):
        index = 0
        permutationed_key = ['' for i in range(56)]
        for i in [57, 58, 59, 60, 63, 62, 61, 28]:
            if i not in [28, 60]:
                count = 8
            else:
                count = 4
            for j in range(count):
                permutationed_key[index] = self.key[i-1]
                index += 1
                i -= 8
        self.key = ''.join(permutationed_key)

    #根据给定的轮数生成子密钥并返回
    def generate_sub_key(self, rounds):
        left = self.key[0:28]
        right = self.key[28:56]
        if self.mode == 'E':
            shift_left_bits = self.ring_shift_left[rounds-1]
            left = left[shift_left_bits:] + left[:shift_left_bits]
            right = right[shift_left_bits:] + right[:shift_left_bits]
        else:
            shift_right_bits = self.ring_shift_right[rounds-1]
            left = left[-shift_right_bits:] + left[:-shift_right_bits]
            right = right[-shift_right_bits:] + right[:-shift_right_bits]
        self.key = left + right
        permutationed_sub_key = ['' for i in range(48)]
        index = 0
        for i in self.PC_2:
            permutationed_sub_key[index] = self.key[i-1]
            index += 1
        return ''.join(permutationed_sub_key)

    #扩展置换数据右半部分并返回
    def expansion_permutation(self):
        right = self.text[32:]
        expansion_right = ['' for i in range(48)]
        expansion_right[0] = right[-1]
        expansion_right[-1] = right[0]
        index = 1
        count = 0
        for i in range(1, 47):
            expansion_right[i] = right[index-1]
            count += 1
            if count == 5:
                index -= 1
                count = -1
            else:
                index += 1
        return ''.join(expansion_right)

    #S盒变换,Rn扩展置换之后的48位与子密钥Kn异或以后输入S盒，将48位变成32位
    def S_box_permutation(self, text, key):
        xor_result = ''
        for i in range(48):
            xor_result += '0' if text[i]==key[i] else '1'
        after_S_box_data = ''
        for i in range(0, 48, 6):
            block = xor_result[i:i+6]
            row = int(block[0]+block[-1], 2)
            colume = int(block[1:5], 2)
            tmp = bin(self.S_box[i//6][16*row+colume])[2:]
            after_S_box_data += '0'*(4-len(tmp)) + tmp
        return after_S_box_data

    #P盒置换，用S盒置换得到后的32位用于输入，与P盒置换后的结果与数据的64位左边异或，然后左右交换
    def P_box_permutation(self, text):
        after_P_box_data = ['' for i in range(32)]
        index = 0
        for i in range(32):
            after_P_box_data[index] = text[self.P_box[i]-1]
            index += 1
        after_P_box_data = ''.join(after_P_box_data)
        left = self.text[0:32]
        xor_result = ''
        for i in range(32):
            xor_result += '0' if left[i]==after_P_box_data[i] else '1'
        self.text = self.text[32:] + xor_result

    #逆置换
    def inverse_permutation(self):
        self.text = self.text[32:] + self.text[0:32]
        inverse_data = ['' for i in range(64)]
        current = 39
        count = 16
        index = 0
        while(count):
            inverse_data[index] = self.text[current]
            inverse_data[index+16] = self.text[current-2]
            inverse_data[index+32] = self.text[current-4]
            inverse_data[index+48] = self.text[current-6]
            index += 1
            if count%2 == 0:
                current -= 32
            else:
                current += 40
            count -= 1
            if count == 8:
                current = 38
        self.text = ''.join(inverse_data)

    #执行操作
    def do_final(self):
        self.text_initial_permutation()
        self.key_initial_permutation()
        for i in range(1, 17):
            sub_key = self.generate_sub_key(i)
            expansioned_right = self.expansion_permutation()
            right_after_S_box = self.S_box_permutation(expansioned_right, sub_key)
            self.P_box_permutation(right_after_S_box)
        self.inverse_permutation()
        #print('原文:', bin2hex(self.original_text))
        print(bin2hex(self.text))
        return self.text

#初始明文
for i in range(20000):
    x = ''.join([rand.choice('0123456789abcdef')for jx in range(16)])
    y = ''.join([rand.choice('0123456789abcdef')for jy in range(16)])
    text = hex2bin(x)
#初始密钥
    key = hex2bin(y)
    des_encrypt = DES_ECB(text, key, 'E')
    cipher_text = des_encrypt.do_final()
#des_decrypt = DES_ECB(cipher_text, key, 'D')
#des_decrypt.do_final()
'''
'''
from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex
from Crypto import Random


class AesEncryption(object):
    def __init__(self, key, mode=AES.MODE_CFB):
        self.key = self.check_key(key)
        # 密钥key长度必须为16,24或者32bytes的长度
        self.mode = mode
        self.iv = Random.new().read(AES.block_size)

    def check_key(self, key):
        '检测key的长度是否为16,24或者32bytes的长度'
        try:
            if isinstance(key, bytes):
                assert len(key) in [16, 24, 32]
                return key
            elif isinstance(key, str):
                assert len(key.encode()) in [16, 24, 32]
                return key.encode()
            else:
                raise Exception(f'密钥必须为str或bytes,不能为{type(key)}')
        except AssertionError:
            print('输入的长度不正确')

    def check_data(self, data):
        '检测加密的数据类型'
        if isinstance(data, str):
            data = data.encode()
        elif isinstance(data, bytes):
            pass
        else:
            raise Exception(f'加密的数据必须为str或bytes,不能为{type(data)}')
        return data

    def encrypt(self, data):
        ' 加密函数 '
        data = self.check_data(data)
        cryptor = AES.new(self.key, self.mode, self.iv)
        return b2a_hex(cryptor.encrypt(data)).decode()

    def decrypt(self, data):
        ' 解密函数 '
        data = self.check_data(data)
        cryptor = AES.new(self.key, self.mode, self.iv)
        return cryptor.decrypt(a2b_hex(data)).decode()


if __name__ == '__main__':
    for i in range(20000):
        key = ''.join([rand.choice('0123456789abcdef')for jx in range(32)])
        data = ''.join([rand.choice('0123456789abcdef')for jx in range(16)])
        aes = AesEncryption(key)
        e = aes.encrypt(data)  # 调用加密函数
        d = aes.decrypt(e)  # 调用解密函数
        print(e)
'''
        #print(d)
  # coding=utf-8

from Crypto.Cipher import AES
from Crypto import Random
import binascii

def aes_encrypt(data):
    BLOCK_SIZE = 16
    from base64 import b64encode
    AES_KEY = b'xxx'
    AES_IV = b'xxxx'
    cipher = AES.new(AES_KEY, AES.MODE_CBC, AES_IV)
    x = data + (BLOCK_SIZE - len(data) % BLOCK_SIZE) * chr(BLOCK_SIZE - len(data) % BLOCK_SIZE)
    x = x.encode()
    e = b64encode(cipher.encrypt(x))
    return str(e, encoding='utf8')

aes_encrypt(''.join(random.choice))