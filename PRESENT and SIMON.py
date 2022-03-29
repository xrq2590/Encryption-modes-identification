def halfByteToBit(halfByte):
    res = []
    for i in range(0, len(halfByte)):
        for j in range(0, 4):
            res.append((halfByte[i] >> j) & 1)
    return res


def bitToHalfByte(state):
    res = []
    for i in range(0, int(len(state) / 4)):
        basePos = i * 4
        tmp = 0
        for j in range(0, 4):
            tmp |= (state[basePos + j] << j)
        res.append(tmp)
    return res


def numToBit(hexNumber, length):
    res = []
    for i in range(0, length):
        res.append((hexNumber >> i) & 1)
    return res


def bitToHex(state):
    res = 0
    for i in range(0, len(state)):
        res += state[i] * (1 << i)
    return format(res, 'x')


def bitToNum(state):
    res = 0
    for i in range(0, len(state)):
        res += state[i] * (1 << i)
    return res


def leftShift(state, length):
    res = list(state)
    for i in range(0, len(state)):
        pos = (i + length) % len(state)
        res[pos] = state[i]
    return res


def rightShift(state, length):
    res = list(state)
    for i in range(0, len(state)):
        pos = (i - length + len(state)) % len(state)
        res[pos] = state[i]
    return res


def bitXor(a, b):
    res = []
    for i in range(0, min(len(a), len(b))):
        res.append(a[i] ^ b[i])
    return res


def bitAnd(a, b):
    res = []
    for i in range(0, min(len(a), len(b))):
        res.append(a[i] & b[i])
    return res


def bitReverse(a):
    res = list(a)
    for i in range(0, len(a)):
        if a[i] == 0:
            res[i] = 1
        else:
            res[i] = 0
    return res


def bitAdd(a, b, n):
    mod = 1 << n
    aNum = bitToNum(a)
    bNum = bitToNum(b)
    return numToBit((aNum + bNum) % mod, len(a))


def bitNumberXor(a, b):
    bitB = numToBit(b, len(a))
    return bitXor(a, bitB)


class Present:
    def __init__(self, code, key):
        self.code = list(code)
        self.key = list(key)
        self.Sx = [0xc, 5, 6, 0xb, 9, 0, 0xa, 0xd, 3, 0xe, 0xf, 8, 4, 7, 1, 2]
        self.Ki = []
        self.Pi = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
                   4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
                   8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
                   12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63]

    def left64(self, key):
        beginPos = len(key) - 64
        res = []
        for i in range(0, 64):
            res.append(key[beginPos + i])
        return res

    def generateRoundKeys(self, key):
        nowKey = list(key)
        self.Ki.append([])
        for i in range(1, 33):
            self.Ki.append(self.left64(nowKey))
            # move 61
            moveTo = len(nowKey) - 61
            newKey = list(nowKey)
            for j in range(0, len(nowKey)):
                newKey[j] = nowKey[moveTo]
                moveTo += 1
                moveTo %= len(nowKey)
            # sx last byte
            halfByte = bitToHalfByte(newKey)
            halfByte[len(halfByte) - 1] = self.Sx[halfByte[len(halfByte) - 1]]
            # xor 15...19
            newKey = halfByteToBit(halfByte)
            for j in range(15, 20):
                newKey[j] = (newKey[j] ^ ((i >> (j - 15)) & 1))
            nowKey = list(newKey)
        return nowKey

    def addRoundKey(self, state, nowK):
        nowState = list(state)
        for i in range(0, 64):
            nowState[i] ^= self.Ki[nowK][i]
        return nowState

    def sBoxLayer(self, state):
        halfByte = bitToHalfByte(state)
        y = []
        for i in range(0, 16):
            y.append(self.Sx[halfByte[i]])
        return halfByteToBit(y)

    def pLayer(self, state):
        newState = list(state)
        for i in range(0, 64):
            newState[self.Pi[i]] = state[i]
        return newState

    def encode(self):
        state = self.code
        self.generateRoundKeys(self.key)
        for i in range(1, 32):
            state = self.addRoundKey(state, i)
            state = self.sBoxLayer(state)
            state = self.pLayer(state)
            # print(bitToHex(state))
            # print(bitToHex(self.Ki[i]))
        state = self.addRoundKey(state, 32)
        return state


class Simon:
    def __init__(self, codeX, codeY, keys):
        self.codeX = list(codeX)
        self.codeY = list(codeY)
        self.keys = list(keys)
        self.n = 16
        self.m = 4
        self.z = [
            [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1,
             0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, ],
            [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1,
             1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, ],
            [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0,
             0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, ],
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, ],
            [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1,
             1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, ]]

        self.T = 32
        self.j = 0

    def keyExpansion(self):
        for i in range(self.m, self.T):
            tmp = list(self.keys[i - 1])
            tmp = rightShift(tmp, 3)
            if self.m == 4:
                tmp = bitXor(tmp, self.keys[i - 3])
            sr1Tmp = rightShift(tmp, 1)
            tmp = bitXor(tmp, sr1Tmp)
            rkidm = bitReverse(self.keys[i - self.m])
            tmp = bitXor(rkidm, tmp)
            tmp = bitNumberXor(tmp, self.z[self.j][(i - self.m) % 62])
            tmp = bitNumberXor(tmp, 3)
            self.keys.append(tmp)

    def encode(self):
        self.keyExpansion()
        x = list(self.codeX)
        y = list(self.codeY)
        for i in range(0, self.T):
            tmp = list(x)
            s1x = leftShift(x, 1)
            s8x = leftShift(x, 8)
            s2x = leftShift(x, 2)
            s1xAs8x = bitAnd(s1x, s8x)
            x = bitXor(bitXor(bitXor(y, s1xAs8x), s2x), self.keys[i])
            y = list(tmp)
        return x, y


class Speck:
    def __init__(self, codeX, codeY, keys, alpha, beta):
        self.n = 16
        self.m = 4
        self.T = 22
        self.alpha = alpha
        self.beta = beta
        self.codeX = list(codeX)
        self.codeY = list(codeY)
        self.keys = []
        self.keys.append(keys[0])
        self.lKey = []
        for i in range(1, self.m):
            self.lKey.append(keys[i])

    def keyExpansion(self):
        for i in range(0, self.T - 1):
            ral = rightShift(self.lKey[i], self.alpha)
            kAddRal = bitAdd(self.keys[i], ral, self.n)
            self.lKey.append(bitNumberXor(kAddRal, i))
            lbk = leftShift(self.keys[i], self.beta)
            self.keys.append(bitXor(lbk, self.lKey[i + self.m - 1]))

    def encode(self):
        self.keyExpansion()
        x = list(self.codeX)
        y = list(self.codeY)
        for i in range(0, self.T):
            rax = rightShift(x, self.alpha)
            raxAddy = bitAdd(rax, y, self.n)
            x = bitXor(raxAddy, self.keys[i])
            lby = leftShift(y, self.beta)
            y = bitXor(lby, x)
        return x, y


def testPresent(code=0x0000000000000000, key=0x00000000000000000000):
    #print("persent:")
    codeList = numToBit(code, 64)
    keyList = numToBit(key, 80)
    present = Present(codeList, keyList)
    return bitToHex(present.encode())
    print(bitToHex(present.encode()))


def testSimon(x=0x6565, y=0x6877, keys=None):
    #print("simon:")
    if keys is None:
        keys = [0x0100, 0x0908, 0x1110, 0x1918]
    kBit = []
    for item in keys:
        kBit.append(numToBit(item, 16))
    simon = Simon(numToBit(x, 16), numToBit(y, 16), kBit)
    resX, resY = simon.encode()
    print(bitToHex(resX),bitToHex(resY))
    #print(bitToHex(resY))


def testSpeck(x=0x6565, y=0x6877, keys=None, alpha=7, beta=2):
    #print("speck:")
    if keys is None:
        keys = [0x0100, 0x0908, 0x1110, 0x1918]
    kBit = []
    for item in keys:
        kBit.append(numToBit(item, 16))
    speck = Speck(numToBit(x, 16), numToBit(y, 16), kBit, alpha, beta)
    resX, resY = speck.encode()
    print(bitToHex(resX),bitToHex(resY))
    #print(bitToHex(resY))


if __name__ == "__main__":
    # 以下所有参数均可以以0x为开头的16进制数表示

    # Present算法参数：
    # code为传入的明文，16个byte (64bits)
    # key为传入的初始秘钥，20个byte (80bits)
    # 调用函数后将在标准输出加密结果
    testPresent(code=0x0000000000000000, key=0x00000000000000000000)

    # Simon算法参数：
    # x和y为传入的明文，各4个byte (16bits)
    # keys为传入的初始秘钥组，一共4个，每个4byte (16bits)，数组低位为二进制/十六进制低位
    # 调用函数后将在标准输出加密结果
    testSimon(x=0x6565, y=0x6877, keys=[0x0100, 0x0908, 0x1110, 0x1918])

    # Speck算法参数：
    # x和y为传入的明文，各4个byte (16bits)
    # keys为传入的初始秘钥组，一共4个，每个4byte (16bits)，数组低位为二进制/十六进制低位
    # alpha和beta为自定参数 (用10进制表示)
    # 调用函数后将在标准输出加密结果
    testSpeck(x=0x6574, y=0x694c, keys=[0x0100, 0x0908, 0x1110, 0x1918], alpha=7, beta=2)
