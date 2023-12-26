import cv2

def dhash(image: cv2.Mat):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # resize to 9*8
    resized_image = cv2.resize(gray, (9, 8))
    print(resized_image)
    # calculate hash
    h = 0
    for i in range(8):
        byte = 0
        for j in range(8):
            byte <<= 1
            if resized_image[i, j] > resized_image[i, j + 1]:
                byte = byte | 1
        h = h << 8
        h = h | byte
    return h

def dhash_distance(a: int, b: int):
    bits = a ^ b
    cnt = 0
    while bits:
        if bits & 1:
            cnt += 1
        bits >>= 1
    return cnt
