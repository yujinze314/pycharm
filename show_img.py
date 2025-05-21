import cv2
import matplotlib.pyplot as plt

img_path = r"E:/gated2depth/Gated2Depth-master/data/real/gated0_10bit/00003.png"
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
plt.imshow(img, cmap='gray')
plt.title("gated0_10bit 00003.png")
plt.axis('off')
plt.show()