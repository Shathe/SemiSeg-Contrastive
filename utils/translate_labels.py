import glob
import cv2
import numpy as np
import tqdm

colors = [[128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
]


labels = glob.glob('../data/GTA5/labels/*/*.png')
assert len(labels) > 0, "Labels not found in ../data/GTA5/labels/*/*.png"

for f in tqdm.tqdm(labels):
        image = cv2.imread(f)
        results = np.ones_like(image[:,:,0]) * 250

        for i in range(len(colors)):
                color_i = colors[i]
                class_i_image1 = image[:,:,0]==color_i[2]
                class_i_image2 = image[:,:,1]==color_i[1]
                class_i_image3 = image[:,:,2]==color_i[0]

                class_i_image = class_i_image1 & class_i_image2 & class_i_image3

                results[class_i_image] = i


        cv2.imwrite(f, results)

