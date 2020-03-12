# Author: Tao Hu <taohu620@gmail.com>
import glob
import os,cv2

src_base_dir = "/data1/dataset/m1-mar16-55"
target_base_dir = "/data1/dataset/m1-mar16-55-cropped"


src_img_dir = os.path.join(src_base_dir, "src")
src_gt_dir = os.path.join(src_base_dir,"gt")
target_img_dir = os.path.join(target_base_dir, "src")
target_gt_dir = os.path.join(target_base_dir,"gt")


src_img_dir_list = glob.glob(os.path.join(src_img_dir,"*.JPG"))
src_img_dir_list.sort()

src_gt_dir_list = glob.glob(os.path.join(src_gt_dir,"*.png"))
src_gt_dir_list.sort()
border=1000
f = open("train.txt","w")


for img_path,gt_path in zip(src_img_dir_list,src_gt_dir_list):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    gt = cv2.imread(gt_path)

    h, w, _ = img.shape
    assert h==3000, w==4000
    h_grid_num = h / border
    w_grid_num = w / border
    for i in range(h_grid_num):
        for j in range(w_grid_num):
            start_i = border * i
            start_j = border * j
            end_i = border * (i + 1)
            end_j = border * (j + 1)
            cv2.imwrite(os.path.join(target_img_dir, "{}_{}_{}.jpg".format(base_name, i, j)),
                        img[start_i:end_i, start_j:end_j])
            cv2.imwrite(os.path.join(target_gt_dir, "{}_{}_{}.png".format(base_name, i, j)),
                        gt[start_i:end_i, start_j:end_j])

            f.write("{} {}\n".format(os.path.join(target_img_dir, "{}_{}_{}.jpg".format(base_name, i, j))
                                                  ,os.path.join(target_gt_dir, "{}_{}_{}.png".format(base_name, i, j))))


f.close()



