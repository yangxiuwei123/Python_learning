import os

"""
给每一个图片生成一个label.txt文件，并存储他们的真实label
"""
root_dir = "C:\\Users\\HP\\Desktop\\数据集\\hymenoptera_data\\train\\"
target_dir = "bees_image"
img_path = os.listdir(os.path.join(root_dir, target_dir))
label = target_dir.split('_')[0]
out_dir = "bees_label"

for i in img_path:
    file_name = i.split('.jpg')[0]
    with open(os.path.join(root_dir, out_dir, "{}.txt".format(file_name)), 'w') as f:
        f.write(label)
