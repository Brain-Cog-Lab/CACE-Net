import h5py
import os

# 文件夹路径
folder_path = '/home/hexiang/Encoders/data/image_opencv_finetune/'

cnt = 0
# 遍历文件夹下的所有文件
for filename in os.listdir(folder_path):
    # 检查文件扩展名是否为.h5
    if filename.endswith('.h5'):
        file_path = os.path.join(folder_path, filename)
        # 打开HDF5文件
        with h5py.File(file_path, 'r') as h5file:
            # 遍历文件中的所有key
            if len(h5file.keys()) == 0:
                print(f"File: {filename} is empty (no keys).")
            for key in h5file.keys():
                if key == "extract_frame":
                    # print(f"File: {filename}, Key: {key}")
                    cnt += 1
                else:
                    print("xx:{}".format(key))

    else:
        print(filename)
print(cnt)
