import os,shutil

folder_path = './data_hero/images'

files_and_folders = os.listdir(folder_path)

file_names = [os.path.splitext(f)[0] for f in files_and_folders if os.path.isfile(os.path.join(folder_path, f))]

total_num = len(file_names)

train_num = int(0.7 * total_num)

valid_num = int(0.2 * total_num)

test_num = total_num - valid_num - train_num

image_dir = 'data_hero/images'
label_dir = 'data_hero/labels'

def cp(source_file,dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    destination_file = os.path.join(dst_folder, os.path.basename(source_file))
    shutil.copy2(source_file,destination_file)

for index in range(train_num):
    cp(os.path.join(image_dir,file_names[index] + '.jpg'),'data_hero/train/images')
    cp(os.path.join(label_dir,file_names[index] + '.txt'),'data_hero/train/labels')


for index in range(train_num,train_num+valid_num):
    cp(os.path.join(image_dir,file_names[index] + '.jpg'),'data_hero/valid/images')
    cp(os.path.join(label_dir,file_names[index] + '.txt'),'data_hero/valid/labels')

for index in range(train_num+valid_num,total_num):
    cp(os.path.join(image_dir,file_names[index] + '.jpg'),'data_hero/test/images')
    cp(os.path.join(label_dir,file_names[index] + '.txt'),'data_hero/test/labels')
