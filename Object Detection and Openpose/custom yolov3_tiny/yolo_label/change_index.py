import glob
import os

os.chdir(r'/home/irlcrossing/Desktop/Object detection/yolo_label/dataset')
my_files = glob.glob('*.txt')
print(my_files)
 

for text_file in my_files:
    with open(text_file, "r") as f:
        f_data = f.read()
    # Replace the target string
    f_data = f_data.replace('2', '0')

    # Write the file out again
    with open(text_file, 'w') as file:
        file.write(f_data)