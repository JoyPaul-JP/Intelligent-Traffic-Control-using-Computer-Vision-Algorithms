import os
import cv2
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# dataset directory. Change with your own path
DATASET_DIR = './bd_data'

# list of all XML annotations
annotations = [filename for filename in os.listdir(DATASET_DIR) if filename.endswith(".xml")]

# split into train and test
train_annotations, test_annotations = train_test_split(annotations, test_size=0.25)
# further split test into test and validation
test_annotations, val_annotations = train_test_split(test_annotations, test_size=0.5)

# View data in all splits
print("Training:", len(train_annotations), "Validation:", len(val_annotations),  "Test:", len(test_annotations))

# classes we want to use
classes = {"car" : 0, "bus" : 1, "cng" : 2, "bike" : 3, "truck": 4}
print(classes)

def xml_to_txt(path, dest):
    # read xml file using xml.etree
    print(path, dest)
    tree = ET.parse(path)
    root = tree.getroot()

    # output labels txt file
    file = open(dest, "w")

    # iterate over each annotation in file
    for member in root.findall('object'):
        class_name = member[0].text
        print("class_name ===========> ", class_name)
        if class_name not in classes: continue # if classes is not in defined classes, ignore it
        # image width and height
        image_width = int(root.find('size')[0].text)
        image_height = int(root.find('size')[1].text)
        # bbox coordinates
        xmin = int(member[4][0].text)
        ymin = int(member[4][1].text)
        xmax = int(member[4][2].text)
        ymax = int(member[4][3].text)

        # convert bbox coordinates to yolo format
        center_x = (xmin + (xmax - xmin) / 2) / image_width # Get center (X) of bounding box and normalize
        center_y = (ymin + (ymax - ymin) / 2) / image_height # Get center (X) of bounding box and normalize
        width = (xmax - xmin) / image_width # Get width of bbox and normalize
        height = (ymax - ymin) / image_height # Get height of bbox and normalize
        # write to file
        file.write(f"{classes[class_name]} {center_x} {center_y} {width} {height}\n")
    # close file
    file.close()
    
def process_list(xmls, destination_dir, split_type):
    # open/create images txt file
    destination_dir_file = open(f"{destination_dir}/{split_type}.txt", "w")
    # destination directory for images and txt annotation
    destination_dir = f"{destination_dir}{split_type}"
    os.makedirs(f"{destination_dir}", exist_ok=True) # create path if not exists
    for xml_filename in xmls: # iterate over each xml
        try:
          # convert to txt using function and store in given path
          txt_name = xml_filename.replace('.xml', '.txt')
          xml_to_txt(f"{DATASET_DIR}/{xml_filename}", f"{destination_dir}/{txt_name}")

          # copy image to destination path
          image_name = xml_filename.replace('.xml', '.jpg')
          shutil.copy(f"{DATASET_DIR}/{image_name}", f"{destination_dir}/{image_name}")
          # add image path to txt file
          destination_dir_file.write(f"{destination_dir}/{image_name}\n")
        except:
          continue
    # close file
    destination_dir_file.close()
    
    
processed_path = "./output_txt_file"
process_list(train_annotations, processed_path , "train")
process_list(test_annotations, processed_path , "test")
process_list(val_annotations, processed_path , "val")
