import yaml
import os
import cv2

CLASSES = ['Ageratum_houstonianum',
'Ageratum_houstonianum_flower',
'Ageratum_houstonianum_leaf',
'Amaranthus_spinosus',
'Amaranthus_spinosus_flower',
'Amaranthus_spinosus_leaf',
'Bidens_pilosa_var_radiata',
'Bidens_pilosa_var_radiata_flower',
'Bidens_pilosa_var_radiata_leaf',
'Bidens_pilosa_var_radiata_seed',
'Celosia_argentea',
'Celosia_cristata_flower',
'Celosia_cristata_leaf',
'Chloris_barbata',
'Chloris_barbata_flower',
'Crassocephalum_crepidioides',
'Crassocephalum_crepidioides_flower',
'Crassocephalum_crepidioides_leaf',
'Eleusine_indica',
'Eleusine_indica_flower',
'Lantana_camara',
'Lantana_camara_flower',
'Lantana_camara_leaf',
'Leucaena_leucocephala',
'Leucaena_leucocephala_flower',
'Leucaena_leucocephala_leaf',
'Leucaena_leucocephala_seed',
'Mikania_micrantha',
'Mikania_micrantha_flower',
'Mikania_micrantha_leaf',
'Miscanthus_species',
'Miscanthus_species_flower',
'Pennisetum_purpureum',
'Pennisetum_purpureum_flower',
'Syngonium_podophyllum',
'Tithonia_diversifolia',
'Tithonia_diversifolia_flower',
'Tithonia_diversifolia_leaf']

def get_names_in_tempfile(datafile): #return classes appeared in current folder
    with open(datafile, 'r') as stream:
        try:
            d=yaml.safe_load(stream)
            return d['names']
        except yaml.YAMLError as e:
            print(e)

def read_labels_from_file(label_file):
    with open(label_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return list(map(lambda x: x.split(' '),lines))

def convert_labels(labels, temp_classes):
    correct_labels = []
    for label in labels:
        temp_class = temp_classes[int(label[0])]
        correct_class = CLASSES.index(temp_class)
        correct_label = [str(correct_class), *label[1:]]
        correct_labels.append(' '.join(correct_label))

    return correct_labels

def write_labels_to_file(labels, label_file):
    labels = '\n'.join(labels)
    with open(label_file, 'w') as f:
       f.write(labels)
    return 0


def rewrite_labels_in_dataset(parent_folder, data_folder, dest_folder):

    # if dest_folder == None:
    #     dest_folder = data_folder
    
    roboflow_folder = os.path.join(parent_folder, data_folder)
    
    temp_classes = get_names_in_tempfile(os.path.join(roboflow_folder, 'data.yaml'))
    sets = ['train', 'valid', 'test']
    for set in sets:
        print(data_folder, set)
        if os.path.exists(os.path.join(roboflow_folder, set)):
            temp_path = os.path.join(roboflow_folder, set, 'labels')
            dest_path = os.path.join(dest_folder, set, 'labels')
            label_files = os.listdir(temp_path)
            for label_file in label_files:
                temp_label_path = os.path.join(temp_path, label_file)
                dest_label_path = os.path.join(dest_path, label_file)
                labels = read_labels_from_file(temp_label_path)  

                labels = convert_labels(labels, temp_classes)
                write_labels_to_file(labels, dest_label_path)



    return 0

def copy_images_in_dataset(parent_folder, data_folder, dest_folder):

    # if dest_folder == None:
    #     dest_folder = data_folder

    roboflow_folder = os.path.join(parent_folder, data_folder)
    
    sets = ['train', 'valid', 'test']
    for set in sets:
        if os.path.exists(os.path.join(roboflow_folder, set)):
            temp_path = os.path.join(roboflow_folder, set,'images')
            dest_path = os.path.join(dest_folder, set, 'images')
            image_files = os.listdir(temp_path)
            for image_file in image_files:
                temp_image_path = os.path.join(temp_path, image_file)
                dest_image_path = os.path.join(dest_path, image_file)
                img = cv2.imread(temp_image_path) 
                cv2.imwrite(dest_image_path, img)



    return 0

def extract_from_folder(folder, dest_folder):
    if os.path.exists(folder):
        for roboflow_folder in os.listdir(folder):
            rewrite_labels_in_dataset(folder, roboflow_folder, dest_folder)
            copy_images_in_dataset(folder, roboflow_folder, dest_folder)

if __name__ == '__main__':
    extract_from_folder('roboflow_folders', dest_folder='final_dataset')



