import os
import config
import random
import shutil
dataset = {}
hard_classes = ['headbutting',	'eating nachos',	'closing door',	'tasting food',	'pinching',	'combing hair',	'tickling',	'fixing hair',	'coughing',	'luge',	'photobombing',	'moving child',	'passing American football (not 1n game)',	'answering questions',	'waving hand',	'throwing tantrum',	'being excited',	'winking',	'karaoke',	'looking at phone']

for root, dirs, files in os.walk('/media/Data/Lorenzo/Code/Kinetcs_no_vid/Videos_OUT'):
    if len(root.split('/')) == 9:
        activity = root.split('/')[-2]
        if activity in hard_classes:
            path = [root + '/' + fl for fl in files]
            if activity not in dataset:
                dataset[activity] = []
            dataset[activity] += path
            print(activity, len(path), len(dataset[activity]))
            # for fl in files:
            #     path = root + '/' + fl
            #     file_extension = fl.split('.')[-1]
            #     print(path)
            #     if file_extension == 'jpg':
            #         folder_structure = root.split('/')
            #         activity = folder_structure[-2]
            #         if activity not in dataset:
            #             dataset[activity] = []
            #         if path not in dataset[activity]:
            #             dataset[activity].append(path)


base_path = '/media/Data/Lorenzo/Code/Kinetcs_no_vid/Sampled'
for activity in dataset:
    print(activity)
    path_list = list(dataset[activity])
    for i in range(0,300):
        random_path = random.choice(path_list)
        folder_structure = random_path.split('/')
        file_name = folder_structure[-1]
        segment_name = folder_structure[-2]
        activity_name = folder_structure[-3]
        copy_folder =  base_path + '/' + activity_name + '/' + segment_name 
        final_path = copy_folder + '/' + file_name

        if not os.path.exists(copy_folder):
            os.makedirs(copy_folder)
        print(copy_folder)
        new_path = shutil.copy(random_path, final_path)
