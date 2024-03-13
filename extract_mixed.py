import json
import os

#load json file and extract points and label
def get_data(json_file):
    with open(json_file) as f:
        data = json.load(f)
        shapes = data.get('shapes', [])

        #take json file and make txt file with same name
        output_file = json_file.replace('.json', '.txt')


        #flatten coordinates and write coordinates of one bounding box in on row
        with open(output_file, 'w') as txtfile:
            for shape in shapes:
                if shape["label"] == "id_text":
                    points = shape.get('points', [])
                    flattened_points = [round(coord) for pair in points for coord in pair]
                    if len(flattened_points) != 8:
                        print(json_file)
                        continue
                    line = ",".join(map(str, flattened_points))
                    txtfile.write(line + '\n')




def dir_labeling():
    for file in os.listdir('.'):
        if file.endswith(".json"):
            file_path = os.path.join('.', file)
            get_data(file_path)


dir_labeling()