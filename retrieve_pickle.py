import pickle
import json

# Open pickle data
with open('data/sewoon_east_transform_mat.pickle', 'rb') as f:
    data = pickle.load(f)

# Package global shift and rotation matrix into an object
transformation_dict = data['global_shift']
for i in range(0, 3):
    for j in range(0, 3):
        transformation_dict['r%d%d' % (i + 1, j + 1)] = data['rotation_matrix'][i][j]

# Serialize to JSON string
# data_json = json.dumps(rotation_matrix_dict)
with open('data/sewoon_east_transform_mat.json', 'w') as f:
    json.dump(transformation_dict, f)
