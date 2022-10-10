import uuid
import random
import json

model_type_list = ['best', 'rl', 'rl-joint', 'switch']

# result = {
#     'test': {
#         'id_to_model_type': ['rl-joint', 'best', 'rl', 'switch'],
#         'match_seeds': [[0, 1], [1, 2], [1919, 810], [2, 114514]],
#     },
# }
with open('users_config.json', 'r') as f:
    result = json.load(f)

with open('id_list.csv', 'a') as f:
    for _ in range(20):
        username = uuid.uuid4().hex
        tmp = model_type_list.copy()
        random.shuffle(tmp)
        result[username] = {
            'id_to_model_type': tmp,
            'match_seeds': [
                [random.randint(0, 65535) for j in range(2)]
                for i in range(len(model_type_list))
            ],
        }
        f.write(username + '\n')

with open('users_config.json', 'w') as f:
    json.dump(result, f, indent=4)