import json
import os

# path = '/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/memory_evolution/generated_tasks'
# all_files = os.listdir(path)
# all_files = [file for file in all_files if 'json' in file]
# all_tasks = {}
# for file in all_files:
#     category = file.split('.')[0].split('_')[1]
#     if category not in all_tasks:
#         all_tasks[category] = []
#     with open(f'/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/memory_evolution/generated_tasks/{file}', 'r') as f:
#         tasks = json.load(f)
#         all_tasks[category].extend(tasks)
# print(len(all_tasks))

target_path = '/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/memory_evolution/generated_tasks_combined'
all_files = os.listdir(target_path)
all_files = [file for file in all_files if 'json' in file]
for file in all_files:
    with open(f'{target_path}/{file}', 'r') as f:
        tasks = json.load(f)
    print(file, len(tasks))
    new_file_name = file.split('.')[0] + '_V1.json'
    os.rename(f'{target_path}/{file}', f'{target_path}/{new_file_name}')
