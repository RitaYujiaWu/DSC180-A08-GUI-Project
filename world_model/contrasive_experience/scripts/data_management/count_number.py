import json
import os
successful_path_txt = '/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/count_result/successful_path.txt'
incomplete_failed_path_txt = '/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/count_result/incomplete_failed_path.txt'
admitted_failed_path_txt = '/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/count_result/admitted_failed_path.txt'
other_failed_path_txt = '/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/count_result/other_failed_path.txt'

path = '/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/training_data/expand_memory'
all_subsets = os.listdir(path)
all_subsets = [subset for subset in all_subsets if os.path.isdir(os.path.join(path, subset))]
total_success = 0
total_incomplete = 0
total_admitted = 0
total_other = 0
for subset in all_subsets:
    print(f'Processing {subset}...')
    success_folder = f'{path}/{subset}/qwen2.5-vl-32b/test/success'
    if not os.path.exists(success_folder):
        continue
    success_files = os.listdir(success_folder)
    success_files = [file for file in success_files if 'jsonl' in file]
    for file in success_files:
        try:
            success_file_path = f'{path}/{subset}/qwen2.5-vl-32b/test/success/{file}'
            with open(success_file_path, 'r') as f:
                data = json.load(f)
                round_num = data['total_rounds']
                conversations = data['rounds']
                last_response = conversations[-1]['response']
                negative_phrases = ['cannot', 'not found', 'not available', "can't"]
                if 'stop' not in last_response.lower():
                    total_incomplete += 1
                    with open(incomplete_failed_path_txt, 'a') as f:
                        f.write(f'{success_file_path}\n')
                elif any(phrase in last_response for phrase in negative_phrases):
                    total_admitted += 1
                    with open(admitted_failed_path_txt, 'a') as f:
                        f.write(f'{success_file_path}\n')
                else:
                    total_success += 1
                    with open(successful_path_txt, 'a') as f:
                        f.write(f'{success_file_path}\n')
        except Exception as e:
            print(f'Error loading {file}: {e}')
            raise e
    
    negative_folder = f'{path}/{subset}/qwen2.5-vl-32b/test/negative'
    if not os.path.exists(negative_folder):
        continue
    negative_files = os.listdir(negative_folder)
    negative_files = [file for file in negative_files if 'jsonl' in file]
    for file in negative_files:
        try:
            negative_file_path = f'{path}/{subset}/qwen2.5-vl-32b/test/negative/{file}'
            with open(negative_file_path, 'r') as f:
                neg_data = json.load(f)
                neg_round_num = neg_data['total_rounds']
                neg_conversations = neg_data['rounds']
            positive_file_path = f'{path}/{subset}/qwen2.5-vl-32b/test/positive/{file}'
            with open(positive_file_path, 'r') as f:
                pos_data = json.load(f)
                pos_round_num = pos_data['total_rounds']
                pos_conversations = pos_data['rounds']
            total_conversations = pos_conversations + neg_conversations
            last_response = total_conversations[-1]['response']
            negative_phrases = ['cannot', 'not found', 'not available', "can't"]
            if 'stop' not in last_response.lower():
                total_incomplete += 1
                with open(incomplete_failed_path_txt, 'a') as f:
                    f.write(f'{negative_file_path}\n')
            elif any(phrase in last_response for phrase in negative_phrases):
                total_admitted += 1
                with open(admitted_failed_path_txt, 'a') as f:
                    f.write(f'{negative_file_path}\n')
            else:
                total_other += 1
                with open(other_failed_path_txt, 'a') as f:
                    f.write(f'{negative_file_path}\n')
        except Exception as e:
            print(f'Error loading {negative_file_path}: {e}')
            raise e

print(f'Total success: {total_success}')
print(f'Total incomplete: {total_incomplete}')
print(f'Total admitted: {total_admitted}')
print(f'Total other: {total_other}')