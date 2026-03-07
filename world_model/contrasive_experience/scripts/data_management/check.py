import os

datasets = os.listdir('training_data')
datasets = [dataset for dataset in datasets if (not 'conversation' in dataset) and ('expand_memory' in dataset)]
total_positive = 0
total_negative = 0
total_success = 0
for dataset in datasets:
    domains = os.listdir(f'training_data/{dataset}')
    for domain in domains:
        models = os.listdir(f'training_data/{dataset}/{domain}')
        if 'qwen2.5-vl-32b' in models:
            target_path = f'training_data/{dataset}/{domain}/qwen2.5-vl-32b/test'
            positive_folder = f'training_data/{dataset}/{domain}/qwen2.5-vl-32b/test/positive'
            negative_folder = f'training_data/{dataset}/{domain}/qwen2.5-vl-32b/test/negative'
            success_folder = f'training_data/{dataset}/{domain}/qwen2.5-vl-32b/test/success'
            if not os.path.exists(target_path):
                continue
            print('*'*50, dataset, domain, '*'*50)
            print(f'Positive: {len(os.listdir(positive_folder))}')
            print(f'Negative: {len(os.listdir(negative_folder))}')
            print(f'Success: {len(os.listdir(success_folder))}')
            total_positive += len(os.listdir(positive_folder))
            total_negative += len(os.listdir(negative_folder))
            total_success += len(os.listdir(success_folder))
            print('*'*100)
print(f'Total Positive: {total_positive}')
print(f'Total Negative: {total_negative}')
print(f'Total Success: {total_success}')