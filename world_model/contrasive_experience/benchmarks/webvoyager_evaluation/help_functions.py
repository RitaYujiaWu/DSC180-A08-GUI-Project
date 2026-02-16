import json
import os
import random

def organize_related_queries(html_folder):
    with open(os.path.join(html_folder, f"results/webvoyager/qwen2.5-vl/test/test/generate_related_queries_template.json"), "r") as f:
        related_queries = json.load(f)
    os.makedirs(os.path.join(html_folder, "related_queries_level1_V2"), exist_ok=True)
    for task_id, items in related_queries.items():
        for idx, item in enumerate(items):
            step = item["step"]
            url = item["url"]
            generated_queries = item["queries"]
            for query_idx, query in enumerate(generated_queries):
                query_id = f"{task_id}-{idx}-{query_idx}"
                query_file = os.path.join(html_folder, "related_queries_level1_V2", f"{query_id}.json")
                with open(query_file, "w", encoding="utf-8") as f:
                    json.dump({"intent": query, "task_id": query_id, "start_url": url, "site": ["generated_level1"]}, f, indent=4)

def organize_atomic_queries(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    new_data = {}
    for id, items in data.items():
        for idx, item in enumerate(items):
            query_id = f"{id}-{idx}"
            if len(item["queries"]) < 3:
                selected_queries = item["queries"]
            else:
                selected_queries = random.sample(item["queries"], 3)
            for query in selected_queries:
                new_data[query_id] = {
                    "task_id": query_id,
                    "start_url": item["url"],
                    "intent": query,
                    "next_state": '',
                    "site": ["generated_level1"]
                }
    with open(os.path.join(os.path.dirname(json_file), "atomic_queries.json"), "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4)
        
if __name__ == "__main__":
    json_file = "results/webvoyager/qwen2.5-vl/test/test/generate_related_queries_template.json"
    organize_atomic_queries(json_file)