import argparse
import os
FileDirPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import glob


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=str)
    parser.add_argument("--max_length", type=int, default=1024*1024*2)
    parser.add_argument("--model_name", type=str, default="")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    PRETRAINED_LEN = int(args.max_length)
    # FOLDER_PATH = args.result
    FOLDER_PATH = os.path.join(FileDirPath, "experiences", "results_needle", "results", args.result+"\\")
    print("FOLDER_PATH:", FOLDER_PATH)
    # Path to the directory containing JSON results
    folder_path = FOLDER_PATH
    if "/" in folder_path:
        model_name = folder_path.split("/")[-2]
    elif "\\" in folder_path:
        model_name = folder_path.split("\\")[-2]
    else: model_name = MODEL_NAME
    MODEL_NAME = args.model_name
    if MODEL_NAME == "":
        if "Qwen" in model_name:
            MODEL_NAME = "Qwen2.5-0.5B-Instruct"
        elif "Mistral" in model_name:
            MODEL_NAME = "Mistral-7B-Instruct-v0.2"
        elif "Llama" in model_name:
            MODEL_NAME = "Llama3-8B-Instruct"
    
    print("folder_path", folder_path)
    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}*.json")

    # import ipdb; ipdb.set_trace()

    # List to hold the data
    data = []
    actual_max_context_length = 0
    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, 'r', encoding="utf-8") as f:
            json_data = json.load(f)
            # Extracting the required fields
            
            try:
                document_depth = json_data.get("depth_percent", None)
                context_length = json_data.get("context_length", None)
            except:
                import pdb
                pdb.set_trace()
            # score = json_data.get("score", None)
            model_input = json_data.get("input_text", None)
            model_response = json_data.get("model_response", None).lower()
            needle = json_data.get("needle", None).lower()
            #expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
            # score = len(set(model_response.split()).intersection(set(expected_answer))) / (len(set(expected_answer))-1)
            # Appending to the list
            # expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day"
            expected_answer = "32000"
            if expected_answer in model_response:
                score = 1.0
            else:
                score = 0.0
            if PRETRAINED_LEN == 0 or context_length <= PRETRAINED_LEN:
                data.append({
                    "Document Depth": document_depth,
                    "Context Length": context_length,
                    "Score": score
                })
                actual_max_context_length = max(actual_max_context_length, context_length)
        
        if score == 0.0:
            os.remove(file)
            print("file:%s  remove" % file)

    PRETRAINED_LEN = actual_max_context_length if PRETRAINED_LEN == 0 else PRETRAINED_LEN
    # Creating a DataFrame
    df = pd.DataFrame(data)
     
    locations = list(df["Context Length"].unique())
    locations.sort()
    for li, l in enumerate(locations):
        if(l > PRETRAINED_LEN): break
    pretrained_len = li
    
    avg_score = df["Score"].mean()
    print(df.head())
    print("Overall score %.3f" % avg_score)

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(32, 10))  # Can adjust these dimensions as needed
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, vmax=1,
        cbar=False,
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=1.0,  # Adjust the thickness of the grid lines here
        linecolor='grey',  # Set the color of the grid lines
        linestyle='--'
    )


    # More aesthetics
    model_name_ = MODEL_NAME

    # plt.title(f'Pressure Testing {model_name_} \nFact Retrieval Across Context Lengths ("Needle In A HayStack")', fontsize=18)  # Adds a title

    # plt.title(f'Pressure Testing {model_name_} \nFact Retrieval Across Context Lengths ("Needle In A HayStack")\nOverall score: %.3f' % avg_score, fontsize=18)  # 

    # plt.title(f'Fact Retrieval Across Context Lengths ("Needle In A HayStack")\nQwen2-0.5B-Instruct(Using InfiniComp) In 1M\nOverall score: %.3f' % avg_score, fontsize=28)  # 
    plt.title(f'\n Overall score: %.3f' % avg_score, fontsize=28)  # 
    plt.xlabel('Token Limit', fontsize=28)  # X-axis label
    plt.ylabel('Depth Percent', fontsize=28)  # Y-axis label
    plt.xticks(rotation=45, fontsize=20)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0, fontsize=20)  # Ensures the y-axis labels are horizontal
    # plt.tick_params(axis='both', which='both', length=0)
    # plt.subplots_adjust(left=0.5)

    plt.tight_layout()  # Fits everything neatly into the figure area
    # 调整子图布局

    # Add a vertical line at the desired column index
    # plt.axvline(x=pretrained_len + 0.8, color='white', linestyle='--', linewidth=4)

    # save_path = "./results_needle/img/%s_%.3f.png" % (model_name, avg_score)
    save_path = os.path.join(FileDirPath, "experiences", "results_needle", "img", "%s_%.3f.png" % (model_name, avg_score))
    print("saving at %s" % save_path)
    plt.savefig(save_path, dpi=198)


if __name__ == "__main__":
    main()