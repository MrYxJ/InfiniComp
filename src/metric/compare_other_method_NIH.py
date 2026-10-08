import os

# compare_methods = ["pyramidkv", "snapkv", "streamingllm", "h2o"]
compare_methods = ["h2o"]
model_name = "Llama"


for method in compare_methods:
    model_version = "Llama3-%s-1k-32k" % method
    run_python = "python run_needle_in_haystack.py --model_name %s --model_version %s -s 1024 -e 32768 --step 1024 --method %s" % (model_name, model_version, "full")
    print(run_python)
    os.system(run_python)