"""
This script is adapted from 
https://github.com/FranxYao/Long-Context-Data-Engineering
"""
import tiktoken
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048,expandable_segments:True"
import sys
DirPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NIHDirPath = os.path.join(DirPath, "metrics", "Needle-in-haystack")
sys.path.append(os.path.join(DirPath, "src"))
sys.path.append(NIHDirPath)

import pdb
import glob
import jieba

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
# from anthropic import Anthropic
import numpy as np
import argparse
from rouge_score import rouge_scorer

import sys
import os
# import tensor_parallel as tp

# from openai import OpenAI
from datetime import datetime, timezone
import time
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# from pyramidkv.monkeypatch import replace_llama, replace_mistral
from pipline import Pipline as InfiniComp

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# 统一 NIH 任务模板（InfiniComp / full 两分支共用，保证打分口径一致）。
# 打分是 rouge1(needle 全句)：只有逐字复诵 needle 原句才能拿满分 10，
# 改写或续写都会拉低 precision/recall。0.5B 小模型对"word for word"这类
# 抽象指令遵循不稳，故内置一个 few-shot 示例做格式锚：示例 Q 同样带
# "mentioned in the provided text" 从句、A 示范丢弃该从句并逐字复诵
# （答案语法严格镜像 needle 的 "numbers ... is:" 格式）。换 needle 时请
# 同步改写示例。
# 注意：{question} 必须用 \n\n 双侧隔离、且不得带 "Question: " 前缀——
# retrieval.pre_process_sentence 靠逐字句匹配定位 question（split_sentence
# 以 \n\n 或 .!? 为切分点），前缀粘连会触发
# "Searching Question Index ... Error!" 断言；示例 Q 用 New York 与真实
# question 区分，不会误匹配。
NIH_TASK_PROMPT = (
    "There is an important info hidden inside a lot of irrelevant text. "
    "Find it and memorize it. I will quiz you about the important information.\n\n"
    "{context}\n\n"
    "Answer the quiz question with one complete sentence copied word for word "
    "from the text. Do not repeat the question. Do not add any extra words.\n\n"
    "Example:\n"
    "Question: What is the special magic number for New York mentioned in the "
    "provided text?\n"
    "Answer: The special magic numbers for New York is: 12345.\n\n"
    "{question}\n\n"
    "Answer:"
)

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 #needle="\n\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n", 
                 #retrieval_question = "What is the best thing to do in San Francisco ?",
                 needle="\n\nThe special magic numbers for San Francisco is: 32000.",
                 retrieval_question = "What is the special magic number for San Francisco mentioned in the provided text?",
                 haystack_dir = None, # PaulGrahamEssays
                 results_version = 1,
                 context_lengths_min = None,
                 context_lengths_max = None,
                 context_lengths_num_intervals = 40,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 5,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 model_provider = "Mistral",
                 openai_api_key = None,
                 anthropic_api_key = None,
                 model_name = '',
                 model_name_suffix = None,
                 model_version = None, 
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True, 
                 step=1024, 
                 method="InfiniComp",  #'pyramidkv', 
                 attn_implementation= "eager", #'flash_attention_2',
                 max_capacity_prompt=128,
                 ir_topk=300,
                 ir_phrase_token_num=8,
                 ir_chunk_size=1024):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 0.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        haystack_dir = os.path.join(NIHDirPath, "PualGrahamEssays")
        print("haystck_dir:", haystack_dir)
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        self.step = step
        self.method = method
        self.max_capacity_prompts = max_capacity_prompt
        self.attn_implementation = attn_implementation
        self.ir_topk = ir_topk
        self.ir_phrase_token_num = ir_phrase_token_num
        self.ir_chunk_size = ir_chunk_size
        self.test_num = 0

        self.model_version = model_version
        if(model_name_suffix is not None): self.model_version += "_" + model_name_suffix

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                # self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
                self.context_lengths = np.arange(context_lengths_min, context_lengths_max+1, step=self.step)
        else:
            self.context_lengths = context_lengths
        print("context length:", self.context_lengths)

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        self.model_name = model_name

        if(self.model_provider in ["LLaMA3", "Mistral", "Qwen"]):
            self.enc = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
            # self.enc.add_special_tokens({'pad_token': '[PAD]'})
            print("loading from %s" % model_name)

            # if torch.cuda.device_count()>1:

            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
            from transformers import AutoConfig 

            if self.method == 'full':
                print("Using Full attention")
                # flash_attention_2 未安装的环境（如本机）自动回退 sdpa/eager
                last_err = None
                for attn in ("flash_attention_2", "sdpa", "eager"):
                    try:
                        self.model_to_test = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.bfloat16,
                            attn_implementation=attn,
                            device_map="auto",
                            low_cpu_mem_usage=True,
                            use_cache=False
                            ).eval()
                        print("full branch attn_implementation=%s" % attn)
                        break
                    except Exception as e:
                        last_err = e
                        print("attn_implementation=%s unavailable: %s" % (attn, type(e).__name__))
                else:
                    raise RuntimeError("full branch model load failed") from last_err
            elif self.method == "InfiniComp":
                 print("Using InfiniComp")
                 self.model_to_test=AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                    device_map="auto",
                    # low_cpu_mem_usage=True, 
                    # use_cache=False
                ).eval()

            if self.method in ["pyramidkv", "snapkv","streamingllm", "h2o", "cam"]:
            
                if self.model_provider == 'LLaMA3':
                    pass
                    # replace_llama(self.method.lower())
                elif self.model_provider == 'Mistral':
                    pass
                    # replace_mistral(self.method.lower())
                     
                if self.max_capacity_prompts != -1:
                    max_capacity_prompts = self.max_capacity_prompts
                
                print("Using %s and FlashAttention2" % self.method)
                self.model_to_test=AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                    low_cpu_mem_usage=True, 
                    # use_cache=False
                    ).eval()

                if self.method.lower() == "pyramidkv":
                    window_sizes = 8
                elif self.method.lower() in ["snapkv","streamingllm","h2o","cam"]:
                    window_sizes = 32
                    
                kernel_sizes = 7
                pooling = "maxpool"

                layers = len(self.model_to_test.model.layers)
                # check if window_sizes is a list
                if not isinstance(window_sizes, list):
                    window_sizes = [window_sizes] * layers
                if not isinstance(max_capacity_prompts, list):
                    max_capacity_prompts = [max_capacity_prompts] * layers
                if not isinstance(kernel_sizes, list):
                    kernel_sizes = [kernel_sizes] * layers
                for i in range(layers):
                    self.model_to_test.model.layers[i].self_attn.config.window_size = window_sizes[i]
                    self.model_to_test.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                    self.model_to_test.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                    self.model_to_test.model.layers[i].self_attn.config.pooling = pooling
                    
                
                # self.model_to_test = tp.tensor_parallel(self.model_to_test, sharded=True)

        else:raise ValueError("model_provider must be either 'LLaMA3' or 'Mistral'")

        config = {"compress_budget": self.ir_topk, "phrase_token_num":self.ir_phrase_token_num, "chunk_size": self.ir_chunk_size}
        config["data_format_map"] = {"context": "context", "question": "question"}
        save_path = os.path.join(DirPath, "experiences", "process")
        config["record_retrieval_path"] = save_path
        self.ir = InfiniComp(config=config, model=self.model_to_test, tokenizer=self.enc)
            

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self, args):

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len: continue
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(context_length, depth_percent, args.result_dir)

    def generate_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure
        if(self.model_provider not in ["OpenAI", "Anthropic"]):
            test_format=f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
            return test_format
        else: 
            return [
                {
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                },
                {
                    "role": "user",
                    "content": context
                    },
                {
                    "role": "user",
                    "content": f"{self.retrieval_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer, and I'm 100% sure. So try your best to find it."
                },
                {
                    "role": "assistant",
                    "content":"",
                },
                
            ]

    def evaluate_and_log(self, context_length, depth_percent, result_dir):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent, result_dir):
                print("result exists, skipping")
                return
            else:
                print("result does not exist, testing")

        context = self.generate_context(context_length, depth_percent)

        if self.method == "InfiniComp":
            test_start_time = time.time()
            data_dict = {"context": context,
                         "question": self.retrieval_question}

            pred, info = self.ir.generate(
                data_dict=data_dict,
                gen_name="%s_len_%d_depth_%d" % (self.model_version, context_length, depth_percent*100),
                max_new_tokens=64,
                prompt=NIH_TASK_PROMPT,
                temperature=0.1,
                task_name="nih"
            )

            # 只保留首个非空行：答案按设计是单句；nih 不在 PLAIN_TEXT_TASKS
            # 中、generate 无换行停止符，啰嗦续写会稀释 rouge1 precision。
            response = (pred or "").strip()
            for line in response.splitlines():
                if line.strip():
                    response = line.strip()
                    break
        else:
            # FullKV：与 InfiniComp 共用同一任务模板（口径一致），并正确套用
            # chat 模板（原裸 prompt 对 instruct 模型是 OOD 输入）。
            prompt_text = NIH_TASK_PROMPT.format(
                context=context, question=self.retrieval_question)
            test_start_time = time.time()

            if(self.model_provider in ["LLaMA3", "Mistral", "Qwen"]):
                messages = [{"role": "user", "content": prompt_text}]
                chat_enc = self.enc.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt")
                # transformers 5.x 返回 BatchEncoding，4.x 返回 tensor，两者兼容
                input_ids = (chat_enc["input_ids"] if hasattr(chat_enc, "keys")
                             else chat_enc).to(self.model_to_test.device)

                output_ids = self.model_to_test.generate(
                    input_ids,
                    output_attentions=False,
                    max_new_tokens=48,
                    num_beams=1,
                    do_sample=False,
                    eos_token_id=[self.enc.eos_token_id, self.enc.encode("\n", add_special_tokens=False)[-1]]
                )
                response = self.enc.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            
        print(response)
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        if len(response) != 0:
            score = scorer.score(self.needle, response)['rouge1'].fmeasure*10
        else:
            score = 0.0

        results = {
            'model' : self.model_name,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'), 
        }

        if self.method == "InfiniComp":
            results["input_text"] = info["final_input_text"]

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

  
        if self.save_contexts:
            results['file_name'] = context_file_location

            # Save the context to file for retesting
            path = os.path.join(result_dir, "results_needle", "contexts")
            if not os.path.exists(path):
                os.makedirs(path)
            
            path =  os.path.join(result_dir, "results_needle", "contexts", self.model_version)
            if not os.path.exists(path):
                os.makedirs(path)
            
            path = os.path.join(path, f'{context_file_location}_context.txt')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(context)
            
        if self.save_results:
            # Save the context to file for retesting
            path = os.path.join(result_dir, "results_needle", "results")
            if not os.path.exists(path):
                os.makedirs(path)
            
            path = os.path.join(path, self.model_version)
            if not os.path.exists(path):
                os.makedirs(path)

            # Save the result to file for retesting
            path = os.path.join(path, f'{context_file_location}_results.json')
            print("Writing at %s" % path)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False)

    def result_exists(self, context_length, depth_percent, result_dir):
        """
        Checks to see if a result has already been evaluated or not
        """

        # results_dir = os.path.join(DirPath, "src", "results_needle", "results", self.model_version)
        # results_dir = os.path.join(DirPath, "experiences", "results_needle", "results", self.model_version)
        results_dir = os.path.join(result_dir, "results_needle", "results", self.model_version)
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r', encoding='utf-8') as f:
                    try:
                        result = json.load(f)
                    except UnicodeDecodeError:
                        # legacy files may have been written with the system default (GBK)
                        with open(os.path.join(results_dir, filename), 'r', encoding='gbk') as f2:
                            result = json.load(f2)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider in ["Mistral", "LLaMA3", "Qwen"]:
            return self.enc.encode(text, add_special_tokens=False)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            return self.enc.encode(text)
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            print("insertion_potin:", insertion_point)

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            if(self.model_provider in ["LLaMA", "LongLLaMA"]): period_tokens = [29889, 869]
            elif(self.model_provider == "LLaMA3"): period_tokens = [13]
            elif(self.model_provider == "Mistral"): period_tokens = [842, 28723]
            elif(self.model_provider == "GLM"): period_tokens = [918, 30930]
            else: period_tokens = self.encode_text_to_tokens('.')
            
            print("period_tokens:", period_tokens)
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        if self.model_provider in ["Mistral", "LLaMA3", "Qwen"]:
            return len(self.enc.encode(context, add_special_tokens=False))
        else:
            return len(self.enc.encode(context))
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r', encoding="utf-8") as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider in ["Mistral", "LLaMA3", "Qwen"]:
            return self.enc.encode(context, add_special_tokens=False)
        else:
            return self.enc.encode(context)
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider in ["Mistral", "LLaMA3", "Qwen"]:
            return self.enc.decode(tokens[:context_length], skip_special_tokens=True)
        else:
            return self.enc.decode(tokens[:context_length])
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        #asyncio.run(self.run_test())
        self.run_test(args)


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', default=2*1024*1024, type=int, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', default=2*1024*1024, type=int, help='a number')
    parser.add_argument('--model_name', type=str, default="Qwen", help='name of model')
    parser.add_argument("--attn_implementation", type=str,  default="eager", choices=["flash_attention_2", "sdpa", "None", "eager"])
    parser.add_argument('--model_version', type=str, default="Qwen3-0.6B", help='provider of model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="Qwen", help='which model to use')
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--step', type=int, default=1024*32*16)
    parser.add_argument('--method', type=str, default="InfiniComp", choices=['InfiniComp', 'full', 'pyramidkv', 'snapkv', 'streamingllm', 'h2o', 'cam'])
    parser.add_argument('--max_capacity_prompt', type=int, default=128)
    parser.add_argument('--compress_budget', type=int, default=300)
    parser.add_argument("--phrase_token_num", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument("--results_version", type=int, default=1)
    args = parser.parse_args()

    # 模型路径解析：支持别名 / 目录名 / 完整路径，本机与服务器双候选根目录
    MODEL_ROOTS = [r"D:\Code\models", "/data/coding/models"]
    MODEL_ALIAS = {
        "Qwen": "Qwen2.5-1.5B-Instruct",
        # "Qwen": "Qwen2.5-0.5B-Instruct",
        "Llama": "Llama3-8B-Instruct",
        "Mistral": "Mistral-7B-Instruct-v0.2",
    }
    PROVIDER_BY_KEYWORD = [("Llama", "LLaMA3"), ("Mistral", "Mistral"), ("Qwen", "Qwen")]

    provider = None
    for kw, prov in PROVIDER_BY_KEYWORD:
        if kw in args.model_name:
            provider = prov
            break
    if provider is None:
        raise ValueError("unknown model provider for model_name=%s" % args.model_name)
    args.model_provider = provider

    if not os.path.isdir(args.model_name):
        resolved = None
        for root in MODEL_ROOTS:
            cand = os.path.join(root, MODEL_ALIAS.get(args.model_name, args.model_name))
            if os.path.isdir(cand):
                resolved = cand
                break
        if resolved is None:
            raise ValueError("model not found: %s (roots: %s)" % (args.model_name, MODEL_ROOTS))
        args.model_name = resolved
    args.model_version = os.path.basename(os.path.normpath(args.model_name))

    if not args.result_dir:
        args.result_dir = os.path.join(DirPath, "experiences")
    ht = LLMNeedleHaystackTester(model_name=args.model_name,
                                 model_name_suffix=args.model_name_suffix,
                                 model_provider=args.model_provider,
                                 model_version=args.model_version,
                                 results_version=args.results_version,
                                 context_lengths_min= args.s_len,
                                 context_lengths_max= args.e_len,
                                 save_contexts=False,
                                 save_results=True,
                                 openai_api_key=args.api_key,
                                 step= args.step,
                                 document_depth_percent_intervals=args.interval,
                                 method=args.method,
                                 max_capacity_prompt=args.max_capacity_prompt,
                                 attn_implementation=args.attn_implementation,
                                 ir_topk=args.compress_budget,
                                 ir_phrase_token_num=args.phrase_token_num,
                                 ir_chunk_size=args.chunk_size)

    ht.start_test(args)
    os.system('"%s" visual.py --result %s' % (sys.executable, args.model_version))