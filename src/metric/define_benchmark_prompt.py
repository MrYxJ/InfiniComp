#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
'''
@File    :   define_benchmark_prompt.py
@Time    :   2025/01/12 14:15:37
@Author  :   Anonymous (double-blind review)
@Version :   0.1
@Contact:    Anonymous (double-blind review)
@Desc    :   None
'''

import re

def define_infinitebench(func):
    def wrapper(*args, **kwargs):
        eg = kwargs["data_dict"]
        template = kwargs["prompt"]
        data_name = eg["task_name"]

        # The default ``Retrieval.build_prompt`` strips every text field in
        # place. ``Retrieval.pre_process_sentence`` locates the context span
        # with ``context.strip() + "\n\n"``, so the built prompt must contain
        # the *stripped* context; otherwise a trailing "\n" kept by the
        # chunker makes the token accounting (and hence the index search) fail.
        for key, value in eg.items():
            if isinstance(value, str):
                eg[key] = value.strip()

        if data_name == "code_run":
            find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", eg['input'])
            func_call = find_result[0]
            func = func_call.split("(")[0]
            return template.format(
                func=func,
                func_call=func_call,
                context=eg["context"],
            )
        elif data_name in ["code_debug", "code_debug_qa"]:
            code = eg["context"]
            if data_name == "code_debug":
                return template.format(
                    context=code,
                    OPTION_A=eg["options"][0],
                    OPTION_B=eg["options"][1],
                    OPTION_C=eg["options"][2],
                    OPTION_D=eg["options"][3],
                )
            return template.format(context=code)
        elif data_name == "longdialogue_qa_eng":
            script = eg["context"]
            prompt = template.format(context=script)
            return prompt
        elif data_name in [
            "longbook_choice_eng",
            "longbook_qa_eng",
            "longbook_sum_eng",
            "longbook_qa_chn",
        ]:
            book = eg["context"]
            if data_name == "longbook_choice_eng":
                return template.format(
                    question=eg["input"],
                    context=book,
                    OPTION_A=eg["options"][0],
                    OPTION_B=eg["options"][1],
                    OPTION_C=eg["options"][2],
                    OPTION_D=eg["options"][3],
                )
            elif data_name == "longbook_qa_eng":
                return template.format(
                    question=eg["input"].strip(),
                    context=book.strip(),
                )
            elif data_name == "longbook_sum_eng":
                return template.format(context=book)
            elif data_name == "longbook_qa_chn":
                return template.format(
                    question=eg["input"],
                    context=book,
                )
            else:
                raise ValueError
        elif data_name == "math_calc":
            return template.format(context=eg["context"])
        elif data_name == "math_find":
            prompt = eg['input']
            context = eg['context']
            find_result = re.findall(r"The .+ of", prompt)
            assert find_result, f"Cannot find the target number in {prompt}"
            target_number = find_result[0].lower()[:-3]
            prefix = f"What is {target_number} in the following list?"
            return template.format(
                prefix=prefix,
                context=context,
                input=prompt,
            )

        # Default behavior if content key exists
        if "content" in eg:
            content = eg["content"]
            del eg["content"]
            eg["context"] = content

        format_dict = {
            "context": eg["context"].strip(),
            "input": eg["input"].strip(),
        }
        prompt = template.format(**format_dict)
        return prompt
    return wrapper


def define_longbench(func):
    def wrapper(*args, **kwargs):
        data = kwargs["data_dict"]
        template = kwargs["prompt"]

        # 在context, question 与 prompt部分的边界根据实际情况补充"\n\n"
        pattern = r"\n*{context}\n*" 
        template = re.sub(pattern, "{context}", template)
        
        if "{input}" in template:
            pattern = r"\n*{input}*\n"
            template = re.sub(pattern, "{input}", template)
        
        context_pos = template.find("{context}")
        input_pos = template.find("{input}")
        # context, input 位置是相连，只需要两个"\n\n"隔开即可
        if input_pos - context_pos == len("{context}"):
            data["context"] = "\n\n"+data["context"].strip()
            data["input"] = "\n\n"+data["input"].strip()+"\n\n"
        else:
            data["context"] = "\n\n"+data["context"].strip()+"\n\n"
            data["input"] = "\n\n"+data["input"].strip()+"\n\n"
        
        prompt = template.format(**data)
        return prompt
    return wrapper


def define_longbenchv2(func):
    def wrapper(*args, **kwargs):
        data = kwargs["data_dict"]
        task_name = data["task_name"]
        template = kwargs["prompt"]
        if task_name == "cot_ans":
            prompt = template.replace("$DOC$", data["context"].strip()).\
                replace("$Q$", data["question"].strip()).replace("$C_A$", data["choice_A"].strip()).\
                replace('$C_B$', data['choice_B'].strip()).replace('$C_C$', data["choice_C"].strip()).\
                replace('$C_D$', data['choice_D'].strip()).replace('$COT$', data["response_cot"])
        else:
            prompt = template.replace("$DOC$", data["context"].strip()).\
                replace('$Q$', data["question"].strip()).replace('$C_A$', data["choice_A"].strip()).\
                replace('$C_B$', data["choice_B"].strip()).replace('$C_C$', data["choice_C"].strip()).\
                replace('$C_D$', data["choice_D"].strip())
        return prompt
     
    return wrapper

