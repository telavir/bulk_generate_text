#!/usr/bin/env python
# coding: utf-8

# Copyright 2023 Evan Davis

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
import argparse
import logging
import torch

def main(): # the main function called when run in command line.
    if torch.cuda.is_available(): # use pytorch to set device
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    logging.basicConfig( # Logging generates the command line progress info.
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser() # initialize Argument Parser
    # COLLECT ARGUMENTS - Also sets help info. -h or --help
    parser.add_argument("--model_name",type=str,default="bigscience/bloom-560m",help="Full name of the model. Default: bigscience/bloom-560m")
    parser.add_argument("--hf_key", type=str,default="",help="The API token that allows you to access the to HuggingFace Models")
    parser.add_argument("--seed",type=int, default=91375,help="The seed for random numbers.")
    parser.add_argument("--prompt",type=str,help="The prompt you want to generate a blog with surrounded by \"\"")
    parser.add_argument("--min_tokens",type=int,default=350,help="Minimum number of words to generate. Default: 350")
    parser.add_argument("--max_tokens",type=int,default=450,help="Maximum number of words to generate. Default: 450")
    parser.add_argument("--penalty_alpha",type=int,default=0.6,help="Penalty Alpha for generating text. Default: 0.6")
    parser.add_argument("--top_k",type=int,default=8,help="Total number of words considered, chosen by highest probibility. Default: 8")
    parser.add_argument("--temperature",type=int,default=0.8,help="Reduces confidence requirements: 1=whatever, 0=only if you're sure, 0.8=Default")
    parser.add_argument("--top_p",type=int,default=0.8,help="The total probability of the smallest group of words you want to consider. Default: 0.8")
    parser.add_argument("--repetition_penalty",type=int,default=1.1,help="Penalizes repetion: 1=no penalty, >1=penalty, 1.1=Default")
    parser.add_argument("--prompt_file",type=str, default=None,help="A csv file with one column labeled: prompt")
    parser.add_argument("--model_save_path",type=str,default=None,help="If you want to save the model locally before use. Default: <blank>")
    args = parser.parse_args() # Pass arguments to args
    # Pass arguments to variables.
    checkpoint = args.model_name
    set_seed(args.seed)
    min_tokens = args.min_tokens
    max_tokens = args.max_tokens
    penalty_alpha = args.penalty_alpha
    top_k = args.top_k
    temperature = args.temperature
    top_p = args.top_p
    repetition_penalty = args.repetition_penalty
    model_save_path = args.model_save_path
    # GET MISSING PARAMETERS
    if args.prompt_file:
        prompt_file = args.prompt_file
        prompt_type = "Prompts: From CSV"
    elif args.prompt:
        prompt = args.prompt
        prompt_type = f"Prompt: {prompt}"
    else:
        prompt = input("Generation Prompt >>> ")
        prompt_type = f"Prompt: {prompt}"
    # Update the user
    print(
        f"PARAMETERS SET\n"
        f"Model Name: {checkpoint}\n"
        f"{prompt_type}\n"
        f"HuggingFace ID: *****\n"
    )
    if args.hf_key: login(token=hf_token) # login to the HF api
    tokenizer = AutoTokenizer.from_pretrained(checkpoint) # initialize tokenizer and model
    print("Tokenizer Loaded\n")
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device) # initialize model
    print("Model Loaded\n")
    generation_config = GenerationConfig(
        min_new_tokens = min_tokens, # the minimum length to generate 
        max_new_tokens = max_tokens, # The maximum length of new tokens to generate.
        penalty_alpha=penalty_alpha, # Degeneration penalty for contrastive search
        do_sample=True,
        top_k=top_k, # total number of words considered, chosen by highest probibility
        temperature=temperature, # reduces confidence requirements, 1=whatever, 0=only if you're sure
        top_p=top_p, # total probability of the smallest group of words you want it to consider
        repetition_penalty=repetition_penalty, # reduces repetition, 1=no penalty, >1=penalty
    )
    print(
        f"GENERATION CONFIG\n"
        f"Min New Tokens: {min_tokens}\n"
        f"Max New Tokens: {max_tokens}\n"
        f"Penalty Alpha: {penalty_alpha}\n"
        f"Top K: {top_k}\n"
        f"Temperature: {temperature}\n"
        f"Top P: {top_p}\n"
        f"Repetiton Penalty: {repetition_penalty}\n"
    )
    def single_generation(prompt):
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=16).input_ids.to(device)
        print("Prompt Tokenized\n\nGENERATING...\n\n")
        output = model.generate(input_ids, generation_config=generation_config)
        generated_tokens = output[0].tolist()
        prompt_length = len(tokenizer(prompt)["input_ids"])
        text_output = tokenizer.decode(generated_tokens[prompt_length:], skip_special_tokens=True)
        print(f"Generated text:\n{text_output}")
    def multi_generate(examples):
        outputs = [] # initialize list
        counter = 0 # Count it out for the people.
        for title in examples['prompt']: # iterate through the column Title
            input_ids = tokenizer(title, return_tensors="pt", truncation=True, padding="max_length", max_length=16).input_ids.to(device)
            output = model.generate(input_ids, generation_config=generation_config)
            generated_tokens = output[0].tolist() # Create a list of tokens
            prompt_length = len(tokenizer(title)["input_ids"]) # How many tokens was the prompt
            # Remove the prompt and convert tokens to words.
            text_output = tokenizer.decode(generated_tokens[prompt_length:], skip_special_tokens=True)
            outputs.append(text_output) # append list of outputs
            counter += 1
            print(f"Generating from Prompt {counter}\n")
        return {'output': outputs} # Return a dictionary/column of the outputs

    if args.prompt_file:
        dataset = load_dataset("csv", data_files=prompt_file) # initialize dataset
        print("Dataset Loaded\n\nGENERATING...\n\n")
        dataset = dataset.map(multi_generate, batched=True)
        print("Generation complete.\nUpdating CSV file.\n")
        dataset['train'].to_csv(prompt_file)
        print("CSV File Updated\n")
    else:
        single_generation(prompt)
    if args.model_save_path:
        print(f"Saving {checkpoint} to {model_save_path}\n")
        model.save_pretrained(args.model_save_path)
        tokenizer.save_pretrained(args.model_save_path)
        print(f"{checkpoint} saved.\n\nRun Complete")
    else:
        print("Run complete")

if __name__ == "__main__":
    main()

