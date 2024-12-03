#!/usr/bin/python3
import sys
import os
import json
import openai
from datetime import datetime
import time

# Define models and their costs, context lengths, and providers
models = {
    "meta-llama/Meta-Llama-3.1-405B":         {"input_cost": 4.0, "output_cost": 4.0, "context_length": 32768, "max_tokens": 8192, "provider": "hyperbolic"},
    "meta-llama/Meta-Llama-3.1-405B-FP8":     {"input_cost": 2.0, "output_cost": 2.0, "context_length": 32768, "max_tokens": 8192, "provider": "hyperbolic"},
    "Qwen/Qwen2.5-72B-Instruct":              {"input_cost": 0.4, "output_cost": 0.4, "context_length": 16384, "max_tokens": 8192, "provider": "hyperbolic"},
    "meta-llama/Meta-Llama-3.1-70B-Instruct": {"input_cost": 0.4, "output_cost": 0.4, "context_length": 16384, "max_tokens": 8192, "provider": "hyperbolic"},
    "Qwen/Qwen2.5-Coder-32B-Instruct":        {"input_cost": 0.2, "output_cost": 0.2, "context_length": 32768, "max_tokens": 8192, "provider": "hyperbolic"},
    "Qwen/QwQ-32B-Preview":                   {"input_cost": 0.2, "output_cost": 0.2, "context_length": 32768, "max_tokens": 8192, "provider": "hyperbolic"},
    "gpt-4o-mini":                           {"input_cost": 0.15, "output_cost": 0.60, "context_length": 128000, "max_tokens": 16384, "provider": "openai"},
}

# Select a default model
default_model = 'Qwen/Qwen2.5-Coder-32B-Instruct'

continuation_message = "<!--generation interrupted, continuing-->"  # this needs to be defined before the system message. THIS SHOULD NOT CHANGE
# The system prompt is sent out-of-band to instruct the model how to act.  THIS SHOULD NOT CHANGE WITHOUT CONSULTING THE LEAD DEVELOPER
system_content = f"""You are an expert assistant with creative freedom. You change code as requested and proactively fix any problems you notice. You may see {continuation_message} messages which you should ignore except for when the prompt ends with one, then you should continue the previous answer exactly from the point of interruption. The reason is that your answers are limited in the number of tokens, but your context and reasoning are not. In order to get a complete answer we may have to interrupt you a number of times, so your willingness to seamlessly continue after such message is critical to the success of this mission. I cannot stress enough: if the prompt ends with the continuation message {continuation_message}, you must continue exactly where the previous invocation left off with no additional verbiage or repeating of previous characters/tokens. 

[Example]
USER: Make the snake game in Python.

ASSISTANT: Creating a simple Snake game in Python can be a fun and educational project. We'll use the pygame library, which is a set of Python modules designed for writing video games. If you don't have pygame installed, you can install it using pip:

pip install pygame

Here's a step-by-step g{continuation_message}uide to creating a basic Snake game:

    Import the necessary modules:

import pygame
...
[End Example]

You got this. Thanks!
"""

def print_help():
    print("Usage: python autocoder.py --ctl <control_file> [--force] [--mode <model>]")
    print("       python autocoder.py --in <input_file> --out <output_file> --req <requirements> [--force] [--mode <model>]")
    print()
    print("Options:")
    print("  --ctl <control_file>  Specify a control file in JSON format.")
    print("  --in <input_file>     Specify the input file.")
    print("  --out <output_file>   Specify the output file.")
    print("  --req <requirements>  Specify the requirements as a string.")
    print("  --force               Overwrite the output file if it exists.")
    print("  --mode <model>        Specify the model to use. Available models:")
    for model in models:
        print(f"                          {model}")
    print()
    print("Sample control file (autocoder.ctl):")
    print('''
{
    "infilename": "autocoder.py",
    "outfilename": "autocoder.001.py",
    "requirements": "I want you to modify this code with these requirements:"
}
''')
    sys.exit(1)

def parse_command_line():
    infilename = None
    outfilename = None
    requirements = None
    model = default_model
    force = False

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--ctl":
            if i + 1 >= len(sys.argv):
                print_help()
            with open(sys.argv[i + 1], 'r') as ctl_file:
                config = json.load(ctl_file)
                infilename = config.get("infilename")
                outfilename = config.get("outfilename")
                requirements = config.get("requirements")
            i += 2
        elif arg == "--in":
            if i + 1 >= len(sys.argv):
                print_help()
            infilename = sys.argv[i + 1]
            i += 2
        elif arg == "--out":
            if i + 1 >= len(sys.argv):
                print_help()
            outfilename = sys.argv[i + 1]
            i += 2
        elif arg == "--req":
            if i + 1 >= len(sys.argv):
                print_help()
            requirements = sys.argv[i + 1]
            i += 2
        elif arg == "--force":
            force = True
            i += 1
        elif arg == "--mode":
            if i + 1 >= len(sys.argv):
                print_help()
            model = sys.argv[i + 1]
            if model not in models:
                print(f"Model {model} not found. Please choose from the available models.")
                print_help()
            i += 2
        else:
            print_help()

    if not infilename or not outfilename or not requirements:
        print_help()

    return infilename, outfilename, requirements, model, force

class AICoder:
    def __init__(self, infilename, outfilename, requirements, model, input_cost_per_million, output_cost_per_million, max_tokens, context_length, force):
        self.infilename = infilename
        self.outfilename = outfilename
        self.requirements = requirements
        self.model = model
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.max_tokens = max_tokens
        self.context_length = context_length
        self.force = force
        self.program = self.read_program()
        self.api_key = self.read_api_key()
        self.language = self.determine_language()
        self.user_content = self.create_user_content()
        if model_info['provider'] == 'openai':
            self.client = openai.OpenAI(api_key=self.api_key, base_url="https://api.openai.com/v1")
        elif model_info['provider'] == 'hyperbolic':
            self.client = openai.OpenAI(api_key=self.api_key, base_url="https://api.hyperbolic.xyz/v1")
        self.continuation_message = "<!--generation interrupted, continuing-->"

    def read_program(self):
        with open(self.infilename, 'r') as file:
            return file.read()

    def read_api_key(self):
        if model_info['provider'] == 'openai':
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                return api_key.strip()
        elif model_info['provider'] == 'hyperbolic':
            api_key = os.environ.get("HYPERBOLIC_API_KEY")
            if api_key:
                return api_key.strip()
        # If not found in environment variables, read from a file
        try:
            if model_info['provider'] == 'openai':
                with open('openai-api-key.txt', 'r') as file:
                    return file.read().strip()
            elif model_info['provider'] == 'hyperbolic':
                with open('hyperbolic-api-key.txt', 'r') as file:
                    return file.read().strip()
        except FileNotFoundError:
            print(f"API key file not found for {model_info['provider']}. Please ensure the key is set in an environment variable or in the correct file.")
            sys.exit(1)

    def determine_language(self):
        language_map = {
            '.txt': 'text',
            '.py': 'python',
            '.pl': 'perl',
            '.sql': 'sql',
            '.sh': 'shell',
        }
        file_extension = os.path.splitext(self.infilename)[1]
        return language_map.get(file_extension, 'text')  # Default to 'text' if unknown extension

    def create_user_content(self):
        return f"""
USER: What follows is a {self.language} file. Please reference it for instructions following.

```{self.language}
{self.program}
```

{self.requirements}

ASSISTANT:
"""

    def generate_response(self, prompt, temperature=0.7):
        response_chunks = []
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = None

        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=self.max_tokens,
            stream=True,  # Enable streaming
        )

        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                partial_response = chunk.choices[0].delta.content
                print(partial_response, end='', flush=True)

                response_chunks.append(partial_response)
            prompt_tokens = max(prompt_tokens, chunk.usage.prompt_tokens)
            completion_tokens = max(completion_tokens, chunk.usage.completion_tokens)
            finish_reason = chunk.choices[0].finish_reason

            if finish_reason is not None:
                break

        return response_chunks, prompt_tokens, completion_tokens, finish_reason

    def run(self):
        # Capture the start time
        start_time = time.time()
        start_timestring = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check if outfile exists and overwrite if --force is specified
        if os.path.exists(self.outfilename) and not self.force:
            print(f"{self.outfilename} exists already; cowardly refusing.")
            sys.exit(1)

        # Main loop for generating and continuing responses
        all_response_chunks = []
        prompt_tokens_total = 0
        completion_tokens_total = 0
        final_reason = None

        while True:
            response_chunks, prompt_tokens, completion_tokens, finish_reason = self.generate_response(self.user_content)
            all_response_chunks.extend(response_chunks)
            prompt_tokens_total += prompt_tokens
            completion_tokens_total += completion_tokens

            if finish_reason == "length":
                all_response_chunks = all_response_chunks[:-5]
                with open(self.outfilename, "a", encoding="utf-8") as response_file:
                    response_file.writelines(all_response_chunks)
                self.user_content += ''.join(all_response_chunks)
                self.user_content += self.continuation_message
                print(self.continuation_message)
                response_file.writelines(self.continuation_message)
                all_response_chunks = []
            else:
                final_reason = finish_reason
                break

        # Write the remaining valid chunks to the output file
        with open(self.outfilename, "a", encoding="utf-8") as response_file:
            response_file.writelines(all_response_chunks)

        # Match the file permissions of the input file for the output file
        shutil.copymode(self.infilename, self.outfilename)

        # Calculate costs
        input_cost = (prompt_tokens_total / 1_000_000) * self.input_cost_per_million
        output_cost = (completion_tokens_total / 1_000_000) * self.output_cost_per_million
        total_cost = input_cost + output_cost

        # Print the program run details
        end_timestring = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\nProgram Begin: {start_timestring}")
        print(f"Program End  : {end_timestring}")
        print(f"  Input tokens: {prompt_tokens_total}, Cost: ${input_cost:.6f}")
        print(f"  Output tokens: {completion_tokens_total}, Cost: ${output_cost:.6f}")
        print(f"  Total cost: ${total_cost:.6f}")
        print(f"  Final finish reason: {final_reason}")

        # Calculate the duration of the run in seconds
        end_time = time.time()
        duration_seconds = end_time - start_time

        # Extrapolate the cost for an 8-hour run
        cost_per_second = total_cost / duration_seconds if duration_seconds > 0 else 0
        eight_hour_cost = cost_per_second * (8 * 60 * 60)

        print(f"If I literally run this all day (8 hours) at this rate, it would cost: ${eight_hour_cost:.2f}")

# Usage
if __name__ == "__main__":
    import shutil  # Import shutil for copymode

    infilename, outfilename, requirements, model, force = parse_command_line()
    model_info = models[model]

    ai_coder = AICoder(infilename, outfilename, requirements, model, model_info['input_cost'], model_info['output_cost'], model_info['max_tokens'], model_info['context_length'], force)
    ai_coder.run()
