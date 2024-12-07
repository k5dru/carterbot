# Autocoder: AI-Powered Code Generation Tool
==============================================

Autocoder is a command-line tool designed to leverage the power of artificial intelligence (AI) for code generation. It utilizes models from various providers, including OpenAI and Hyperbolic, to generate code based on user input. The tool is written in Python and uses the `openai` library to interact with the AI models.

## Goals and Details
--------------------

The primary goal of Autocoder is to provide a flexible and user-friendly tool for generating code using AI models. The project aims to demonstrate the capabilities of AI in code generation and provide a useful tool for developers. Key features include:

- Support for multiple AI models from different providers.
- Estimating token count for input and output to manage costs effectively.
- A continuation mechanism to handle responses that exceed the model's context length.
- System prompt to ensure seamless continuation of responses after interruptions.
- Integration with `argparse` for setting options and parameters.
- Debugging modes to help with troubleshooting and understanding the tool's behavior.

## Similar Projects
-------------------

There are several similar projects available, including:

* **GitHub's Copilot**: A code generation tool that uses AI to suggest code completions. It integrates directly into popular code editors and focuses on real-time suggestions.
* **Kite**: An AI-powered coding assistant that provides code completions and documentation. Kite also integrates with code editors and offers a wide range of features for developers.
* **TabNine**: A code completion tool that uses AI to predict and complete code. TabNine supports multiple programming languages and integrates with various code editors.

**Autocoder** differs from these projects in its focus on command-line interaction and support for multiple AI models. It is designed to be a standalone tool that can be used for generating code from a set of input files and requirements, rather than an editor plugin.

## API Key Storage
------------------

It is recommended to store API keys as environment variables, as this is a more secure approach. However, Autocoder also supports storing API keys in a text file as a fallback. Please note that storing API keys in a text file is widely considered insecure and should be avoided whenever possible.

## Hyperbolic API Key
---------------------

As of December 2024, a key from Hyperbolic.xyz is free and comes with $10 in free credits. This amount of credits would take literal weeks to exhaust, even when running the program constantly.

## OpenAI's Moat
----------------

It's worth noting that OpenAI's moat, if it ever existed, is gone. The availability of alternative AI models and providers has leveled the playing field, and Autocoder takes advantage of this by supporting multiple models.

## Continuation Mechanism
------------------------

Incomplete chunks from the AI provider have been addressed through a continuation mechanism and system prompt. However, when using models with small max token limits, continuation messages may be present in the final output. As always, a human should review the results with `diff -u` before discarding the input files. The smallest max token API call is 8192 tokens, which translates to a source code file of about 32KB. Models with larger max token parameters should scale accordingly.

## Author and Contributions
---------------------------

Autocoder is a personal non-work project created by James Lemley, who has no connection to any AI service provider. Pull requests may be responded to or ignored, as this is a personal project.

## Supported Models
--------------------

The following models are currently supported by Autocoder:

| Model Name                      | Input Cost (per million tokens) | Output Cost (per million tokens) | Context Length | Max Tokens | Provider   |
|---------------------------------|---------------------------------|----------------------------------|----------------|------------|------------|
| gpt-4o-mini                     | 0.15                            | 0.6                              | 128000         | 16384      | OpenAI     |
| meta-llama/Llama-3.3-70B-Instruct | 0.4                             | 0.4                              | 131072         | 131072     | Hyperbolic |
| Qwen/Qwen2.5-Coder-32B-Instruct   | 0.2                             | 0.2                              | 131072         | 8192       | Hyperbolic |
| Qwen/QwQ-32B-Preview            | 0.2                             | 0.2                              | 32768          | 32768      | Hyperbolic |

## Usage Examples
------------------

### Example 1: Generate a Basic Python Script

**Requirements**: Create a simple Python script that prints "Hello, World!".

**Command**:
```bash
./autocoder.py -i input_script.py -o output_script.py -r "Create a simple Python script that prints 'Hello, World!'."
```

### Example 2: Use the `argparse` Library to Set Options

**Requirements**: Use the `argparse` library to set options for a Python script that accepts a filename and prints its contents.

**Command**:
```bash
./autocoder.py -i input_script.py -o output_script.py -r "Use the argparse library to set options for a Python script that accepts a filename and prints its contents."
```

## Usage
---------

To use Autocoder, simply run the `autocoder.py` script with the required arguments. For more information, please refer to the `autocoder.py` file and the command-line help.

## Requirements
---------------

* Python 3.x
* `openai` library
* API key from a supported provider (OpenAI or Hyperbolic)

## Conclusion
----------

Autocoder is a powerful tool for generating code using AI models. Its flexibility, support for multiple models, and user-friendly interface make it an attractive choice for developers. With the availability of free API keys and credits, there's never been a better time to explore the capabilities of AI-powered code generation.
