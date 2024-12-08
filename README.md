# Autocoder: AI-Powered Code Generation Tool
==============================================

Autocoder is a command-line tool designed to leverage the power of artificial intelligence (AI) for code generation. It utilizes models from various providers, including OpenAI and Hyperbolic, to generate code based on user input. The tool is written in Python and uses the `openai` library to interact with the AI models.

## Goals and Details
--------------------

The primary goal of Autocoder is to provide a flexible and user-friendly tool for generating code using AI models. The project aims to demonstrate the capabilities of AI in code generation and provide a useful tool for developers. Key features include:

- **Support for Multiple AI Models**: Autocoder can use different models from various providers to generate code. This flexibility allows users to choose the best model based on their specific needs and budgets.
- **Token Count Estimation**: The tool estimates the number of input and output tokens to help users manage costs effectively. This is crucial for optimizing the use of API credits.
- **Continuation Mechanism**: Autocoder handles responses that exceed the model's context length by using a continuation message and system prompt. This ensures that the generated code is complete and coherent.
- **System Prompt**: The system prompt guides the AI to provide detailed, neutral, and fact-based answers, facilitating inquiry without judgment or censorship.
- **Integration with `argparse`**: Autocoder uses `argparse` for setting options and parameters, making it easy to use and customize.
- **Debugging Modes**: The tool includes different debugging modes to help with troubleshooting and understanding its behavior.

## Design Points and Rationale
------------------------------

### Multiple Model Support
- **Reason**: By supporting multiple models from different providers, Autocoder offers users a range of options. This is particularly beneficial given the variability in model performance and pricing.
- **Implementation**: Models are defined in a dictionary with their respective costs, context lengths, and providers. Users can specify their preferred model using the `-m` or `--model` option.

### Token Count Estimation
- **Reason**: Estimating token counts helps users manage their API credits more effectively. It also ensures that the input does not exceed the model's context length, which could lead to incomplete or erroneous responses.
- **Implementation**: A function `estimate_token_count` is used to estimate the number of tokens in the input. If `tiktoken` is available and the model is from OpenAI, it uses `tiktoken` for accurate token counting. Otherwise, it uses a rule of thumb based on character counts.

### Continuation Mechanism
- **Reason**: AI models have context length limitations. When the generated response exceeds these limits, a continuation mechanism is necessary to ensure the response is complete.
- **Implementation**: The tool uses a continuation message (`<!--generation interrupted, continuing-->`) to request the completion of the response. It also ensures that the response is appended correctly to the output file.

### System Prompt
- **Reason**: The system prompt ensures that the AI behaves consistently and provides the expected type of responses. It also allows the tool to handle interruptions seamlessly.
- **Implementation**: The system prompt is defined as a constant string and sent with every request to the AI model.

### Integration with `argparse`
- **Reason**: Using `argparse` makes the tool more accessible and user-friendly. It allows users to specify various options and parameters easily.
- **Implementation**: The tool parses command-line arguments using `argparse` and uses these to configure the behavior of the AI model.

### Debugging Modes
- **Reason**: Debugging modes help users understand how the tool works and troubleshoot issues. They provide detailed information about the input, output, and internal state of the tool.
- **Implementation**: The tool supports different debug levels, which can be specified using the `-d` or `--debug-level` option. Higher debug levels provide more detailed output.

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

The following models are currently supported by Autocoder. The costs, context lengths, and max tokens are as specified in the code.

| Model Name                      | Input Cost (per million tokens) | Output Cost (per million tokens) | Context Length | Max Tokens | Provider   |
|---------------------------------|---------------------------------|----------------------------------|----------------|------------|------------|
| gpt-4o-mini                     | 0.15                            | 0.6                              | 128000         | 16384      | OpenAI     |
| meta-llama/Llama-3.3-70B-Instruct | 0.4                             | 0.4                              | 131072         | 65535      | Hyperbolic |
| Qwen/Qwen2.5-Coder-32B-Instruct   | 0.2                             | 0.2                              | 131072         | 8192       | Hyperbolic |
| Qwen/QwQ-32B-Preview            | 0.2                             | 0.2                              | 32768          | 16384      | Hyperbolic |

## Getting Started
------------------

### Prerequisites
1. **Python 3.x**: Ensure you have Python 3.x installed on your system.
2. **OpenAI or Hyperbolic API Key**: Obtain an API key from a supported provider. It is recommended to store the API key as an environment variable.
    - For OpenAI, set the `OPENAI_API_KEY` environment variable.
    - For Hyperbolic, set the `HYPERBOLIC_API_KEY` environment variable.
3. **Install Required Libraries**: Install the necessary Python packages using pip.
    ```bash
    pip install -r requirements.txt
    ```

### Basic Usage
To use Autocoder, run the `autocoder.py` script with the required arguments. Here are some examples:

#### Example 1: Generate a Basic Python Script

**Requirements**: Create a simple Python script that prints "Hello, World!".

**Command**:
```bash
./autocoder.py -i input_script.py -o output_script.py -r "Create a simple Python script that prints 'Hello, World!'."
```

#### Example 2: Use the `argparse` Library to Set Options

**Requirements**: Use the `argparse` library to set options for a Python script that accepts a filename and prints its contents.

**Command**:
```bash
./autocoder.py -i input_script.py -o output_script.py -r "Use the argparse library to set options for a Python script that accepts a filename and prints its contents."
```

### Advanced Usage
Autocoder includes several options to customize its behavior. You can specify multiple input files, overwrite the output file, choose a model, and set the debug level.

**Command**:
```bash
./autocoder.py -i file1.py file2.py -o output.py -r "Add functionality to handle multiple input files." -m "Qwen/Qwen2.5-Coder-32B-Instruct" -f -d 2
```

### Options
- `-c`, `--control-file`: Specify a control file in JSON format.
- `-i`, `--input-files`: Specify one or more input files. You can use this option multiple times or provide multiple files separated by spaces.
- `-o`, `--output-file`: Specify the output file.
- `-r`, `--requirements`: Specify the requirements as a string.
- `-f`, `--force`: Overwrite the output file if it exists and truncate it first.
- `-m`, `--model`: Specify the model to use. The default model is `Qwen/Qwen2.5-Coder-32B-Instruct`.
- `-l`, `--list-models`: List available models and their costs.
- `-d`, `--debug-level`: Set the debug level (1=info, 2=debug, 3=dump all JSON objects).

## Potential Future Development Options
--------------------------------------

1. **Support for Additional Models**: Expand the list of supported models to include more providers and models as they become available.
2. **Improved Token Count Estimation**: Enhance the token count estimation mechanism to use `tiktoken` for all models, if possible.
3. **Graphical User Interface (GUI)**: Develop a GUI for users who prefer a more visual interface.
4. **Batch Processing**: Add support for batch processing multiple sets of input files and requirements.
5. **Integration with Version Control Systems (VCS)**: Integrate Autocoder with VCS systems to automatically commit and track changes in generated code.
6. **Enhanced Debugging Tools**: Improve debugging tools to provide more detailed insights into the tool's behavior and performance.
7. **User Feedback Loop**: Implement a user feedback loop to continuously improve the quality and relevance of generated code.

## Conclusion
----------

Autocoder is a powerful tool for generating code using AI models. Its flexibility, support for multiple models, and user-friendly interface make it an attractive choice for developers. With the availability of free API keys and credits, there's never been a better time to explore the capabilities of AI-powered code generation. Whether you're a developer looking to speed up your workflow or an AI enthusiast interested in code generation, Autocoder has something to offer.