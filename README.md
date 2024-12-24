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
- **Control File**: Users can save sets of parameters that work well in a control file for later use, making it easier to replicate successful configurations.

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

### Control File
- **Reason**: A control file allows users to save and reuse sets of parameters that have been proven to work well for specific tasks.
- **Implementation**: Users can create a control file using the `--create-controlfile` option. The control file is a JSON file that contains the input files, output file, requirements, model, force flag, debug level, and temperature.

## Similar Projects
-------------------

There are several similar projects available, including:

* **GitHub's Copilot**: A code generation tool that uses AI to suggest code completions. It integrates directly into popular code editors and focuses on real-time suggestions.
* **Kite**: An AI-powered coding assistant that provides code completions and documentation. Kite also integrates with code editors and offers a wide range of features for developers.
* **TabNine**: A code completion tool that uses AI to predict and complete code. TabNine supports multiple programming languages and integrates with various code editors.

**Autocoder** differs from these projects in its focus on command-line interaction and support for multiple AI models. It is designed to be a standalone tool that can be used for generating code from a set of input files and requirements, rather than an editor plugin.

## Author and Contributions
---------------------------

Autocoder is a personal non-work project created by James Lemley, who has no connection to any AI service provider. Pull requests may be responded to or ignored, as this is a personal project.

## Supported Models
--------------------

The following models are currently supported by Autocoder. The costs, context lengths, and max tokens are as specified in the code.

| Model Name                      | Input Cost (per million tokens) | Output Cost (per million tokens) | Context Length | Max Tokens | Provider   |
|---------------------------------|---------------------------------|----------------------------------|----------------|------------|------------|
| gpt-4o-mini                     | 0.15                            | 0.6                              | 128000         | 16384      | OpenAI     |
| gpt-4o                          | 2.50                            | 10.0                             | 128000         | 16384      | OpenAI     |
| gpt-4o-2024-05-13               | 5.00                            | 15.0                             | 128000         | 16384      | OpenAI     |
| o1-preview                      | 15.00                           | 60.0                             | 128000         | 16384      | OpenAI     |
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
- `-t`, `--temperature`: Set the temperature for the model.
- `--create-controlfile`: Create a control file with the specified parameters in JSON format.

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

*James did nothing but yell at me; this entire project was written by Qwen2.5-Coder-32B-Instruct*
