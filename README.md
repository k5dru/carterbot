# Autocoder: AI-Powered Code Generation Tool

Autocoder is a command-line tool that leverages the power of artificial intelligence to generate code based on user-provided requirements and input files. With Autocoder, you can automate the process of writing code, freeing up your time to focus on more complex and creative tasks.

## Features

* Supports multiple AI models from various providers, including OpenAI and Hyperbolic
* Allows users to specify requirements and input files in various programming languages
* Generates code based on user input, with options for customizing the generation process
* Supports streaming output, allowing for real-time feedback and debugging
* Calculates costs based on input and output tokens, providing transparency and cost control
* List available models and their costs with the `--list-models` option

## Usage

To use Autocoder, simply run the `autocoder.py` script with the required arguments, including the input file, output file, and requirements. For example:
```bash
python autocoder.py -i input.py -o output.py -r "Generate a Python class for handling user authentication"
```
You can also use a control file to specify the input file, output file, and requirements in a JSON format. For example:
```json
{
    "input_files": ["input.py"],
    "output_file": "output.py",
    "requirements": "Generate a Python class for handling user authentication"
}
```
To use the control file, run the `autocoder.py` script with the `--control-file` argument:
```bash
python autocoder.py -c control.json
```
You can also specify the model to use with the `--model` option. For example:
```bash
python autocoder.py -i input.py -o output.py -r "Generate a Python class for handling user authentication" -m Qwen/Qwen2.5-Coder-32B-Instruct
```
To list the available models and their costs, use the `--list-models` option:
```bash
python autocoder.py -l
```
## Requirements

* Python 3.x
* OpenAI or Hyperbolic API key (depending on the chosen model)
* Optional: control file in JSON format

## Installation

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies using pip: `pip install -r requirements.txt`
3. Set up your API key by creating a file named `openai-api-key.txt` (or `hyperbolic-api-key.txt`) in the project directory, containing your API key.

## Costs and API Keys

As of December 2024, a key from Hyperbolic.xyz is free and comes with $10 in free credits. This amount of credits would take literal weeks to blow through with this program, making it an attractive option for users. Note that the author of this project, James Lemley, has no connection to any AI service provider.

## OpenAI's Moat

It's worth noting that OpenAI's moat, if it ever existed, is gone. With the rise of alternative AI providers like Hyperbolic, users now have more options than ever before. This increased competition is driving innovation and reducing costs, making AI-powered tools like Autocoder more accessible to everyone.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request. Please note that, as this is a personal non-work project, pull requests might be responded to or ignored.

## License

Autocoder is released under the MIT License. See the LICENSE file for details.