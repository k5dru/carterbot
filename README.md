# Autocoder: AI-Powered Code Generation Tool

Autocoder is a command-line tool that leverages the power of artificial intelligence to generate code based on user-provided requirements and input files. With Autocoder, you can automate the process of writing code, freeing up your time to focus on more complex and creative tasks.

## Features

* Supports multiple AI models from various providers, including OpenAI and Hyperbolic
* Allows users to specify requirements and input files in various programming languages
* Generates code based on user input, with options for customizing the generation process
* Supports streaming output, allowing for real-time feedback and debugging
* Calculates costs based on input and output tokens, providing transparency and cost control

## Usage

To use Autocoder, simply run the `autocoder.py` script with the required arguments, including the input file, output file, and requirements. For example:
```bash
python autocoder.py --input-file input.py --output-file output.py --requirements "Generate a Python class for handling user authentication"
```
You can also use a control file to specify the input file, output file, and requirements in a JSON format. For example:
```json
{
    "input_file": "input.py",
    "output_file": "output.py",
    "requirements": "Generate a Python class for handling user authentication"
}
```
To use the control file, run the `autocoder.py` script with the `--control-file` argument:
```bash
python autocoder.py --control-file control.json
```
## Requirements

* Python 3.x
* OpenAI or Hyperbolic API key (depending on the chosen model)
* Optional: control file in JSON format

## Installation

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies using pip: `pip install -r requirements.txt`
3. Set up your API key by creating a file named `openai-api-key.txt` (or `hyperbolic-api-key.txt`) in the project directory, containing your API key.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

Autocoder is released under the MIT License. See the LICENSE file for details.