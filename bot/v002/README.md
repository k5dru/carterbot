# DynamicHambot

DynamicHambot is an IRC chatbot designed to interact with users in an IRC channel. It uses OpenAI's GPT models to generate responses based on the conversation context. The bot is configured via a SQLite database and can be controlled by superusers through IRC commands.

## Features

- Dynamic bot name changeable via IRC command.
- Control commands to adjust settings.
- Anti-flood delay between messages.
- Streaming of AI responses.
- Configurable maximum line length for responses.
- Token usage and cost tracking.
- Context buffer includes user nicks and messages.
- Logs all messages for debugging and analysis.
- Pre-populates context with last 200 lines from the database.

## Requirements

- Python 3.8 or higher.
- `irc` library for Python: `pip install irc`.
- `openai` library for Python: `pip install openai`.
- `sqlite3` for Python (usually included with Python).
- API keys for OpenAI or Hyperbolic API.

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/k5dru/carterbot/bot/v002.git
   cd v002
   ```

2. **Install Dependencies**

   ```bash
   pip install irc openai
   ```

3. **Configure API Keys**

   You need to configure the API keys for either OpenAI or Hyperbolic API. You can do this by setting environment variables or by creating a file with the API key.

   - **Using Environment Variables**

     ```bash
     export HYPERBOLIC_API_KEY='your_hyperbolic_api_key'
     export LOCAL_API_KEY='your_local_api_key'
     ```

   - **Using Files**

     Create `hyperbolic-api-key.txt` and `local-api-key.txt` in the project directory and add your API keys to these files respectively.

4. **Configure System Message**

   The system message is crucial for defining the bot's personality and behavior. You can customize it by editing the `system_message.txt` file. A sample template is provided in `example_system_message.txt`.

   ```text
   You are BOT_NAME, an IRC chatbot interacting with an IRC channel. 

   You are (add personality traits here) 

   Respond (in a way you like here)

   Key Points

   - Concise Responses: Keep each response under 30 words. Do not answer in 50 words when 10 will do.
   - Do Not Repeat Yourself: Vary your responses, even if given the same instruction.
   - Contextual Relevance: Tailor responses to the interests of the channel.

   Anything you write will be sent as a regular response, unless you prefix it with /me which will turn it into an action.

   /me rolls eyes
   /me laughs
   /me sighs 
   /me gives (participant) a sandwich

   You get the idea. It only works if you begin the whole message with /me.

   (other important instructions here)
   ```

   Replace `BOT_NAME` with your desired bot name and add your own personality traits and instructions.

5. **Configure Model**

   The bot uses the Meta-Llama 3.1 8B Instruct model by default, which works pretty well for most use cases. You can change the model by updating the `model_large` setting in the `control_settings` table of the `irc_bot_log.db` database.

   ```sql
   UPDATE control_settings SET value = 'your_new_model_name' WHERE setting = 'model_large';
   ```

## Running the Bot

DynamicHambot requires both `bot_api.py` and `bot_irc.py` to be running simultaneously in separate terminals or windows.

1. **Start `bot_api.py`**

   ```bash
   python bot_api.py --nickname your_bot_name --api-provider hyperbolic
   ```

   - `your_bot_name`: Replace with your desired bot name.
   - `hyperbolic`: You can also use `local` if you have a local API server.

2. **Start `bot_irc.py`**

   ```bash
   python bot_irc.py --server irc.geekshed.net --port 6667 --nickname your_bot_name --channel #your_channel
   ```

   - `irc.geekshed.net`: Replace with your desired IRC server.
   - `6667`: Replace with the port number of the IRC server.
   - `your_bot_name`: Replace with your desired bot name.
   - `#your_channel`: Replace with the channel you want the bot to join.

## Control Commands

The bot can be controlled via IRC commands by superusers. Superusers are defined in the `control_settings` table under the `superusers` setting.

- **Set a Setting**

  ```
  your_bot_name set setting_name setting_value
  ```

  Example:

  ```
  carterbot set temperature 0.5
  ```

- **Show a Setting**

  ```
  your_bot_name show setting_name
  ```

  Example:

  ```
  carterbot show temperature
  ```

- **Show All Settings**

  ```
  your_bot_name show all
  ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
