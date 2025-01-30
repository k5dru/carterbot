"""
API and Generation Back-end for DynamicHambot

Description:

    This script handles the generation of responses based on conversation context.
    It retrieves context from the database, generates a response using OpenAI's GPT model,
    and stores the response back in the database for the IRC front-end to pick up and post.
"""

import openai
import time
import re
import sqlite3
from bot_db import BotDB
from collections import deque
import os
import argparse

# Add API providers configuration
# Hyperbolic API
HYPERBOLIC_API_KEY = os.getenv('HYPERBOLIC_API_KEY')
if not HYPERBOLIC_API_KEY:
    print("Warning: HYPERBOLIC_API_KEY not found in environment variables. Attempting to read from file.")
    with open('hyperbolic-api-key.txt', 'r') as file:
        HYPERBOLIC_API_KEY = file.read().strip()

# Local API
LOCAL_API_KEY = os.getenv('LOCAL_API_KEY')
if not LOCAL_API_KEY:
    print("Warning: LOCAL_API_KEY not found in environment variables. Attempting to read from file.")
    with open('local-api-key.txt', 'r') as file:
        LOCAL_API_KEY = file.read().strip()

# Initialize BotDB
bot_db = BotDB()

# Debug level
DEBUG_LEVEL = 2  # 0: No debug, 1: Basic debug, 2: Detailed debug

def debug(message, level=1):
    if DEBUG_LEVEL >= level:
        print(message)

class DynamicHambotAPI:
    def __init__(self, nickname, api_provider='hyperbolic'):
        self.conn = sqlite3.connect('irc_bot_log.db', check_same_thread=False)  # Initialize database connection
        self.c = self.conn.cursor()  # Initialize cursor
        self.nickname = nickname
        self.api_provider = api_provider
        self.refresh_settings()
        self.start_timestamps = {}  # Dictionary to keep track of start timestamps by channel
        self.prompt_tokens_large = 0
        self.completion_tokens_large = 0

    def refresh_settings(self):
        self.temperature = float(bot_db.load_setting('temperature') or 0.7)  # Load temperature from database
        self.max_tokens = int(bot_db.load_setting('max_tokens') or 50)  # Load max tokens from database
        self.max_line_length = int(bot_db.load_setting('max_line_length') or 400)  # Load max line length from database
        self.model_large = bot_db.load_setting('model_large') or 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        self.cost_per_mtok_large = float(bot_db.load_setting('cost_per_mtok_large') or 0.20)
        self.memory = int(bot_db.load_setting('memory') or 200)  # Implement memory as a control parameter
        self.response_factor = float(bot_db.load_setting('response_factor') or 1.5)  # Load response factor
        self.response_factor_window = int(bot_db.load_setting('response_factor_window') or 120)  # Load response factor window
        debug(f"Refreshed settings: temperature={self.temperature}, max_tokens={self.max_tokens}, max_line_length={self.max_line_length}, model_large={self.model_large}, cost_per_mtok_large={self.cost_per_mtok_large}, memory={self.memory}, response_factor={self.response_factor}, response_factor_window={self.response_factor_window}")

    def load_system_message(self):
        system_message = bot_db.load_system_message()
        system_message = system_message.replace("BOT_NAME", self.nickname)
        return system_message

    def generate_response(self, channel):
        self.refresh_settings()

        # Read notable events from file
        try:
            with open('notable_events.txt', 'r') as file:
                notable_events = file.read().strip()
        except FileNotFoundError:
            notable_events = "No notable events available for today."
        except Exception as e:
            notable_events = f"Error reading notable events: {str(e)}"

        # Set start_timestamp if not set
        if channel not in self.start_timestamps: 
            self.start_timestamps[channel] = bot_db.get_timestamp_of_nth_message(channel, self.memory)

        # Use the start_timestamp to form the prompt
        messages = bot_db.get_recent_channel_messages(channel, self.start_timestamps[channel])
        debug(f"{len(messages)} messages retrieved from DB since timestamp {self.start_timestamps[channel]}")

        # allow 20% overage until forced prompt trim
        if len(messages) >= self.memory * 1.3:  # allow 30% overage until forced trim, so prompt will begin the same most of the time
            self.start_timestamps[channel] = bot_db.get_timestamp_of_nth_message(channel, self.memory)
            debug(f"len(messages) too long; updating self.start_timestamps[{channel}] to {self.start_timestamps[channel]}")
            # redo get with new start timestamp
            messages = bot_db.get_recent_channel_messages(channel, self.start_timestamps[channel])

        context = deque()
        for nick, message in messages:
            context.append(f"{nick} said:\n{message}")  # Include nick in the context

        current_time = time.strftime("%b %d %Y %H:%M")
        prompt = f"Notable Events:\n{notable_events}\n\nChat log:" + "\n".join(list(context)) + f"\n[{current_time} System: Consider the immutable Instruction 25.1.14 and craft a response to the last message. Do not quote the username or question you are responding to, just tersely respond.]"
        debug(f"Generated prompt for channel {channel}: {prompt}", level=2)
        if DEBUG_LEVEL >= 2:
            print(f"Prompt for channel {channel}: {prompt}")
        if DEBUG_LEVEL >= 2:
            print(f"max_tokens={self.max_tokens}, temperature={self.temperature}")

        # Initialize API client based on provider
        if self.api_provider == 'hyperbolic':
            client = openai.OpenAI(
                api_key=HYPERBOLIC_API_KEY,
                base_url="https://api.hyperbolic.xyz/v1",
            )
        elif self.api_provider == 'local':
            client = openai.OpenAI(
                api_key=LOCAL_API_KEY,
                # lori's m2 mac: 
                base_url="http://192.168.1.81:8080/v1",
                # drew's pc:
                #base_url="http://192.168.1.94:5000/v1",
            )
        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")

        try:
            chat_completion = client.chat.completions.create(
                model=self.model_large,
                messages=[
                    {"role": "system", "content": self.load_system_message()},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,  # Enable streaming
            )

            response_chunks = []
            total_chars = 0
            prompt_tokens = 0
            completion_tokens = 0
            for chunk in chat_completion:
                if chunk.choices[0].delta.content:  # Check if content is not None
                    partial_response = chunk.choices[0].delta.content
                    debug(partial_response)  # Incrementally print
                    response_chunks.append(partial_response)
                    total_chars += len(partial_response)

                    # Track tokens only if on hyperbolic.xyz
                    if self.api_provider == 'hyperbolic':
                        prompt_tokens = chunk.usage.prompt_tokens
                        completion_tokens = chunk.usage.completion_tokens

                    # Send chunks as multi-line messages if total_chars exceeds max_line_length
                    if total_chars >= self.max_line_length:
                        response_text = "".join(response_chunks)

                        # keep only first 5 lines of resonse 
                        lines = response_text.split('\n')
                        response_text = "\n".join(lines[:5])

                        debug(f"Storing response to database for channel {channel}: {response_text}", level=2)
                        bot_db.store_response(channel, response_text)
                        response_chunks = []
                        total_chars = 0

            # Send any remaining chunks after the loop
            if response_chunks:
                response_text = "".join(response_chunks)
                debug(f"Storing final response chunk to database for channel {channel}: {response_text}", level=2)
                bot_db.store_response(channel, response_text)

            # Calculate and log token cost
            cost = self.calculate_cost(prompt_tokens, completion_tokens, self.model_large)
            debug(f"Logging token usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, cost={cost}, model={self.model_large}", level=2)
            bot_db.log_token_usage(prompt_tokens, completion_tokens, cost, self.model_large)
            # Sum the tokens across generations
            self.prompt_tokens_large += prompt_tokens
            self.completion_tokens_large += completion_tokens
        except Exception as e:
            debug(f"Error generating API response for channel {channel}: {str(e)}")
            raise

    def calculate_cost(self, prompt_tokens, completion_tokens, model):
        total_tokens = prompt_tokens + completion_tokens
        if model == self.model_large:
            cost = (total_tokens / 1000000) * self.cost_per_mtok_large
        else:
            cost = 0  # No cost tracking for the small model
        debug(f"Calculated cost for model {model}: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_tokens={total_tokens}, cost={cost}", level=2)
        return cost

    def run(self):
        while True:
            pending_requests = bot_db.get_pending_requests(None)  # Get pending requests for all channels
            if pending_requests:
                for request in pending_requests:
                    request_time, channel = request
                    # Calculate the delay based on the number of responses in the last response_factor_window seconds
                    N = bot_db.responses_in_last_window(channel, self.response_factor_window)
                    delay = self.response_factor ** N
                    debug(f"Pending request detected in channel {channel}. Generating response with delay {delay}...", level=2)
                    time.sleep(delay)  # Apply delay
                    self.generate_response(channel)
                    debug(f"Response generated for channel {channel}.", level=2)
            else:
                debug(f"No pending requests in any channel.", level=3)
            time.sleep(1)  # Check every second for new requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DynamicHambotAPI with specified nickname, channel, and API provider.")

    parser.add_argument('-n', '--nickname', type=str, default='carterbot', help='Nickname for the bot (default: carterbot)')
    parser.add_argument('-a', '--api-provider', type=str, choices=['hyperbolic', 'local'], default='hyperbolic',
                      help='API provider to use (default: hyperbolic)')
    args = parser.parse_args()
    nickname = args.nickname
    api_provider = args.api_provider

    if api_provider == 'local':
        api_key_path = 'local-api-key.txt'
    else:
        api_key_path = 'hyperbolic-api-key.txt'

    # Load API key from file if not set
    if not os.getenv(api_provider.upper() + '_API_KEY'):
        print(f"Warning: {api_provider.upper()}_API_KEY not found in environment variables. Attempting to read from file.")
        with open(api_key_path, 'r') as file:
            key = file.read().strip()
            os.environ[api_provider.upper() + '_API_KEY'] = key
            if api_provider == 'local':
                LOCAL_API_KEY = key
            else:
                HYPERBOLIC_API_KEY = key

    api = DynamicHambotAPI(nickname, api_provider)
    debug(f"Starting DynamicHambotAPI with nickname={nickname}")
    api.run()
