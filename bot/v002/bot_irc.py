"""
DynamicHambot Requirements and Implementation Details

Description:

    Connects to an IRC server and participates in a specified channel.
    Uses OpenAI's GPT models to generate responses based on conversation context.
    Controlled via IRC commands by specified superusers.
    Context stored in a SQLite database.
    Configurable settings include temperature, max tokens, antiflood delay, and context buffer.

Features:

    Dynamic bot name changeable via IRC command.
    Control commands to adjust settings.
    Anti-flood delay between messages.
    Streaming of AI responses.
    Configurable maximum line length for responses.
    Token usage and cost tracking.
    Context buffer includes user nicks and messages.
    Logs all messages for debugging and analysis.
    Pre-populates context with last 200 lines from the database.
    Uses a smaller, quicker model to evaluate responses.

Implementation Details:

    System prompt never ages out of the context buffer.
    Context buffer trimming done in chunks.
    Asynchronous multi-line response handling.
    Error handling for API responses.
    Token tracking for the larger model.
    Status report command providing adjustable variables, token usage, and cost.
    Database schema includes token costs and control table for model parameters.
    Bot's own messages added to the context buffer for consistent history.
"""

import irc.bot
import irc.strings
import sqlite3
from argparse import RawTextHelpFormatter
from bot_db import BotDB
import time
import re
import threading

# Initialize BotDB
bot_db = BotDB()

# Debug level
DEBUG_LEVEL = 2  # 0: No debug, 1: Basic debug, 2: Detailed debug

def debug(message, level=1):
    if DEBUG_LEVEL >= level:
        print(message)

class DynamicHambotIRC(irc.bot.SingleServerIRCBot):
    def __init__(self, server, port, nickname, channel):
        irc.bot.SingleServerIRCBot.__init__(self, [(server, port)], nickname, nickname)
        self.channel = channel
        self.nickname = nickname
        self.refresh_settings()
        self.f_count = 0
        self.polling_thread = None  # Initialize without starting the thread

    def refresh_settings(self):
        self.temperature = float(bot_db.load_setting('temperature') or 0.7)  # Load temperature from database
        self.max_tokens = int(bot_db.load_setting('max_tokens') or 50)  # Load max tokens from database
        self.antiflood_delay = float(bot_db.load_setting('antiflood_delay') or 1.0)  # Load antiflood delay from database
        self.posting_enabled = bot_db.load_setting('posting_enabled') == 'True'  # Load posting enabled status from database
        self.system_message = bot_db.load_system_message().replace("BOT_NAME", self.nickname)
        self.max_line_length = int(bot_db.load_setting('max_line_length') or 400)  # Load max line length from database
        self.model_large = bot_db.load_setting('model_large') or 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        self.cost_per_mtok_large = float(bot_db.load_setting('cost_per_mtok_large') or 0.20)
        self.messages_since_activation = int(bot_db.load_setting('messages_since_activation') or 999)
        self.superusers = bot_db.load_setting('superusers').split(',') if bot_db.load_setting('superusers') else []
        self.memory = int(bot_db.load_setting('memory') or 200)  # Implement memory as a control parameter

    def on_welcome(self, c, e):
        c.join(self.channel)
        debug(f"Joined channel {self.channel}")
        self.polling_thread = threading.Thread(target=self.poll_responses, args=(c,))
        self.polling_thread.daemon = True
        self.polling_thread.start()

    def on_pubmsg(self, c, e):
        message = e.arguments[0]
        sender_nick = e.source.nick

        # Log incoming message
        bot_db.log_message(self.channel, sender_nick, message, 'in')

        # Reset activation counter if bot's nickname is mentioned
        if message.lower().startswith(self.nickname.lower()) and sender_nick != self.nickname:
            if len(bot_db.get_pending_requests(self.channel)) < 2:
                # Handle control commands
                if message.startswith(f"{self.nickname} set"): 
                    if sender_nick in self.superusers:
                        self.handle_set_command(c, message)
                    else:
                        self.send_privmsg(c, self.channel, f"only {', '.join(self.superusers)} may set")
                elif message.startswith(f"{self.nickname} show"):
                    if sender_nick in self.superusers:
                        self.handle_show_command(c, message)
                    else:
                        self.send_privmsg(c, self.channel, f"only {', '.join(self.superusers)} may show")
                else:
                    self.messages_since_activation = 0
                    bot_db.request_response(self.channel)  # Signal the back-end to generate a response
            else:
                debug(f"Already two pending requests. Ignoring additional request from {sender_nick}.")
        else:
            if message.lower() == 'f':
                self.f_count += 1
            else:
                self.f_count = 0
            self.messages_since_activation += 1

        debug(f"messages_since_activation {self.messages_since_activation}, f_count {self.f_count}")
        if self.f_count >= 3 and self.messages_since_activation >= 20:
            self.send_privmsg(c, self.channel, "f")
            self.f_count = 0
            self.messages_since_activation = 1

    def on_ctcp(self, c, e):
        if e.arguments[0] == "ACTION":
            action_message = e.arguments[1]
            sender_nick = e.source.nick

            # Log incoming action
            bot_db.log_message(self.channel, sender_nick, f"* {action_message}", 'in')

            # Reset activation counter if bot's nickname is mentioned in the action
            if action_message.lower().startswith(self.nickname.lower()) and sender_nick != self.nickname:
                if len(bot_db.get_pending_requests(self.channel)) < 2:
                    self.messages_since_activation = 0
                    bot_db.request_response(self.channel)  # Signal the back-end to generate a response
                else:
                    debug(f"Already two pending requests. Ignoring additional request from {sender_nick}.")

    def handle_set_command(self, c, message):
        try:
            # Remove the bot's name and "set" from the message
            command_str = message.split(f"{self.nickname} set ")[1].strip()
            # Split the command into parts
            parts = command_str.split()
            if len(parts) < 2:
                self.send_privmsg(c, self.channel, "Invalid set command. Usage: set <setting> <value>")
                return
            setting = parts[0]
            value = ' '.join(parts[1:])
            bot_db.save_setting(setting, value)
            self.refresh_settings()
            self.send_privmsg(c, self.channel, f"Setting '{setting}' updated.")
        except Exception as e:
            self.send_privmsg(c, self.channel, f"Error processing command: {str(e)}")

    def handle_show_command(self, c, message):
        try:
            # Remove the bot's name and "show" from the message
            command_str = message.split(f"{self.nickname} show ")[1].strip()
            if command_str.lower() == 'all':
                all_settings = bot_db.load_all_settings()
                response = ', '.join([f"{key}:{value}" for key, value in all_settings.items()])
                self.send_privmsg(c, self.channel, response)
            else:
                # Load the setting value from the database
                value = bot_db.load_setting(command_str)
                if value:
                    self.send_privmsg(c, self.channel, f"The current value of '{command_str}' is: {value}")
                else:
                    self.send_privmsg(c, self.channel, f"Setting '{command_str}' not found.")
        except Exception as e:
            self.send_privmsg(c, self.channel, f"Error processing command: {str(e)}")

    def send_privmsg(self, c, channel, message):
        if self.posting_enabled:
            c.privmsg(channel, message)
            # Log outgoing message
            bot_db.log_message(channel, self.nickname, message, 'out')
            # Reset activation counter on bot response
            self.messages_since_activation = 0
            debug(f"Sent message: {message}")
            bot_db.mark_response_posted(channel)  # Mark the response as posted

    def send_action(self, c, channel, action):
        if self.posting_enabled:
            c.action(channel, action)
            # Log outgoing action
            bot_db.log_message(channel, self.nickname, f"\u0001ACTION {action}\u0001", 'out')
            # Reset activation counter on bot action
            self.messages_since_activation = 0
            debug(f"Sent action: {action}")
            bot_db.mark_response_posted(channel)  # Mark the response as posted

    def poll_responses(self, c):
        local_bot_db = BotDB() # need a new sqlite connection for the new thread
        while True:
            self.process_response(c, local_bot_db)
            time.sleep(0.5)  # Polling delay
        # no close on local_bot_db; never ends

    def process_response(self, c, local_bot_db):
        response = local_bot_db.get_response(self.channel)
        if response:
            debug(f"about to call send_multiline_response")
            self.send_multiline_response(c, response)
            debug(f"back from send_multiline_response")

    def send_multiline_response(self, c, response):
        debug(f"Sending multiline response: {response}")
        lines = response.split('\n')
        for line in lines:
            if not self.posting_enabled:
                break
            # Check if the line is an action
            if line.startswith('/me'):
                action_text = line[4:].strip()
                self.send_action(c, self.channel, action_text)
            else:
                # Scrub the response to avoid "^.*{self.nickname}: " pattern
                scrubbed_line = re.sub(f'^{self.nickname}: ', '', line, flags=re.MULTILINE)
                self.send_privmsg(c, self.channel, scrubbed_line)
            time.sleep(self.antiflood_delay)  # Apply anti-flood delay

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DynamicHambotIRC with specified server, nickname, and channel.",
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('-s', '--server', type=str, default='irc.geekshed.net',
                        help='IRC server to connect to (default: irc.geekshed.net)')
    parser.add_argument('-p', '--port', type=int, default=6667,
                        help='Port to connect to the IRC server on (default: 6667)')
    parser.add_argument('-n', '--nickname', type=str, default='carterbot',
                        help='Nickname for the bot (default: carterbot)')
    parser.add_argument('-c', '--channel', type=str, default='#redditnet',
                        help='IRC channel to join (default: #redditnet)')

    args = parser.parse_args()
    server = args.server
    port = args.port
    nickname = args.nickname
    channel = args.channel

    bot = DynamicHambotIRC(server, port, nickname, channel)
    bot.start()
