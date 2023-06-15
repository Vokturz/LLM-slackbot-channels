import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os

class SlackBot:
    def __init__(self, event_handlers={},  command_handlers={}, 
                 name='LLM-QA') -> None:
        """
        Initialize a new SlackBot instance.
        
        Raises:
            ValueError: If the SLACK_BOT_TOKEN or SLACK_APP_TOKEN 
                        environment variables could not be found.
        """
        self.name = name
        try:
            self.bot_token = os.environ['SLACK_BOT_TOKEN']
            self.app_token = os.environ['SLACK_APP_TOKEN']
        except KeyError as e:
            raise ValueError("Could not find required environment variable") from e

        self.app =  App(token=self.bot_token, name=self.name)
        self.socket = SocketModeHandler(self.app, self.app_token)

    def get_app(self) -> App:
        return self.app
    
    def get_bot_token(self) -> str:
        return self.bot_token
    
    def start(self) -> None:
        """
        Start the SlackBot instance.

        Raises:
            RuntimeError: If the SocketModeHandler could not start.
        """
        try:
            self.socket.start()
        except Exception as e:
            raise RuntimeError("Could not start the SocketModeHandler") from e

    # extend Slack Bolt decorators
    def event(self, event_name):
        def decorator(handler):
            self.app.event(event_name)(handler)
            return handler
        return decorator

    def command(self, command_name):
        def decorator(handler):
            self.app.command(command_name)(handler)
            return handler
        return decorator

    def shortcut(self, shortcut_name):
        def decorator(handler):
            self.app.shortcut(shortcut_name)(handler)
            return handler
        return decorator

    def action(self, action_name):
        def decorator(handler):
            self.app.action(action_name)(handler)
            return handler
        return decorator

    def view(self, callback_id):
        def decorator(handler):
            self.app.view(callback_id)(handler)
            return handler
        return decorator

    def options(self, callback_id):
        def decorator(handler):
            self.app.options(callback_id)(handler)
            return handler
        return decorator

    def step(self, callback_id):
        def decorator(handler):
            self.app.step(callback_id)(handler)
            return handler
        return decorator
