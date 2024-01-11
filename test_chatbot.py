import unittest
from unittest.mock import MagicMock, patch
from chatbot import ChatBot

class TestChatBot(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock()
        self.bot = ChatBot(self.model)

    def test_process_input(self):
        with patch.object(self.bot, 'process_input', return_value='processed_input'):
            self.assertEqual(self.bot.process_input('user_input'), 'processed_input')

    def test_generate_response(self):
        with patch.object(self.bot, 'generate_response', return_value='response'):
            self.assertEqual(self.bot.generate_response('processed_input'), 'response')

    def test_start_conversation(self):
        with patch('builtins.input', side_effect=['user_input', 'quit']), \
             patch.object(self.bot, 'process_input', return_value='processed_input'), \
             patch.object(self.bot, 'generate_response', return_value='response'), \
             patch('builtins.print') as mock_print:
            self.bot.start_conversation()
            mock_print.assert_called_with('Bot: response')

if __name__ == '__main__':
    unittest.main()