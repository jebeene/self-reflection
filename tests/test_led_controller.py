# tests/test_led_controller.py

import unittest
from unittest.mock import patch
from led_controller.controller import LEDController
import config

class TestLEDController(unittest.TestCase):
    @patch('led_controller.controller.GPIO')
    def setUp(self, mock_gpio):
        self.led_pins = config.LED_PINS
        self.controller = LEDController(self.led_pins)

    @patch('led_controller.controller.GPIO')
    def test_set_color_valid(self, mock_gpio):
        color = 'red'
        self.controller.set_color(color)
        mock_gpio.output.assert_any_call(self.led_pins['red'], mock_gpio.HIGH)

    @patch('led_controller.controller.GPIO')
    def test_set_color_off(self, mock_gpio):
        self.controller.set_color('off')
        # Ensure all LEDs are turned off
        for pin in self.led_pins.values():
            if pin is not None:
                mock_gpio.output.assert_any_call(pin, mock_gpio.LOW)

    @patch('led_controller.controller.GPIO')
    def tearDown(self, mock_gpio):
        self.controller.cleanup()
        mock_gpio.cleanup.assert_called_once()

if __name__ == '__main__':
    unittest.main()
