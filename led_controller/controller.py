# led_controller/controller.py

# import RPi.GPIO as GPIO
import logging
import config

class LEDController:
    def __init__(self, led_pins):
        self.led_pins = led_pins
        # Set up GPIO
        GPIO.setmode(GPIO.BCM)
        for color, pin in self.led_pins.items():
            if pin is not None:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
        logging.info("LED Controller initialized with pins: " + str(self.led_pins))

    def set_color(self, color):
        try:
            # Turn off all LEDs first
            for clr, pin in self.led_pins.items():
                if pin is not None:
                    GPIO.output(pin, GPIO.LOW)
            # Turn on the specified LED
            pin = self.led_pins.get(color)
            if pin is not None:
                GPIO.output(pin, GPIO.HIGH)
                logging.info(f"LED set to {color}.")
            else:
                logging.info("LEDs turned off.")
        except Exception as e:
            logging.error(f"Error setting LED color: {e}")

    def cleanup(self):
        GPIO.cleanup()
        logging.info("GPIO cleanup done.")
