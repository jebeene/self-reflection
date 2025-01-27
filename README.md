# Real-Time Facial Emotion Analysis with LED Control

![Project Logo](https://github.com/yourusername/emotion_led_project/blob/main/logo.png)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contributing](#contributing)

## Overview

This project is a **real-time facial emotion analysis tool** that controls LED lights based on detected emotions. It runs on a small computer like a **Raspberry Pi** and uses a **webcam** for capturing video. The system analyzes the user's facial expressions and changes the LED colors accordingly:

- **Angry**: Red
- **Sad**: Blue
- **Happy**: Green
- **Surprise**: Yellow
- **Neutral**: LEDs Off

## Features

- **Real-Time Emotion Detection**: Utilizes a TensorFlow Lite model for efficient emotion recognition.
- **LED Control**: Changes LED colors based on the detected emotion.
- **Modular Design**: Organized codebase for easy maintenance and scalability.
- **Logging**: Records detections and system behavior for debugging purposes.
- **Unit Testing**: Ensures the reliability of core components.

## Hardware Requirements

- **Raspberry Pi** (any model with GPIO pins, preferably Raspberry Pi 3 or later)
- **Webcam** (compatible with Raspberry Pi)
- **LEDs** (Red, Blue, Green, Yellow)
- **Resistors** (220Ω recommended for each LED)
- **Breadboard and Jumper Wires**
- **Power Supply** for Raspberry Pi

### **LED Wiring Diagram**

![LED Wiring Diagram](https://github.com/yourusername/emotion_led_project/blob/main/led_wiring_diagram.png)

**Pin Configuration:**

- **Red LED**: GPIO 17
- **Blue LED**: GPIO 27
- **Green LED**: GPIO 22
- **Yellow LED**: GPIO 10

*Ensure that each LED is connected with a current-limiting resistor to prevent damage.*

## Software Requirements

- **Operating System**: Raspberry Pi OS (formerly Raspbian) or any compatible Linux distribution
- **Python**: Version 3.7 or higher
- **Libraries**:
  - OpenCV (`opencv-python`)
  - TensorFlow (`tensorflow`)
  - NumPy (`numpy`)
  - RPi.GPIO (`RPi.GPIO`)

## Setup Instructions

### 1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/emotion_led_project.git
cd emotion_led_project
