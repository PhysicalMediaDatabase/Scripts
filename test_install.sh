#!/bin/bash

check_command() {
    if command -v "$1" &> /dev/null; then
        echo "$1 is installed."
    else
        echo "$1 is not installed."
    fi
}

check_command ffmpeg
check_command python3
check_command parallel