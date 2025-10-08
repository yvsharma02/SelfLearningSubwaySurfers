#!/bin/bash
adb devices
echo "Please Wait for few seconds"
sleep 10
python src/player.py
