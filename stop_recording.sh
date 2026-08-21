#!/bin/bash
# Clean stop for headless recording (no keyboard needed)
STOP_FILE="$HOME/weapon-detection-pipeline/STOP"
SERVICE="record-cameras.service"

if systemctl is-active --quiet "$SERVICE" 2>/dev/null; then
  echo "Stopping $SERVICE ..."
  sudo systemctl stop "$SERVICE"
  echo "Stopped."
else
  echo "Creating STOP file: $STOP_FILE"
  touch "$STOP_FILE"
  echo "Recorder will exit within ~1 second."
fi