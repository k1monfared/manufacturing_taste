#!/bin/bash
# Check progress of the cultural market simulation analysis

OUTPUT_FILE="/tmp/claude/-home-k1-public-manufacturing-taste/tasks/bd2ff9a.output"
LOG_FILE="/home/k1/public/manufacturing_taste/results/progress.log"

TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "=== Progress Check: $TIMESTAMP ===" >> "$LOG_FILE"

if [ -f "$OUTPUT_FILE" ]; then
    # Get last 30 lines to capture current experiment status
    echo "Recent output:" >> "$LOG_FILE"
    tail -30 "$OUTPUT_FILE" >> "$LOG_FILE"

    # Check if completed
    if grep -q "Done!" "$OUTPUT_FILE"; then
        echo "STATUS: COMPLETED" >> "$LOG_FILE"
        # Remove the cron job once done
        crontab -l | grep -v "check_progress.sh" | crontab -
        echo "Cron job removed (analysis complete)" >> "$LOG_FILE"
    else
        echo "STATUS: RUNNING" >> "$LOG_FILE"
    fi
else
    echo "Output file not found - task may have finished or failed" >> "$LOG_FILE"
fi

echo "" >> "$LOG_FILE"
