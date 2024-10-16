#!/bin/bash
set -e

if [ "$1" = "run" ]; then
    python3 pipeline_check.py --file_path "$FILE_PATH" --base_dir "$BASE_DIR" --source_lang "$SOURCE_LANG" --target_lang "$TARGET_LANG" --name "$VIDEO_NAME"
elif [ "$1" = "bash" ]; then
    exec /bin/bash
else
    exec "$@"
fi