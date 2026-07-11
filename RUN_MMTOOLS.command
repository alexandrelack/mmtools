#!/bin/zsh
cd "$(dirname "$0")"

echo "Starting MMTools by Alexandre Lack..."
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install Python 3 first."
  read "dummy?Press Enter to close..."
  exit 1
fi

python3 -m streamlit run app.py

echo
read "dummy?Press Enter to close..."
