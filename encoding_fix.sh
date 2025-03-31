#!/bin/bash

# Script to fix UTF-16 encoding issues
echo "Checking and fixing file encodings..."

# Function to detect and convert encoding
fix_encoding() {
  local file=$1
  # Detect encoding
  encoding=$(file -I "$file" | awk -F= '{print $2}')
  
  echo "File: $file"
  echo "Current encoding: $encoding"
  
  # If not UTF-8, convert to UTF-8
  if [[ $encoding != "utf-8" ]]; then
    echo "Converting to UTF-8..."
    # Create a temporary file
    temp_file="${file}.utf8"
    # Convert
    iconv -f $(echo $encoding | tr -d ',') -t UTF-8 "$file" > "$temp_file"
    # Replace original file
    mv "$temp_file" "$file"
    echo "✓ Converted successfully"
  else
    echo "✓ Already UTF-8, no conversion needed"
  fi
  echo ""
}

# Check and fix requirements.txt
fix_encoding "requirements.txt"

# Check other potential problem files
for file in *.txt *.md *.py *.json; do
  if [ -f "$file" ]; then
    fix_encoding "$file"
  fi
done

echo "All done! Files should now be in UTF-8 encoding."
