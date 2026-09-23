"""
Fix encoding issues in chat.py - replace smart quotes with regular quotes
"""

# Read the corrupted file
with open('chat.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Replace all smart quotes and special characters with regular ones
replacements = {
    '“': '"',  # Left double quotation mark "
    '”': '"',  # Right double quotation mark "
    '‘': "'",  # Left single quotation mark '
    '’': "'",  # Right single quotation mark '
    '–': '-',  # En dash –
    '—': '-',  # Em dash —
    ' ': ' ',  # Non-breaking space
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Write back
with open('chat.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed all smart quotes and special characters in chat.py!')
print('You can now run: python chat.py')
