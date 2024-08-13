import os
import re
from pathlib import Path

# List of common 2-letter and 3-letter words that should remain capitalized
common_words = [
    "Sun", "Red", "Old", "New", "Big", "Sky", "Sea", "Man", "Dog", "Cat",
    "One", "Two", "Day", "Job", "Box", "Cup", "Pen", "Map", "Art", "Act",
    "Cap", "Top", "Pop", "Set", "Lot", "Run", "Net", "Key", "Fit", "Win",
    "Tie", "Gas", "Bag", "Use", "Way", "Fan", "Hat", "Tax", "Fix", "Buy",
    "Oak", "The", "Sol", "Mai", "Sax",
    # Adding common 2-letter words
    "An", "As", "At", "By", "He", "If", "In", "Is", "It", "My", "Of", "On", "Or", "To", "Up", "Us", "We"
]

# List of suffixes to remove with different capitalizations
suffixes_to_remove = [
    r'LLC', r'L\.L\.C', r'Llc', r'LP', r'L\.P', r'Lp', r'Inc', r'INC', r'Corp', r'CORP', r'CO', r'Ltd', r'LTD',
    r'LLC\.', r'L\.L\.C\.', r'LLC,', r'Inc\.', r'INC\.', r'Corp\.', r'CORP\.', r'CO\.', r'LTD\.',
    r' LLC', r' Llc', r' L.L.C', r' LP', r' Lp', r' NC', r' INC.', r' INC', r' CORP.', r' CORP',
    r' CO.', r' CO', r' LTD.', r' LTD'
]


def proper_case_with_exceptions(word):
    """Convert a word to all caps if it is 1-2-3 letters and not in the common list, else capitalize."""
    if len(word) in [1, 2, 3] and word.capitalize() not in common_words:
        return word.upper()
    return word.capitalize()


def rename_files_in_directory():
    current_directory = Path('.')
    for file in current_directory.iterdir():
        if file.suffix in ['.xls', '.xlsx']:
            # Remove commas and stray punctuation
            name = re.sub(r'[^\w\s]', '', file.stem)

            # Convert to proper case with exceptions
            name_parts = name.split()
            name_parts = [proper_case_with_exceptions(part) for part in name_parts]
            name = "_".join(name_parts)

            # Remove any unwanted suffixes (allowing for spaces or underscores before them)
            for suffix in suffixes_to_remove:
                name = re.sub(rf'[_\s]*{suffix}\b', '', name, flags=re.IGNORECASE)

            # Remove any trailing underscores or stray characters
            name = re.sub(r'_+$', '', name)

            # Remove any extra spaces or underscores from double replacements
            name = re.sub(r'\s+', '_', name).strip('_')
            name = re.sub(r'_+', '_', name)

            # Rename the file
            new_file_name = f"{name}{file.suffix}"
            new_file_path = current_directory / new_file_name
            file.rename(new_file_path)
            print(f"Renamed '{file.name}' to '{new_file_name}'")


if __name__ == "__main__":
    rename_files_in_directory()
