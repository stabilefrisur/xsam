import re

def flatten_dict(d, parent_key=""):
    """Recursively flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def make_windows_compliant_filename(name: str) -> str:
    """Sanitize a string to be a valid Windows file name."""
    # Remove invalid characters: \\ / : * ? " < > |
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    # Remove leading/trailing whitespace and dots
    name = name.strip().strip('.')
    # Limit length (255 is max for Windows, but keep shorter for safety)
    return name[:100]


def make_excel_compliant_sheetname(name: str) -> str:
    """Sanitize a string to be a valid Excel sheet name (<=31 chars, no \\ / ? * [ ] : )."""
    # Remove invalid characters: \\ / ? * [ ] :
    name = re.sub(r'[\\/?*\[\]:]', '_', name)
    # Remove leading/trailing whitespace and single quotes
    name = name.strip().strip("'")
    # Sheet names can't be empty
    if not name:
        name = 'Sheet1'
    return name[:31]