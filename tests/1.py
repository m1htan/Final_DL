import os

def print_tree_limited(startpath, prefix="", level=0, max_level=4):
    if level >= max_level:
        return

    try:
        items = sorted([
            f for f in os.listdir(startpath)
            if not f.startswith('.') and f not in ['venv', '.venv', '__pycache__', '.git', '.idea']
        ])
    except PermissionError:
        return

    for i, name in enumerate(items):
        path = os.path.join(startpath, name)
        connector = "└── " if i == len(items) - 1 else "├── "
        print(prefix + connector + name)

        if os.path.isdir(path):
            extension = "    " if i == len(items) - 1 else "│   "
            print_tree_limited(path, prefix + extension, level + 1, max_level)

# Gọi hàm
print_tree_limited(r"D:\Github\Final_DL", max_level=4)
