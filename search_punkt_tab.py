import os

search_text = "punkt_tab"
root_dir = os.getcwd()

print(f"Searching for '{search_text}' in {root_dir} ...")

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".py") or file.endswith(".txt") or file.endswith(".json") or file.endswith(".cfg"):
            filepath = os.path.join(subdir, file)
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if search_text in content:
                        print(f"Found '{search_text}' in: {filepath}")
            except Exception as e:
                print(f"Cannot read {filepath}: {e}")
