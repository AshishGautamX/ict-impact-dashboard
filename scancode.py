import os
from pathlib import Path

def should_skip_folder(folder_name):
    """Check if folder should be skipped"""
    skip_folders = {
        'node_modules',
        '.git',
        '__pycache__',
        '.vscode',
        'dist',
        'build',
        '.next',
        'venv',        # common virtualenv folder
        '.idea'        # common IDE folder
    }
    return folder_name in skip_folders

def should_include_file(file_path):
    """Check if file should be included"""
    # Include these extensions
    include_extensions = {
        '.tsx', '.ts', '.jsx', '.js',
        '.html', '.css', '.scss', '.sass',
        '.json', '.md', '.txt',
        '.py', '.yml', '.yaml',
        '.env.example', '.gitignore',
        '.bib', '.tex'   # <-- added bib and tex support
    }

    # Skip these specific files
    skip_files = {
        'package-lock.json',
        '.DS_Store'
    }

    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_path)[1]

    if file_name in skip_files:
        return False

    return file_ext in include_extensions or file_name in ['.gitignore', '.env.example']

def read_file_content(file_path):
    """Read file content with error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            return f"[Error reading file: {str(e)}]"
    except Exception as e:
        return f"[Error reading file: {str(e)}]"

def scan_and_create_text_file():
    """Main function to scan folders and create output file"""

    # Get the current directory (where the script is located / run)
    current_dir = os.getcwd()
    output_file = os.path.join(current_dir, 'scannedcode.txt')

    # determine script filename safely (works even if __file__ not available)
    try:
        script_name = os.path.basename(__file__)
    except NameError:
        script_name = ''

    print(f"Scanning directory: {current_dir}")
    print(f"Output file will be: {output_file}")

    file_count = 0

    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write(f"full code of the project\n")
        out_f.write(f"Generated from: {current_dir}\n")
        out_f.write(f"{'=' * 80}\n\n")

        # Walk through all directories
        for root, dirs, files in os.walk(current_dir):
            # Remove folders to skip from dirs list (modifies in-place)
            dirs[:] = [d for d in dirs if not should_skip_folder(d)]

            # Process each file
            for file in sorted(files):
                file_path = os.path.join(root, file)

                # Skip the output file itself and the script
                if file == os.path.basename(output_file) or (script_name and file == script_name):
                    continue

                if should_include_file(file_path):
                    file_count += 1

                    # Write file header
                    out_f.write("\n" + "=" * 80 + "\n")
                    out_f.write(f"FILE: {file_path}\n")
                    out_f.write("=" * 80 + "\n\n")

                    # Write file content
                    content = read_file_content(file_path)
                    out_f.write(content)
                    out_f.write("\n\n")

                    print(f"Processed: {file_path}")

        # Write summary at the end
        out_f.write("\n" + "=" * 80 + "\n")
        out_f.write(f"Total files processed: {file_count}\n")
        out_f.write("=" * 80 + "\n")

    print(f"\n✓ Complete! Processed {file_count} files.")
    print(f"✓ Output saved to: {output_file}")

if __name__ == "__main__":
    scan_and_create_text_file()
