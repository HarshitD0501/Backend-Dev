import os
import shutil

def organize_files(folder_path):
    """
    Scans the folder_path.
    Moves all .txt files to a 'Text_Files' folder.
    Moves all .jpg/.png files to an 'Image_Files' folder.
    """
    # Step 1: Define Proper Destinations (Absolute Paths!)
    # We want these folders INSIDE the scan folder, not near the script
    text_folder = os.path.join(folder_path, "Text_Files")
    img_folder  = os.path.join(folder_path, "Image_Files")

    # Step 2: Ensure they exist
    if not os.path.exists(text_folder):
        os.makedirs(text_folder)
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    
    # Step 2: Get a list of all files in the folder_path
    # Hint: os.listdir()
    files=os.listdir(folder_path)
    
    # Step 3: Loop through each file
    for filename in files:
        source_path = os.path.join(folder_path, filename)

        # Skip if it's a directory (don't move folders!)
        if os.path.isdir(source_path):
            continue

        # Step 4: Check extension and move automatically
        if filename.endswith(".txt"):
            shutil.move(source_path, text_folder)
            print(f"Moved {filename} -> Text_Files")
            
        elif filename.endswith(".jpg") or filename.endswith(".png"):
            shutil.move(source_path, img_folder)
            print(f"Moved {filename} -> Image_Files")

# Test Code
folder_path = input("Enter the folder path: ")
# file_type is no longer needed! automatic!
organize_files(folder_path)