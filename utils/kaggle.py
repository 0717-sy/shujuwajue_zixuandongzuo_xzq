import os
import kagglehub
    
# Download latest version
path = kagglehub.dataset_download("geyuanzhu/jaychou-lyrics")
    
print("Path to dataset files:", path)
