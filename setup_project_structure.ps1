# PowerShell Script to Create Final Project Structure for Banana Classification and Detection

# Define the base project directory (current directory)
$baseDir = Get-Location

# Define the project structure as a nested hashtable
$projectStructure = @{
    "dataset" = @{
        "original" = @("unripe", "ripe", "overripe")
        "augmented" = @{
            "images" = @("unripe", "ripe", "overripe")
            "labels" = @("unripe", "ripe", "overripe")
        }
        "yolov8" = @{
            "images" = @("train", "val", "test")
            "labels" = @("train", "val", "test")
        }
        "detectron2" = @()  # Will include JSON files later
    }
    "models" = @{
        "detectron2_output" = @()  # Placeholder for Detectron2 outputs
    }
}

# Function to create directories recursively
function Create-Directories {
    param (
        [string]$currentPath,
        [hashtable]$structure
    )

    foreach ($key in $structure.Keys) {
        $newPath = Join-Path $currentPath $key
        if (-Not (Test-Path $newPath)) {
            New-Item -Path $newPath -ItemType Directory -Force | Out-Null
            Write-Host "Created Directory: $newPath"
        } else {
            Write-Host "Directory already exists: $newPath"
        }

        # If the value is a hashtable, recurse
        if ($structure[$key] -is [hashtable]) {
            Create-Directories -currentPath $newPath -structure $structure[$key]
        }
        # If the value is an array, create subdirectories
        elseif ($structure[$key] -is [System.Array]) {
            foreach ($subDir in $structure[$key]) {
                $subPath = Join-Path $newPath $subDir
                if (-Not (Test-Path $subPath)) {
                    New-Item -Path $subPath -ItemType Directory -Force | Out-Null
                    Write-Host "Created Subdirectory: $subPath"
                } else {
                    Write-Host "Subdirectory already exists: $subPath"
                }
            }
        }
    }
}

# Create the main directory structure
Create-Directories -currentPath $baseDir -structure $projectStructure

# Create placeholder JSON files for Detectron2
$detectron2Dir = Join-Path $baseDir "dataset\detectron2"
$annotationsTrainPath = Join-Path $detectron2Dir "annotations_train.json"
$annotationsValPath = Join-Path $detectron2Dir "annotations_val.json"

# Function to create empty JSON files if they don't exist
function Create-Empty-File {
    param (
        [string]$filePath
    )
    if (-Not (Test-Path $filePath)) {
        New-Item -Path $filePath -ItemType File -Force | Out-Null
        Write-Host "Created File: $filePath"
    }
    else {
        Write-Host "File already exists: $filePath"
    }
}

Create-Empty-File -filePath $annotationsTrainPath
Create-Empty-File -filePath $annotationsValPath

# Create placeholder files in the project root
$placeholderFiles = @(
    "cnn_data_loader.py",
    "cnn_model.py",
    "cnn_train.py",
    "alexnet_train.py",
    "yolov8_train.py",
    "convert_yolo_to_coco.py",
    "detectron2_register.py",
    "detectron2_train.py",
    "streamlit_app.py",
    "requirements.txt"
)

foreach ($file in $placeholderFiles) {
    $filePath = Join-Path $baseDir $file
    if (-Not (Test-Path $filePath)) {
        New-Item -Path $filePath -ItemType File -Force | Out-Null
        Write-Host "Created File: $filePath"
    }
    else {
        Write-Host "File already exists: $filePath"
    }
}

# Create the 'models' directory and subdirectories (already created above)
# No need to create model files; they will be generated after training

Write-Host "Final Project Structure Setup Complete!"
