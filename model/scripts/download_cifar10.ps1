# CIFAR-10 Dataset Auto Downloader

$dataDir = "datasets/cifar-10"
$url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
$tarFile = "$dataDir/cifar-10-binary.tar.gz"

Write-Host "=== CIFAR-10 Dataset Auto Download ===" -ForegroundColor Green

# Create directory
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

Write-Host "Downloading CIFAR-10 dataset..." -ForegroundColor Green
Write-Host "URL: $url" -ForegroundColor Cyan

# Download
if (!(Test-Path $tarFile)) {
    try {
        Invoke-WebRequest -Uri $url -OutFile $tarFile -UseBasicParsing
        Write-Host "Download completed: $tarFile" -ForegroundColor Green
    } catch {
        Write-Host "Download error: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Already downloaded: $tarFile" -ForegroundColor Yellow
}

# Check archive size
$fileSize = (Get-Item $tarFile).Length
Write-Host "File size: $([math]::Round($fileSize/1MB, 2)) MB" -ForegroundColor Cyan

# Extract with 7-Zip or tar.exe
Write-Host "Extracting archive..." -ForegroundColor Green

if (Get-Command "7z" -ErrorAction SilentlyContinue) {
    # Use 7-Zip
    Write-Host "Extracting with 7-Zip..." -ForegroundColor Cyan
    & 7z x $tarFile -o"$dataDir" -y | Out-Null
    & 7z x "$dataDir/cifar-10-binary.tar" -o"$dataDir" -y | Out-Null
    Remove-Item "$dataDir/cifar-10-binary.tar" -Force -ErrorAction SilentlyContinue
} elseif (Get-Command "tar" -ErrorAction SilentlyContinue) {
    # Use Windows 10/11 tar.exe
    Write-Host "Extracting with Windows tar..." -ForegroundColor Cyan
    Push-Location $dataDir
    & tar -xzf (Split-Path $tarFile -Leaf)
    Pop-Location
} else {
    Write-Host "Error: 7-Zip or Windows tar is required" -ForegroundColor Red
    Write-Host "Please manually extract $tarFile" -ForegroundColor Yellow
    Write-Host "Ensure cifar-10-batches-bin folder exists in $dataDir" -ForegroundColor Yellow
    exit 1
}

# Check extraction result
$batchDir = "$dataDir/cifar-10-batches-bin"
if (Test-Path $batchDir) {
    Write-Host "CIFAR-10 dataset preparation completed!" -ForegroundColor Green
    Write-Host "Data location: $batchDir" -ForegroundColor Cyan
    
    # Display file list
    Write-Host "`n=== Data Files ===" -ForegroundColor Green
    $files = Get-ChildItem $batchDir -Name
    foreach ($file in $files) {
        $fullPath = Join-Path $batchDir $file
        $size = (Get-Item $fullPath).Length
        Write-Host "  $file ($([math]::Round($size/1KB, 1)) KB)" -ForegroundColor White
    }
    
    Write-Host "`nReady! You can start training:" -ForegroundColor Green
    Write-Host "  cargo run -- train --dataset cifar10 --epochs 10 --batch-size 32" -ForegroundColor Cyan
} else {
    Write-Host "Error: Extraction failed. Please check manually." -ForegroundColor Red
    Write-Host "Expected path: $batchDir" -ForegroundColor Yellow
}