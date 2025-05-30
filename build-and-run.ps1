# Build script with caching for Windows PowerShell

# Enable Docker BuildKit for better caching
$env:DOCKER_BUILDKIT = 1
$env:COMPOSE_DOCKER_CLI_BUILD = 1

Write-Host "Building DeepFace Docker image with dependency caching..." -ForegroundColor Green

# Build with cache from previous builds
docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful! Starting the application..." -ForegroundColor Green
    docker-compose up -d
    
    Write-Host ""
    Write-Host "Application is starting up..." -ForegroundColor Yellow
    Write-Host "API will be available at: http://localhost:5000" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To check logs: docker-compose logs -f deepface-api" -ForegroundColor Gray
    Write-Host "To stop: docker-compose down" -ForegroundColor Gray
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}
