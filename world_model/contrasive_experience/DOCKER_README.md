# Docker Setup for GUI Agent

This guide explains how to run the GUI Agent project in a Docker container with all necessary dependencies, including Playwright and Chromium.

## Prerequisites

1. **Docker**: Install Docker from [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
2. **Docker Compose**: Usually included with Docker Desktop
3. **nvidia-docker2** (for GPU support): Required if you want to use GPU acceleration

### Installing nvidia-docker2 (Optional, for GPU support)

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Quick Start

### Option 1: Interactive Menu

Simply run the helper script without arguments:

```bash
./docker-run.sh
```

This will show an interactive menu with options to build, start, stop, and enter the container.

### Option 2: Command Line

```bash
# Build the Docker image
./docker-run.sh build

# Start the container
./docker-run.sh start

# Enter the container shell
./docker-run.sh shell

# Stop the container
./docker-run.sh stop

# Rebuild everything
./docker-run.sh rebuild
```

### Option 3: Using Docker Compose Directly

```bash
# Build and start
docker-compose up -d

# Enter the container
docker exec -it gui-agent-container bash

# Stop the container
docker-compose down
```

## Running Your Application

Once inside the container:

```bash
# Navigate to the project directory (already set as working directory)
cd CoMEM-Agent-Inference

# Run your script
python run.py --your-arguments-here

# Or use the shell scripts
./run_agent.sh
```

## Project Structure

```
.
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── docker-run.sh          # Helper script for easy Docker management
├── DOCKER_README.md       # This file
├── requirements.txt       # Python dependencies
└── CoMEM-Agent-Inference/ # Main project directory
    ├── run.py
    ├── requirements_web.txt
    └── ...
```

## Features

- **Playwright Support**: All system dependencies for Chromium are pre-installed
- **Headless Browser**: Runs browser in headless mode (no GUI required)
- **GPU Support**: If nvidia-docker2 is installed, GPU will be available
- **Volume Mounting**: Project directory is mounted, so changes persist
- **Network Access**: Uses host network mode for easy access to services

## Troubleshooting

### Playwright Dependencies Error

If you still get Playwright dependency errors, rebuild the image:

```bash
./docker-run.sh rebuild
```

### GPU Not Detected

1. Verify nvidia-docker2 is installed: `docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi`
2. Check GPU is available on host: `nvidia-smi`
3. Restart Docker service: `sudo systemctl restart docker`

### Permission Denied

If you get permission errors:

```bash
# Add your user to docker group
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

### Container Memory Issues

If the browser crashes due to memory, increase shared memory in `docker-compose.yml`:

```yaml
shm_size: 4gb  # Increase from 2gb to 4gb
```

## Environment Variables

You can modify environment variables in `docker-compose.yml`:

- `NVIDIA_VISIBLE_DEVICES`: Control which GPUs to use
- `DISPLAY`: X11 display for GUI applications (if needed)

## Customization

### Modifying Python Dependencies

Edit `requirements.txt` or `CoMEM-Agent-Inference/requirements_web.txt`, then rebuild:

```bash
./docker-run.sh rebuild
```

### Changing CUDA Version

Edit the base image in `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04
```

### Adding System Packages

Add packages to the `RUN apt-get install` command in `Dockerfile`.

## Tips

1. **Development**: The project directory is mounted as a volume, so you can edit files outside the container and run them inside
2. **Persistence**: Use volumes for data you want to persist between container restarts
3. **Logs**: View container logs with `docker-compose logs -f`
4. **Resource Limits**: Add resource limits in `docker-compose.yml` if needed

## Support

For Docker-specific issues, check:
- Docker logs: `docker-compose logs`
- Container status: `docker ps -a`
- Image info: `docker images`
