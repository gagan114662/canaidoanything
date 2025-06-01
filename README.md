# Garment Creative AI 🎨

Transform ugly garment photos into professional creative imagery using state-of-the-art AI models.

## Features

- **FLUX 1.1 Integration** - Advanced image generation and style transfer
- **SAM 2 Background Removal** - Precise object segmentation and background removal
- **Real-ESRGAN Upscaling** - High-quality image enhancement and upscaling
- **FastAPI Backend** - High-performance REST API
- **Celery Task Queue** - Asynchronous background processing
- **Docker Support** - Easy deployment with Docker and docker-compose
- **GPU Acceleration** - CUDA support for faster processing

## Quick Start

### Option 1: Local Installation

1. **Clone and install:**
   ```bash
   git clone <your-repo>
   cd garment-creative-ai
   chmod +x scripts/*.sh
   ./scripts/install.sh
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start Redis:**
   ```bash
   # Ubuntu/Debian
   sudo systemctl start redis-server
   
   # macOS
   brew services start redis
   
   # Docker
   docker run -d -p 6379:6379 redis:alpine
   ```

4. **Start the application:**
   ```bash
   ./scripts/start.sh
   ```

### Option 2: Docker Deployment

1. **Using docker-compose:**
   ```bash
   cp .env.example .env
   # Edit .env with your HuggingFace token
   docker-compose up -d
   ```

2. **For GPU support:**
   ```bash
   # Build with GPU Dockerfile
   docker build -f docker/Dockerfile.gpu -t garment-ai-gpu .
   
   # Run with GPU support
   docker run --gpus all -p 8000:8000 garment-ai-gpu
   ```

## API Usage

### Process Garment Image

```bash
curl -X POST "http://localhost:8000/api/v1/process-garment" \
  -F "file=@your-garment-image.jpg" \
  -F "style_prompt=professional fashion photography, studio lighting, elegant" \
  -F "negative_prompt=blurry, low quality, amateur" \
  -F "enhance_quality=true" \
  -F "remove_background=false" \
  -F "upscale=true"
```

### Check Task Status

```bash
curl "http://localhost:8000/api/v1/status/{task_id}"
```

### Download Result

```bash
curl -O "http://localhost:8000/api/v1/download/{task_id}"
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `false` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `HF_TOKEN` | HuggingFace API token | Required for some models |
| `MAX_FILE_SIZE` | Maximum upload file size | `10485760` (10MB) |
| `MAX_IMAGE_SIZE` | Maximum image dimension | `2048` |
| `UPSCALE_FACTOR` | Image upscaling factor | `4` |

### Model Configuration

The application uses these AI models:

- **FLUX 1.1**: `black-forest-labs/FLUX.1-dev`
- **SAM 2**: `facebook/sam2-hiera-large`
- **Real-ESRGAN**: `ai-forever/Real-ESRGAN`

## API Endpoints

### Health Check
- `GET /api/v1/health` - Basic health check
- `GET /api/v1/health/detailed` - Detailed health with system metrics

### Image Processing
- `POST /api/v1/process-garment` - Process garment image
- `GET /api/v1/status/{task_id}` - Get task status
- `GET /api/v1/download/{task_id}` - Download processed image
- `DELETE /api/v1/task/{task_id}` - Cancel task
- `GET /api/v1/tasks` - List active tasks

### Monitoring
- Flower UI: `http://localhost:5555` (Celery task monitoring)
- API Documentation: `http://localhost:8000/docs`

## Processing Pipeline

1. **Input Validation** - Check file format, size, and quality
2. **Background Removal** (optional) - Use SAM 2 for precise segmentation
3. **Style Transfer** - Apply FLUX 1.1 for creative transformation
4. **Quality Enhancement** - Use Real-ESRGAN for upscaling and enhancement
5. **Output Generation** - Save processed image with metadata

## Performance Tips

### GPU Acceleration
- Install CUDA toolkit and GPU drivers
- Use `docker/Dockerfile.gpu` for Docker GPU support
- Set `CUDA_VISIBLE_DEVICES` environment variable

### Memory Management
- Models are loaded on-demand and can be unloaded to save memory
- Adjust Celery worker concurrency based on available RAM/VRAM
- Use image resizing to reduce memory usage for large images

### Scaling
- Run multiple Celery workers for parallel processing
- Use Redis cluster for high-availability task queue
- Load balance multiple API instances behind a proxy

## Development

### Project Structure
```
garment-creative-ai/
├── app/
│   ├── api/endpoints/          # FastAPI route handlers
│   ├── core/                   # Configuration and settings
│   ├── models/                 # Pydantic models and schemas
│   ├── services/               # Business logic and AI services
│   │   ├── ai/                 # AI model integrations
│   │   └── tasks/              # Celery tasks
│   ├── utils/                  # Utility functions
│   └── main.py                 # FastAPI application
├── docker/                     # Docker configurations
├── scripts/                    # Setup and utility scripts
├── tests/                      # Test files
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker compose configuration
└── README.md                   # This file
```

### Adding New Models

1. Create service class in `app/services/ai/`
2. Implement `load_model()`, `process()`, and `unload_model()` methods
3. Add model to processing pipeline in `app/services/tasks/image_processing.py`
4. Update configuration in `app/core/config.py`

### Running Tests

```bash
pytest tests/
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `MAX_IMAGE_SIZE`
   - Lower Celery worker concurrency
   - Enable model CPU offloading

2. **Model Download Fails**
   - Check internet connection
   - Verify HuggingFace token
   - Ensure sufficient disk space

3. **Redis Connection Error**
   - Verify Redis is running
   - Check `REDIS_URL` configuration
   - Ensure Redis port is not blocked

4. **Slow Processing**
   - Enable GPU acceleration
   - Reduce image resolution
   - Optimize model parameters

### Logs

- Application logs: `logs/app.log`
- Celery logs: Check Docker logs or console output
- Redis logs: Check Redis server logs

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- [FLUX](https://github.com/black-forest-labs/flux) for image generation
- [SAM 2](https://github.com/facebookresearch/segment-anything-2) for segmentation
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for image enhancement
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Celery](https://docs.celeryproject.org/) for task processing