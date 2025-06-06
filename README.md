# Feddit Sentiment Analysis API

A production-ready microservice that provides sentiment analysis for comments from the Feddit API (fake Reddit). Built with FastAPI, this service analyzes comment sentiment and provides polarity scores and classifications.

## Features

- **Sentiment Analysis**: Analyzes comment sentiment using TextBlob with polarity scores (-1 to 1)
- **Comment Classification**: Classifies comments as positive or negative
- **Filtering & Sorting**: Filter by date range and sort by sentiment score
- **Caching**: In-memory caching for improved performance
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation
- **Testing**: Full test coverage with unit and integration tests
- **CI/CD**: GitHub Actions workflow with linting, testing, and security checks

## Quick Start

### Prerequisites

- Python 3.10+ (**Note**: For local development, Python 3.10 is recommended)
- Docker and Docker Compose
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/ttanida/feddit-sentiment-analysis.git
cd feddit-sentiment-analysis
```

### 2. Run with Docker Compose (Recommended)

This will start both the Feddit API and the Sentiment Analysis API:

```bash
docker-compose -f docker-compose.sentiment.yml up -d
```

Services will be available at:

- **Sentiment Analysis API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Feddit API**: http://localhost:8080

### 3. Alternative: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start Feddit API first (in separate terminal)
docker-compose -f docker-compose.yml up -d

# Run the sentiment analysis API
python main.py
```

## API Endpoints

### Main Endpoint

**GET** `/api/v1/subfeddits/{subfeddit_name}/sentiment`

Analyze sentiment for comments in a specific subfeddit.

#### Parameters

- `subfeddit_name` (path): Name of the subfeddit
- `limit` (query, optional): Max comments to analyze (1-100, default: 25)
- `start_date` (query, optional): Filter comments after date (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
- `end_date` (query, optional): Filter comments before date (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
- `sort_order` (query, optional): Sort by polarity score - 'asc' (most negative first) or 'desc' (most positive first, default)

#### Example Request

```bash
curl "http://localhost:8000/api/v1/subfeddits/Dummy%20Topic%201/sentiment?limit=3&sort_order=desc"
```

#### Example Response

```json
{
  "subfeddit": "Dummy Topic 1",
  "total_comments": 3,
  "comments": [
    {
      "id": "1",
      "username": "user_0",
      "text": "It looks great!",
      "created_at": 1627625921,
      "sentiment": {
        "polarity_score": 1,
        "classification": "positive"
      }
    },
    {
      "id": "3",
      "username": "user_2",
      "text": "Awesome.",
      "created_at": 1627633121,
      "sentiment": {
        "polarity_score": 1,
        "classification": "positive"
      }
    },
    {
      "id": "2",
      "username": "user_1",
      "text": "Love it.",
      "created_at": 1627629521,
      "sentiment": {
        "polarity_score": 0.5,
        "classification": "positive"
      }
    }
  ],
  "subfeddit_info": {
    "id": "1",
    "username": "admin_1",
    "title": "Dummy Topic 1",
    "description": "Dummy Topic 1"
  }
}
```

### Additional Endpoints

- **GET** `/api/v1/health` - Health check
- **GET** `/docs` - Interactive API documentation
- **GET** `/redoc` - Alternative API documentation

## Architecture

```
src/
├── api/           # FastAPI routes and endpoints
├── clients/       # HTTP clients (Feddit API)
├── config/        # Configuration management
├── models/        # Pydantic data models
├── sentiment/     # Sentiment analysis logic
├── services/      # Business logic layer
└── utils/         # Shared utilities (caching)
```

### Key Components

1. **FastAPI Application** (`main.py`): ASGI application with middleware and routing
2. **Sentiment Analyzer** (`src/sentiment/`): TextBlob-based sentiment analysis with caching
3. **Feddit Client** (`src/clients/`): Async HTTP client with retry logic
4. **Service Layer** (`src/services/`): Business logic combining sentiment analysis and data fetching
5. **API Layer** (`src/api/`): REST endpoints with validation and error handling

## Configuration

Environment variables can be set in a `.env` file:

```env
# API Configuration
API_TITLE="Feddit Sentiment Analysis API"
DEBUG=false

# Feddit API
FEDDIT_BASE_URL="http://localhost:8080"
FEDDIT_TIMEOUT=30
FEDDIT_MAX_RETRIES=3

# Application Settings
DEFAULT_COMMENT_LIMIT=25
MAX_COMMENT_LIMIT=100
CACHE_TTL_SECONDS=3600
```

## Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_sentiment_analyzer.py -v
```

### Code Quality

```bash
# Format code
black src tests

# Lint code
flake8 src tests

# Type checking
mypy src
```

## Deployment

### Docker

```bash
# Build image
docker build -t sentiment-analysis-api .

# Run container
docker run -p 8000:8000 -e FEDDIT_BASE_URL="http://host.docker.internal:8080" sentiment-analysis-api
```

### Production Considerations

1. **Environment Variables**: Set appropriate values for production
2. **Logging**: Configure structured logging for monitoring
3. **Monitoring**: Add health checks and metrics collection
4. **Security**: Use HTTPS, rate limiting, and authentication as needed
5. **Scaling**: Consider horizontal scaling with load balancers

## Testing Strategy

### Unit Tests

- Sentiment analyzer functionality
- Data model validation
- Utility functions

### Integration Tests

- API endpoint behavior
- Error handling
- Service integration

### CI/CD Pipeline

- Automated testing on multiple Python versions
- Code quality checks (linting, formatting, type checking)
- Security vulnerability scanning
- Docker image building and testing

## Performance

### Optimizations

- **Caching**: Sentiment results cached to avoid recomputation
- **Async Operations**: Non-blocking HTTP requests to Feddit API
- **Connection Pooling**: Efficient HTTP client with connection reuse
- **Batch Processing**: Efficient sentiment analysis for multiple comments

### Monitoring

- Health check endpoint for service monitoring
- Structured logging for debugging and analytics
- Request/response timing and error tracking

## API Examples

### Analyze Recent Comments

```bash
curl "http://localhost:8000/api/v1/subfeddits/Dummy%20Topic%201/sentiment?limit=5"
```

### Filter by Date Range

```bash
curl "http://localhost:8000/api/v1/subfeddits/Dummy%20Topic%201/sentiment?start_date=2021-08-01&end_date=2021-08-02"
```

### Sort by Sentiment Score (Most Negative First)

```bash
curl "http://localhost:8000/api/v1/subfeddits/Dummy%20Topic%201/sentiment?sort_order=asc&limit=10"
```

### Sort by Sentiment Score (Most Positive First)

```bash
curl "http://localhost:8000/api/v1/subfeddits/Dummy%20Topic%201/sentiment?sort_order=desc&limit=10"
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure code quality (`pytest && black src tests && flake8 src tests`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Support

For questions or issues:

1. Check the [API documentation](http://localhost:8000/docs) when running locally
2. Review the test cases for usage examples
3. Open an issue in the repository for bugs or feature requests
