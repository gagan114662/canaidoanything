# Usage Guidelines - Garment Creative AI

## Table of Contents
1. [Quick Start](#quick-start)
2. [API Usage](#api-usage)
3. [Quality Guidelines](#quality-guidelines)
4. [Performance Optimization](#performance-optimization)
5. [Brand Consistency](#brand-consistency)
6. [Safety and Ethics](#safety-and-ethics)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)

---

## Quick Start

### System Requirements
- **Minimum:** 8GB RAM, 4GB GPU memory
- **Recommended:** 16GB RAM, 8GB GPU memory
- **Operating System:** Linux (Ubuntu 20.04+), macOS, Windows 10+
- **Python:** 3.8+

### Installation
```bash
# Clone and install
git clone <repository-url>
cd garment-creative-ai
./scripts/install.sh

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start services
./scripts/start.sh
```

### First Transformation
```bash
# Upload and transform a model photo
curl -X POST "http://localhost:8000/api/v1/transform-model" \
  -F "file=@model_photo.jpg" \
  -F "style_prompt=professional fashion photography, studio lighting" \
  -F "num_variations=5"
```

---

## API Usage

### Authentication
Currently, no authentication is required for local development. For production deployment, implement API key authentication.

### Core Endpoints

#### 1. Transform Model Photo
```http
POST /api/v1/transform-model
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- style_prompt: Style description (required)
- negative_prompt: What to avoid (optional)
- num_variations: Number of variations 1-5 (default: 5)
- enhance_model: Enhance model appearance (default: true)
- optimize_garment: Optimize garment presentation (default: true)
- generate_scene: Generate professional scenes (default: true)
- quality_mode: fast|balanced|high (default: balanced)
- brand_name: Brand for consistency (optional)
- seed: Random seed for reproducibility (optional)
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/transform-model" \
  -F "file=@input_image.jpg" \
  -F "style_prompt=editorial fashion photography, dramatic lighting, artistic composition" \
  -F "negative_prompt=amateur, low quality, blurry" \
  -F "num_variations=3" \
  -F "quality_mode=high" \
  -F "brand_name=luxury_brand"
```

**Response:**
```json
{
  "transformation_id": "uuid-string",
  "status": "queued",
  "message": "Model transformation queued successfully",
  "estimated_time": 25,
  "created_at": "2024-12-01T10:00:00Z"
}
```

#### 2. Check Transformation Status
```http
GET /api/v1/transform-status/{transformation_id}
```

**Response:**
```json
{
  "transformation_id": "uuid-string",
  "status": "completed",
  "progress": 100,
  "message": "Transformation completed successfully",
  "variations": [...],
  "quality_scores": {
    "overall_average": 8.7,
    "variation_scores": [8.5, 8.9, 8.6, 8.8, 8.7]
  },
  "performance_metrics": {
    "meets_time_target": true,
    "total_processing_time": 28.5
  }
}
```

#### 3. Download Results
```http
# Download single variation
GET /api/v1/download-variation/{transformation_id}/{variation_index}

# Download all variations as ZIP
GET /api/v1/download-all/{transformation_id}
```

---

## Quality Guidelines

### Input Image Requirements

#### ✅ **Optimal Input Characteristics:**
- **Resolution:** 1024x1024 or higher
- **Format:** JPG, PNG, WebP
- **File size:** Under 10MB
- **Lighting:** Even, natural or studio lighting
- **Model visibility:** Full body or torso visible
- **Garment clarity:** Clear, well-lit garment details
- **Background:** Any (will be replaced)
- **Pose:** Natural, frontal or 3/4 view preferred

#### ❌ **Avoid These Input Issues:**
- Extremely low resolution (<512px)
- Heavy motion blur or camera shake
- Severe over/under exposure
- Multiple people in frame
- Heavily distorted or tilted images
- Extreme close-ups or distant shots

### Style Prompting Best Practices

#### **Effective Style Prompts:**
```
✅ Good Examples:
- "editorial fashion photography, dramatic lighting, high contrast, magazine quality"
- "commercial product photography, clean studio lighting, professional presentation"
- "lifestyle photography, natural outdoor lighting, candid elegance"
- "artistic fashion, creative composition, avant-garde styling"
```

#### **Style Prompt Structure:**
1. **Photography Type:** editorial, commercial, lifestyle, artistic
2. **Lighting Style:** dramatic, natural, studio, soft, rim lighting
3. **Quality Keywords:** professional, high-end, magazine quality
4. **Specific Elements:** composition, mood, environment
5. **Technical Aspects:** sharp focus, high resolution, detailed

#### **Negative Prompts (What to Avoid):**
```
✅ Effective Negative Prompts:
- "amateur, low quality, blurry, distorted, bad lighting"
- "cluttered background, distracting elements, poor composition"
- "oversaturated, artificial, fake-looking, unrealistic"
```

---

## Performance Optimization

### Quality Modes

#### **Fast Mode** (15-20 seconds)
- **Use Case:** Quick previews, batch processing
- **Quality:** Good (7.0-8.0/10)
- **Features:** Basic enhancement, simple backgrounds
- **Resource Usage:** Low

#### **Balanced Mode** (25-30 seconds) - **Recommended**
- **Use Case:** Production work, general usage
- **Quality:** High (8.0-9.0/10)
- **Features:** Full enhancement, professional backgrounds
- **Resource Usage:** Medium

#### **High Quality Mode** (35-45 seconds)
- **Use Case:** Premium campaigns, portfolio work
- **Quality:** Excellent (8.5-9.5/10)
- **Features:** Maximum enhancement, premium backgrounds
- **Resource Usage:** High

### Processing Time Expectations

| Number of Variations | Fast Mode | Balanced Mode | High Quality |
|--------------------|-----------|---------------|--------------|
| 1 variation        | 8-12s     | 15-20s        | 25-30s       |
| 3 variations       | 15-20s    | 25-30s        | 35-45s       |
| 5 variations       | 20-25s    | 30-40s        | 45-60s       |

### Hardware Recommendations

#### **Development/Testing:**
- CPU: 4+ cores
- RAM: 8GB
- GPU: 4GB VRAM (GTX 1660 or better)
- Storage: 50GB free space

#### **Production:**
- CPU: 8+ cores (Intel i7/AMD Ryzen 7)
- RAM: 16-32GB
- GPU: 8-12GB VRAM (RTX 3080/4070 or better)
- Storage: 100GB+ SSD

---

## Brand Consistency

### Setting Up Brand Guidelines

#### **Upload Brand Guidelines:**
```bash
curl -X POST "http://localhost:8000/api/v1/brand-guidelines/luxury_brand" \
  -H "Content-Type: application/json" \
  -d '{
    "brand_name": "luxury_brand",
    "color_palette": ["#000000", "#FFFFFF", "#C9B037"],
    "style_keywords": ["elegant", "sophisticated", "premium"],
    "composition_style": "minimalist",
    "lighting_preference": "soft_dramatic",
    "mood": "sophisticated"
  }'
```

#### **Brand Guideline Structure:**
```json
{
  "color_palette": ["#hex1", "#hex2", "#hex3"],
  "style_keywords": ["keyword1", "keyword2"],
  "composition_style": "minimalist|dynamic|classic",
  "lighting_preference": "soft|dramatic|natural",
  "mood": "sophisticated|energetic|serene|nostalgic",
  "typography_style": "serif|sans-serif|script",
  "exclusions": ["avoid1", "avoid2"]
}
```

### Brand Consistency Scoring
- **A Grade (90-100%):** Excellent brand alignment
- **B Grade (80-89%):** Good brand consistency
- **C Grade (70-79%):** Acceptable with minor adjustments
- **D Grade (60-69%):** Needs improvement
- **F Grade (<60%):** Poor brand alignment

---

## Safety and Ethics

### Content Guidelines

#### ✅ **Appropriate Use Cases:**
- Professional fashion photography
- E-commerce product imagery
- Marketing and advertising content
- Portfolio and artistic work
- Brand campaign development
- Social media content creation

#### ❌ **Prohibited Use Cases:**
- Deceptive or misleading representations
- Non-consensual image modification
- Identity theft or impersonation
- Inappropriate or adult content
- Copyright infringement
- Discriminatory content creation

### Privacy and Data Protection

#### **Data Handling:**
- Images processed temporarily only
- No permanent storage of user content
- Automatic cleanup after 24 hours
- No personal data collection
- GDPR compliant processing

#### **User Responsibilities:**
- Obtain proper model releases
- Respect copyright and licensing
- Ensure appropriate consent
- Follow platform terms of service
- Comply with local regulations

### Bias Mitigation

#### **Our Commitments:**
- Diverse training datasets
- Regular bias auditing
- Equal performance across demographics
- Cultural sensitivity in outputs
- Fallback mechanisms for edge cases

#### **User Best Practices:**
- Use inclusive language in prompts
- Test across diverse model types
- Review outputs for bias
- Report issues for improvement
- Consider representation in campaigns

---

## Troubleshooting

### Common Issues and Solutions

#### **Issue: Poor Quality Results**
**Symptoms:** Low quality scores, artifacts, unrealistic outputs
**Solutions:**
1. Use higher resolution input images (>1024px)
2. Improve lighting in source photos
3. Use more specific style prompts
4. Try "high" quality mode
5. Ensure clear garment visibility

#### **Issue: Slow Processing Times**
**Symptoms:** Processing takes longer than expected
**Solutions:**
1. Check system resources (GPU memory)
2. Use "fast" mode for quicker results
3. Reduce number of variations
4. Close other GPU-intensive applications
5. Consider upgrading hardware

#### **Issue: Inconsistent Brand Adherence**
**Symptoms:** Results don't match brand guidelines
**Solutions:**
1. Upload detailed brand guidelines
2. Use brand-specific keywords in prompts
3. Increase brand consistency level
4. Review and refine brand guidelines
5. Use brand consistency reports

#### **Issue: API Timeouts**
**Symptoms:** Requests timeout before completion
**Solutions:**
1. Increase timeout settings
2. Use async processing with status checks
3. Reduce image size or complexity
4. Check system resources
5. Implement retry mechanisms

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 400 | Invalid input format | Check file format and size |
| 413 | File too large | Reduce file size under 10MB |
| 429 | Rate limit exceeded | Wait and retry request |
| 500 | Processing error | Check logs, retry with different settings |
| 503 | Service unavailable | Check system resources, restart services |

### Performance Monitoring

#### **Key Metrics to Track:**
- Processing time per transformation
- Quality scores distribution
- Success/failure rates
- System resource usage
- User satisfaction scores

#### **Monitoring Tools:**
- Built-in analytics dashboard
- Prometheus metrics (if configured)
- Custom logging integration
- Performance profiling tools

---

## Examples

### Example 1: E-commerce Product Photography
```bash
# Transform casual model photo to clean product imagery
curl -X POST "http://localhost:8000/api/v1/transform-model" \
  -F "file=@casual_model.jpg" \
  -F "style_prompt=clean e-commerce photography, white background, professional lighting, product focused" \
  -F "negative_prompt=artistic, dramatic, cluttered, personal" \
  -F "num_variations=3" \
  -F "quality_mode=balanced"
```

### Example 2: Editorial Fashion Campaign
```bash
# Create high-fashion editorial variations
curl -X POST "http://localhost:8000/api/v1/transform-model" \
  -F "file=@model_portrait.jpg" \
  -F "style_prompt=editorial fashion photography, dramatic lighting, artistic composition, magazine quality, high contrast" \
  -F "negative_prompt=commercial, product focused, amateur, low quality" \
  -F "num_variations=5" \
  -F "quality_mode=high"
```

### Example 3: Brand-Consistent Campaign
```bash
# First, upload brand guidelines
curl -X POST "http://localhost:8000/api/v1/brand-guidelines/streetwear_brand" \
  -H "Content-Type: application/json" \
  -d '{
    "brand_name": "streetwear_brand",
    "color_palette": ["#FF6B6B", "#4ECDC4", "#45B7D1"],
    "style_keywords": ["urban", "edgy", "contemporary", "bold"],
    "composition_style": "dynamic",
    "mood": "energetic"
  }'

# Then transform with brand consistency
curl -X POST "http://localhost:8000/api/v1/transform-model" \
  -F "file=@street_model.jpg" \
  -F "style_prompt=urban streetwear photography, dynamic composition, contemporary style" \
  -F "brand_name=streetwear_brand" \
  -F "num_variations=4" \
  -F "quality_mode=balanced"
```

### Example 4: Batch Processing
```python
import requests
import asyncio

async def process_batch_transformations():
    files = ["model1.jpg", "model2.jpg", "model3.jpg"]
    transformation_ids = []
    
    # Submit all transformations
    for file in files:
        response = requests.post(
            "http://localhost:8000/api/v1/transform-model",
            files={"file": open(file, "rb")},
            data={
                "style_prompt": "professional fashion photography",
                "num_variations": 3,
                "quality_mode": "balanced"
            }
        )
        transformation_ids.append(response.json()["transformation_id"])
    
    # Monitor progress
    while True:
        all_complete = True
        for tid in transformation_ids:
            status = requests.get(f"http://localhost:8000/api/v1/transform-status/{tid}")
            if status.json()["status"] != "completed":
                all_complete = False
        
        if all_complete:
            break
        
        await asyncio.sleep(5)
    
    # Download results
    for tid in transformation_ids:
        download_url = f"http://localhost:8000/api/v1/download-all/{tid}"
        # Process downloads...
```

---

## Support and Resources

### Documentation
- **API Reference:** `/api/v1/docs` (Swagger UI)
- **Model Cards:** `MODEL_CARDS.md`
- **Installation Guide:** `README.md`
- **Development Setup:** `CONTRIBUTING.md`

### Community and Support
- **Issues:** GitHub Issues for bug reports
- **Discussions:** GitHub Discussions for questions
- **Examples:** `/examples` directory
- **Tutorials:** `/docs/tutorials`

### Updates and Versioning
- **Current Version:** 1.0.0
- **Release Notes:** `CHANGELOG.md`
- **Migration Guides:** Available for major updates
- **Deprecation Policy:** 6-month notice for breaking changes

---

*Last Updated: December 2024*  
*Version: 1.0.0*  
*For the latest updates, visit our [documentation](http://localhost:8000/docs)*