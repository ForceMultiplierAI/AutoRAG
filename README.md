# AutoRAG 🚀: Intelligent Prompt Compression Proxy

AutoRAG is a high-performance streaming proxy for OpenAI-compatible APIs that provides intelligent system message compression and enhanced token management.

## ⭐ Key Features

- **Intelligent System Message Compression**: Automatically reduces system message token count
- **Caching Mechanism**: Intelligent caching of compressed system messages
- **Streaming Compatible**: Full OpenAI API streaming compatibility
- **Configurable Token Limits**: Customizable context window and buffer settings
- **Thinking and Reasoning Tag Injection**: Adds `<thinking>` tags to provide insight into model processing
- **System Tag Information**: Adds `<system>` tag to show progress and stats on compression, etc.

## LongBench V2
Here's the data converted to markdown tables:

| Category | Score | Count |
|----------|-------|-------|
| Overall | 34.9% | 63 |

**By Difficulty:**
| Category | Score | Count |
|----------|-------|-------|
| Easy | 39.1% | 23 |
| Hard | 32.5% | 40 |

**By Length:**
| Category | Score | Count |
|----------|-------|-------|
| Short | 41.4% | 29 |
| Medium | 33.3% | 21 |
| Long | 23.1% | 13 |

**By Domain:**
| Category | Score | Count |
|----------|-------|-------|
| Code Repository Understanding | 37.5% | 8 |
| Long In-context Learning | 44.4% | 9 |
| Long Structured Data | 0.0% | 5 |
| Long-dialogue History | 33.3% | 6 |
| Multi-Document QA | 41.2% | 17 |
| Single-Document QA | 33.3% | 18 |


## 🛠 How It Works

### Prompt Compression
- Uses LLMlingua-2 for intelligent prompt compression
- Dynamically adjusts compression rate to fit within token limits
- Preserves critical newline and structural information

### Caching Strategy
- Generates a unique hash for each system message
- Caches compressed messages to reduce redundant processing
- Only caches messages that fit within the token limit

### Prompt Output Format
The proxy adds two key XML tags to the streaming response:
- `<thinking>`: Shows internal processing steps
- `<system>`: System message compression details are streamed as informative messages

## 🚀 Quick Start

1. Put all of your content into `llm.txt`
2. Start the proxy


### Configuration Options
- `--url`: Target LLM API endpoint
- `--listen`: Local proxy listening port (default: 8000)
- `--limit`: Context window size in thousands of tokens (default: 32k)
- `--buffer`: Reserved tokens for response (default: 1024)


```bash
# Run the proxy
python autorag.py \
  --url http://your-llm-api.com/v1 \
  --listen 8000 \
  --limit 32 \ # for 32k token window
  --buffer 1024
```

3. You will get a new streaming web service at http://localhost:8000/v1
4. Run the client. It will automatically put

```bash
python rag_client.py 
```

# 

## 📝 Example Usage

```python
import aiohttp
import asyncio

async def chat_with_proxy():
    async with aiohttp.ClientSession() as session:
        messages = [
            {
                "role": "system", 
                "content": "Long system context that might exceed token limits..."
            },
            {
                "role": "user", 
                "content": "Your question here"
            }
        ]
        
        async with session.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "your-model",
                "messages": messages,
                "stream": True
            }
        ) as response:
            # Handle streaming response
```

## 🔍 Use Cases
- Large context management
- Multi-document reasoning
- Enterprise documentation processing
- Research paper analysis


## 📦 Installation
```bash
pip install aiohttp llmlingua transformers torch
```

## 🤝 Contributing
Contributions welcome! Please open issues or submit PRs on our GitHub repository.

## 📄 License

This project is licensed under the Business Source License 1.1 (BSL-1.1).

### Key License Terms
- Non-production use is permitted
- Commercial use requires a separate license
- After the Change Date (4 years from first release), the software converts to an open-source license

### Obtaining a Commercial License
To use AutoRAG for commercial purposes, you must:
- Purchase a commercial license
- Contact the licensing team for specific terms

### License Inquiries
For all licensing questions, including:
- Commercial use permissions
- Enterprise licensing
- Custom deployment options

**Contact:**
- Email: contact@forcemultiplier.ai
- Subject Line: AutoRAG Licensing Inquiry

### Important Notes
- Unauthorized commercial use is strictly prohibited
- Violations will result in immediate termination of usage rights
- Legal action may be pursued for unauthorized use
