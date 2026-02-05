# 📚 Jewelry AI E-Commerce Documentation

Welcome! Visual workflow guides and comprehensive documentation for the jewelry AI catalog system.

---

## 🎯 Start Here

### **Workflow Diagrams** (Visual Learning - Recommended!)
[**📊 PROJECT_FLOW_DIAGRAMS.md**](00_PROJECT_FLOW_DIAGRAMS.md)
- 12 visual diagrams showing all workflows
- System architecture overview
- Image generation pipeline
- Catalog creation flow
- Web scraping workflow
- Duplicate detection logic
- Vector database storage
- API request lifecycle
- Error handling flows
- Marketplace-specific pipelines
- User journey walkthrough
- Similarity detection process

**⏱️ 10 minutes to understand all major flows**

---

## 📖 Detailed Documentation

### **1. Project Structure** (Big Picture)
[**01_PROJECT_STRUCTURE_GUIDE.md**](01_PROJECT_STRUCTURE_GUIDE.md)
- 30+ files explained with relationships
- Complete architecture overview
- How data flows through the system
- Technology stack breakdown
- Component interactions

**⏱️ 15 minutes for full understanding**

### **2. Code Explanation** (API Deep Dive)
[**02_CODE_EXPLANATION.md**](02_CODE_EXPLANATION.md)
- 9 FastAPI endpoints explained
- Step-by-step code walkthroughs
- Real code examples
- Common issues and fixes

**⏱️ 20 minutes to understand endpoints**

### **3. File Modification Guide** (How to Change Code)
[**03_FILE_MODIFICATION_GUIDE.md**](03_FILE_MODIFICATION_GUIDE.md)
- 14 real-world modification examples
- Exact line numbers for changes
- Before/after code comparisons
- Add new marketplace in 5 steps
- Change AI model configuration
- Modify prompt templates
- Adjust image generation settings

**⏱️ Use as reference when making changes**

### **4. Debugging Checklist** (Fixing Issues)
[**04_DEBUGGING_CHECKLIST.md**](04_DEBUGGING_CHECKLIST.md)
- 11+ debugging scenarios
- Systematic troubleshooting steps
- Checklists for common problems
- Debug code examples

**⏱️ Use when something breaks**

### **5. Testing Guide** (Writing Tests)
[**05_TESTING_GUIDE.md**](05_TESTING_GUIDE.md)
- 40+ unit test examples
- 25+ integration test examples
- Test fixtures and setup
- Testing best practices

**⏱️ Use when writing tests**

### **6. Recent Changes - February 2026** (Latest Updates)
[**06_RECENT_CHANGES_FEB2026.md**](06_RECENT_CHANGES_FEB2026.md)
- IP Metric for duplicate detection
- Batch-4 image generation for earrings
- Parallel execution (4x faster)
- Rate limit handling with retry
- Dual S3 bucket support (alya/blysk)

**⏱️ 10 minutes to understand all recent changes**

---

## ⚡ Quick Navigation

| Task | Go To | Time |
|------|-------|------|
| **See recent changes (Feb 2026)** | [Recent Changes](06_RECENT_CHANGES_FEB2026.md) | 10 min |
| **Understand workflows visually** | [Workflow Diagrams](00_PROJECT_FLOW_DIAGRAMS.md) | 10 min |
| **Understand full architecture** | [Project Structure](01_PROJECT_STRUCTURE_GUIDE.md) | 15 min |
| **Learn how endpoints work** | [Code Explanation](02_CODE_EXPLANATION.md) | 20 min |
| **Add new marketplace** | [Modifications](03_FILE_MODIFICATION_GUIDE.md) | 10 min |
| **Fix a bug** | [Debugging](04_DEBUGGING_CHECKLIST.md) | 10 min |
| **Write tests** | [Testing Guide](05_TESTING_GUIDE.md) | 15 min |
| **Change AI model** | [Modifications - Section 2](03_FILE_MODIFICATION_GUIDE.md) | 5 min |
| **Debug rate limits** | [Recent Changes - Rate Limit](06_RECENT_CHANGES_FEB2026.md#4-rate-limit-handling) | 5 min |

### By Component

| Component | Main Files | Guide |
|---|---|---|
| **API Server** | img_to_csv.py | [02 - CODE_EXPLANATION](02_CODE_EXPLANATION.md) |
| **AI Integration** | utils.py, openai_utils.py | [02 - CODE_EXPLANATION](02_CODE_EXPLANATION.md#section-3-endpoint-1) |
| **Image Processing** | image_search_engine.py | [01 - PROJECT_STRUCTURE](01_PROJECT_STRUCTURE_GUIDE.md#5-image-search-engine-530-lines) |
| **Database** | json_storage.py, Milvus, ChromaDB | [01 - PROJECT_STRUCTURE](01_PROJECT_STRUCTURE_GUIDE.md#-data-flow-diagrams) |
| **Excel Generation** | excel_fields.py | [03 - FILE_MODIFICATION](03_FILE_MODIFICATION_GUIDE.md#8️⃣-change-excel-output-format) |
| **Frontend** | jwellery_front.py, frontend.py | [01 - PROJECT_STRUCTURE](01_PROJECT_STRUCTURE_GUIDE.md#10-frontend-files-streamlit-ui) |
| **Web Scraping** | scraper.py, flp_scraper.py, myn_scraper.py | [01 - PROJECT_STRUCTURE](01_PROJECT_STRUCTURE_GUIDE.md#11-web-scrapers) |

---

## 🚀 Getting Started (For New Developers)

### Day 1: Understand the Basics
1. Read [01 - PROJECT_STRUCTURE_GUIDE.md](01_PROJECT_STRUCTURE_GUIDE.md) (15 minutes)
2. Skim through [02 - CODE_EXPLANATION.md](02_CODE_EXPLANATION.md) (10 minutes)
3. Set up environment variables (.env file)
4. Run the application locally

### Day 2: Explore the Code
1. Open img_to_csv.py and follow along with [02 - CODE_EXPLANATION.md](02_CODE_EXPLANATION.md)
2. Test a simple API endpoint using `/docs`
3. Study one specific endpoint (e.g., `/generate_caption`)
4. Trace the data flow from input → output

### Day 3: Make Your First Change
1. Choose a simple modification from [03 - FILE_MODIFICATION_GUIDE.md](03_FILE_MODIFICATION_GUIDE.md)
2. Follow the step-by-step instructions
3. Test your changes
4. Write a simple test for your change

### Day 4: Set Up Testing
1. Read [05 - TESTING_GUIDE.md](05_TESTING_GUIDE.md) sections 1-2
2. Create your first unit test
3. Run the test suite
4. Understand how mocking works

### Day 5+: Develop & Debug
1. Use [04 - DEBUGGING_CHECKLIST.md](04_DEBUGGING_CHECKLIST.md) when issues arise
2. Use [03 - FILE_MODIFICATION_GUIDE.md](03_FILE_MODIFICATION_GUIDE.md) for modifications
3. Reference [02 - CODE_EXPLANATION.md](02_CODE_EXPLANATION.md) for API details
4. Write tests using [05 - TESTING_GUIDE.md](05_TESTING_GUIDE.md) patterns

---

## 📊 Documentation Structure

```
DOCUMENTATION/
├── README.md                                    ← You are here
├── 00_PROJECT_OVERVIEW.md                      (Enhanced - Quick start!)
├── 00_PROJECT_FLOW_DIAGRAMS.md                 (Enhanced - 6 diagrams)
├── 01_PROJECT_STRUCTURE_GUIDE.md               (1008 lines)
├── 02_CODE_EXPLANATION.md                      (905 lines)
├── 03_FILE_MODIFICATION_GUIDE.md               (797 lines)
├── 04_DEBUGGING_CHECKLIST.md                   (961 lines)
├── 05_TESTING_GUIDE.md                         (1292 lines)
├── 06_RECENT_CHANGES_FEB2026.md                (NEW - Feb 2026 updates)
└── api_docs.pdf                                (API documentation PDF)

Total: 7,500+ lines of comprehensive documentation
```

---

## 🔑 Key Concepts

### Architecture Overview
```
User ↔ Streamlit UI ↔ FastAPI Server ↔ AI Services (Gemini, GPT-4)
                            ↓
                    Cloud Services:
                    - AWS S3 (images)
                    - Milvus (embeddings)
                    - ChromaDB (semantic search)
```

### Main Features
- **Image Upload** → AI analyzes jewelry images
- **Auto-Generate** → Creates product descriptions
- **Multi-Marketplace** → Formats for Amazon, Flipkart, Meesho
- **Duplicate Detection** → Finds similar images
- **Excel Export** → Ready-to-upload catalogs

### Technology Stack
| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit |
| **API** | FastAPI (Python) |
| **AI** | Google Gemini 2.5 Flash, GPT-Image 1.5 |
| **Storage** | AWS S3 (dual bucket: alyaimg, blyskimg) |
| **Vectors** | Milvus/Zilliz (IP metric, HNSW index) |
| **Embeddings** | CLIP (openai/clip-vit-base-patch32) |
| **Database** | JSON (local), PostgreSQL (optional) |

---

## 🐛 Common Issues Quick Reference

| Problem | Solution | Guide |
|---------|----------|-------|
| API key not found | Add to .env file | [04 - DEBUGGING](04_DEBUGGING_CHECKLIST.md#issue-api-key-not-found) |
| Gemini API fails | Check credentials and rate limit | [04 - DEBUGGING](04_DEBUGGING_CHECKLIST.md#-gemini-api-issues) |
| S3 upload fails | Verify AWS credentials | [04 - DEBUGGING](04_DEBUGGING_CHECKLIST.md#issue-failed-to-upload-to-s3) |
| Milvus won't connect | Start Docker container | [04 - DEBUGGING](04_DEBUGGING_CHECKLIST.md#issue-failed-to-initialize-milvus-connection) |
| Excel format wrong | Update excel_fields.py | [03 - FILE_MODIFICATION](03_FILE_MODIFICATION_GUIDE.md#8️⃣-change-excel-output-format) |
| Duplicate detection fails | Adjust threshold | [03 - FILE_MODIFICATION](03_FILE_MODIFICATION_GUIDE.md#4️⃣-change-duplicate-detection-threshold) |

---

## 📋 Checklists

### Pre-Deployment Checklist
- [ ] All tests passing (`pytest`)
- [ ] No syntax errors in modified files
- [ ] .env file has all required keys
- [ ] Database connections work
- [ ] API endpoints tested with sample data
- [ ] Excel output validated
- [ ] S3 bucket accessible
- [ ] No hardcoded credentials in code

### Code Review Checklist
- [ ] Changes follow existing code style
- [ ] Related tests added/updated
- [ ] Documentation updated
- [ ] No unnecessary dependencies added
- [ ] Error handling present
- [ ] Logging added for debugging
- [ ] Performance impact assessed

### Debugging Checklist
- [ ] Error message clear and descriptive?
- [ ] Issue reproducible consistently?
- [ ] Recent code changes related?
- [ ] Environment variables correct?
- [ ] External services running?
- [ ] File permissions okay?
- [ ] Disk space available?

---

## 🤝 Contributing

### How to Report Issues
1. Check [04 - DEBUGGING_CHECKLIST.md](04_DEBUGGING_CHECKLIST.md) first
2. Provide exact error message
3. Share steps to reproduce
4. Include environment details (Python version, OS)
5. Attach relevant code snippets

### How to Suggest Changes
1. Use [03 - FILE_MODIFICATION_GUIDE.md](03_FILE_MODIFICATION_GUIDE.md) as reference
2. Create tests for new functionality
3. Update documentation if needed
4. Follow existing code patterns

### How to Add Documentation
- Use the same markdown format
- Include code examples
- Add table of contents for long guides
- Keep explanations beginner-friendly

---

## 📞 Support Resources

### Internal
- 🗂️ **Project Files**: See [01 - PROJECT_STRUCTURE_GUIDE.md](01_PROJECT_STRUCTURE_GUIDE.md)
- 🔍 **API Reference**: See [02 - CODE_EXPLANATION.md](02_CODE_EXPLANATION.md)
- ⚙️ **Configuration**: See [01 - PROJECT_STRUCTURE_GUIDE.md#-environment-variables-env-file](01_PROJECT_STRUCTURE_GUIDE.md)

### External
- 📚 **FastAPI Docs**: https://fastapi.tiangolo.com/
- 🤖 **Google Gemini API**: https://ai.google.dev/
- 🧠 **Milvus Vector DB**: https://milvus.io/
- 💾 **ChromaDB**: https://www.trychroma.com/

---

## 📈 Learning Path

```
Beginner
  ↓
Read: 01_PROJECT_STRUCTURE_GUIDE.md
  ↓
Intermediate
  ↓
Read: 02_CODE_EXPLANATION.md
  ↓
Advanced
  ├─ Read: 03_FILE_MODIFICATION_GUIDE.md (Making changes)
  ├─ Read: 04_DEBUGGING_CHECKLIST.md (Debugging)
  └─ Read: 05_TESTING_GUIDE.md (Quality)
  ↓
Expert
  ↓
Contributing code and improvements
```

---

## ✅ Quick Wins (Low-Hanging Fruit)

If you want to contribute but don't know where to start:

1. **Add logging** [03 - FILE_MODIFICATION](03_FILE_MODIFICATION_GUIDE.md#7️⃣-add-logging-to-track-issues)
   - 30 minutes
   - Medium difficulty
   - Helps everyone debug

2. **Add input validation** [03 - FILE_MODIFICATION](03_FILE_MODIFICATION_GUIDE.md#1️⃣4️⃣-add-custom-marketplace-field-validation)
   - 45 minutes
   - Easy-Medium difficulty
   - Prevents errors

3. **Write unit tests** [05 - TESTING_GUIDE](05_TESTING_GUIDE.md)
   - Variable time
   - Medium difficulty
   - Improves code quality

4. **Improve error messages** [04 - DEBUGGING_CHECKLIST](04_DEBUGGING_CHECKLIST.md)
   - 20 minutes per error
   - Easy difficulty
   - Better user experience

---

## 📊 Stats

- **Total Documentation**: 5,963 lines
- **Number of Guides**: 5 comprehensive guides
- **Code Examples**: 100+
- **Debugging Scenarios**: 11+
- **Modification Examples**: 14+
- **Test Examples**: 65+

---

## 🎓 Learning Resources

### Understanding AI Integration
See [02 - CODE_EXPLANATION.md](#section-6-endpoint-3-generate_caption):
- How Gemini analyzes images
- How GPT-4 generates descriptions
- How CLIP detects duplicates

### Understanding Databases
See [01 - PROJECT_STRUCTURE_GUIDE.md](#-data-files-explanation):
- Milvus for vector embeddings
- ChromaDB for semantic search
- JSON for local caching
- S3 for cloud storage

### Understanding Web Architecture
See [01 - PROJECT_STRUCTURE_GUIDE.md](#-architecture-overview):
- FastAPI endpoints
- Async processing
- Error handling
- Rate limiting

---

## 🚀 Next Steps

1. **New to the project?** Start with [01 - PROJECT_STRUCTURE_GUIDE.md](01_PROJECT_STRUCTURE_GUIDE.md)
2. **Need to fix something?** Use [04 - DEBUGGING_CHECKLIST.md](04_DEBUGGING_CHECKLIST.md)
3. **Need to add a feature?** Use [03 - FILE_MODIFICATION_GUIDE.md](03_FILE_MODIFICATION_GUIDE.md)
4. **Want to improve code?** Use [05 - TESTING_GUIDE.md](05_TESTING_GUIDE.md)
5. **Deep dive into code?** Use [02 - CODE_EXPLANATION.md](02_CODE_EXPLANATION.md)

---

## 📝 Document Version

- **Created**: January 2025
- **Last Updated**: February 2026
- **Documentation Version**: 2.0
- **Compatible with**: Python 3.8+, FastAPI 0.95+

### Recent Updates (February 2026)
- Added IP metric for duplicate detection
- Added batch-4 image generation for earrings
- Added parallel execution (4x faster)
- Added rate limit handling with retry logic
- Added dual S3 bucket support (alya/blysk)
- See [06_RECENT_CHANGES_FEB2026.md](06_RECENT_CHANGES_FEB2026.md) for details

---

## 💡 Pro Tips

- 💾 Keep `.env` file private (add to .gitignore)
- 🧪 Run tests before pushing code
- 📝 Update docs when you change code
- 🔍 Use debugging checklist before asking for help
- 📚 Reference section links in your PRs
- 🚀 Use the quick reference tables
- 🎯 Follow the learning path for best results

---

**Happy coding! 🎉**

Questions? Check the relevant guide or search for your issue in [04 - DEBUGGING_CHECKLIST.md](04_DEBUGGING_CHECKLIST.md).

