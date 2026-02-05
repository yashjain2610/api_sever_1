# Project Workflow Diagrams

Quick reference for visual workflow understanding | [Home](README.md)

---

## 1️⃣ System Architecture

```mermaid
graph TD
    User([User])
    Frontend["Streamlit<br/>Frontend"]
    Backend["FastAPI<br/>Backend"]

    subgraph AI["🤖 AI Services"]
        Gemini["Gemini 2.0"]
        DallE["DALL-E 3"]
        GPT4["GPT-4 Vision"]
        CLIP["CLIP"]
    end

    subgraph DB["💾 Data & Storage"]
        S3["AWS S3"]
        Chroma["ChromaDB"]
        Milvus["Milvus"]
        Excel["Excel"]
        JSON["JSON"]
    end

    subgraph Web["Web Scrapers"]
        Scrape["Scrapers"]
    end

    User -->|Use| Frontend
    Frontend -->|Request| Backend
    Backend -->|Call| AI
    Backend -->|Store| DB
    Scrape -->|Feed| Backend
```

---

## 2️⃣ Image Generation Pipeline

```mermaid
sequenceDiagram
    actor User
    participant API
    participant Prompts
    participant DALLE
    participant S3

    User->>API: Upload Image
    API->>API: Identify Type
    API->>Prompts: Get Templates
    
    loop For Each Variation
        Prompts-->>API: Prompt
        API->>DALLE: Generate
        DALLE-->>API: Image URL
        API->>S3: Save
    end
    
    API-->>User: All URLs + ZIP
```

---

## 3️⃣ Catalog Generation

```mermaid
graph LR
    Input([Input]) --> Select{Marketplace?}
    
    Select -->|Amazon| Path1
    Select -->|Flipkart| Path2
    Select -->|Meesho| Path3
    
    Path1["📦 Cache Check"] --> Process
    Path2["📦 Cache Check"] --> Process
    Path3["📦 Cache Check"] --> Process
    
    Process["🔄 Process with AI"] --> Format["📊 Format Data"]
    Format --> Export["📤 Export Excel"]
    Export --> Done([Done])
```

---

## 4️⃣ Duplicate Detection

```mermaid
graph TD
    A["📸 Upload Image"] --> B["🔢 CLIP<br/>Embedding"]
    B --> C["🔍 Milvus<br/>Search"]
    C --> D{Found<br/>Match?}
    
    D -->|No| E["✅ Process<br/>New Image"]
    D -->|Yes| F{Score<br/>> 95%?}
    
    F -->|No| E
    F -->|Yes| G["⚠️ Duplicate<br/>Found!"]
    
    E --> H["📇 Index"]
    H --> Done([Complete])
    
    style Done fill:#90EE90
    style G fill:#FFB6C6
```

---

## 5️⃣ Vector Search & Storage

```mermaid
graph LR
    IMG["🖼️ Images"] --> CLIP["CLIP<br/>768-dim"]
    CLIP --> MILVUS["Milvus DB"]
    
    DESC["📝 Descriptions"] --> CHROMA["Chroma<br/>1536-dim"]
    CHROMA --> CHROMADB["ChromaDB"]
    
    MILVUS --> QUERY1["Image Query"]
    CHROMADB --> QUERY2["Text Query"]
    
    QUERY1 --> RESULTS["Results"]
    QUERY2 --> RESULTS
```

---

## 6️⃣ API Request Lifecycle

```mermaid
graph TD
    A["HTTP Request"] --> B["FastAPI<br/>Route"]
    B --> C["Validate<br/>Input"]
    C --> D{Valid?}
    
    D -->|No| E["❌ 400 Error"]
    D -->|Yes| F["Execute<br/>Logic"]
    
    F --> G["Call AI<br/>Services"]
    G --> H{Error?}
    
    H -->|Yes| I["Retry or<br/>Fallback"]
    H -->|No| J["Store<br/>Results"]
    
    I --> K["📨 Response"]
    E --> K
    J --> L["Index in DB"]
    L --> K
    
    K --> END(["200 OK"])
    
    style END fill:#90EE90
    style E fill:#FFB6C6
```

---

## 7️⃣ Marketplace Pipelines

```mermaid
graph TD
    DATA["Product Data"] --> SPLIT{Select<br/>Marketplace}
    
    SPLIT -->|Amazon| A["Title: 200 chars<br/>Desc: 2000 chars"]
    SPLIT -->|Flipkart| B["Title: 70 chars<br/>Desc: 500 chars"]
    SPLIT -->|Meesho| C["Title: 150 chars<br/>Desc: 1000 chars"]
    
    A --> EXCEL1["Excel 1"]
    B --> EXCEL2["Excel 2"]
    C --> EXCEL3["Excel 3"]
    
    EXCEL1 --> UPLOAD["Upload to S3"]
    EXCEL2 --> UPLOAD
    EXCEL3 --> UPLOAD
    
    UPLOAD --> FINISH([Done])
```

---

## 8️⃣ Error Recovery

```mermaid
graph TD
    FAIL["API Call<br/>Fails"] --> TYPE{Error<br/>Type?}
    
    TYPE -->|Timeout| R1["Retry<br/>x3"]
    TYPE -->|Rate Limit| R2["Wait 60s"]
    TYPE -->|Invalid Auth| R3["Alert"]
    
    R1 --> AGAIN{Works?}
    R2 --> AGAIN
    
    AGAIN -->|Yes| SUCCESS["✓ Continue"]
    AGAIN -->|No| FALLBACK["Use Cache"]
    
    FALLBACK --> SUCCESS
    R3 --> MANUAL["Manual Fix"]
    MANUAL --> SUCCESS
    
    style SUCCESS fill:#90EE90
```

---

## 9️⃣ User Journey

```mermaid
graph LR
    START(["Open<br/>App"]) --> CHOOSE{Action?}
    
    CHOOSE -->|Single Image| UPLOAD["Upload<br/>Image"]
    CHOOSE -->|Batch| BATCH["Upload +<br/>SKUs"]
    
    UPLOAD --> GENERATE["Generate"]
    BATCH --> GENERATE
    
    GENERATE --> PROCESS["System<br/>Processes"]
    PROCESS --> DONE{Success?}
    
    DONE -->|Yes| RESULT["Show<br/>Results"]
    DONE -->|No| ERROR["Error"]
    
    RESULT --> DOWNLOAD["Download"]
    ERROR --> RETRY["Retry"]
    
    DOWNLOAD --> END(["Done ✓"])
    RETRY --> END
```

---

## 1️⃣0️⃣ Similarity Detection

```mermaid
sequenceDiagram
    participant User1
    participant API
    participant CLIP
    participant DB as Milvus

    User1->>API: Upload Image A
    API->>CLIP: Convert
    CLIP-->>API: Vector
    API->>DB: Index
    
    rect rgb(200, 150, 255)
    note over User1,DB: Later...
    end
    
    participant User2
    User2->>API: Upload Image B
    API->>CLIP: Convert
    CLIP-->>API: Vector
    API->>DB: Find Similar
    DB-->>API: Image A<br/>Similarity: 96%
    API->>API: Check > 95%
    API-->>User2: Duplicate!
```

---

## 1️⃣1️⃣ Web Scraping

```mermaid
sequenceDiagram
    participant API
    participant Scraper
    participant Browser
    participant Website

    API->>Scraper: Scrape URLs
    Scraper->>Browser: Launch
    
    loop For Each Product
        Browser->>Website: Load Page
        Website-->>Browser: HTML
        Scraper->>Scraper: Parse & Extract
        Scraper->>Scraper: Sleep Random
    end
    
    Scraper->>API: Results
```

---

## 1️⃣2️⃣ Catalog Caching

```mermaid
graph TD
    INPUT["New Image"] --> CHECK{In<br/>Cache?}
    
    CHECK -->|Hit| LOAD["Load<br/>Cached Data"]
    CHECK -->|Miss| PROCESS["Call Gemini<br/>Extract Attributes"]
    
    PROCESS --> GPT["Call GPT<br/>Generate Description"]
    GPT --> SAVE["Save to<br/>Cache"]
    
    LOAD --> MERGE["Merge with<br/>Marketplace Rules"]
    SAVE --> MERGE
    
    MERGE --> EXCEL["Write<br/>to Excel"]
    EXCEL --> DONE([Done])
```

---

**[← Home](README.md)**
