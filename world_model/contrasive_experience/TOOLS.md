# CoMEM-Agent Tools Reference

This document provides comprehensive documentation for all tools available in the CoMEM-Agent framework.

## Table of Contents

- [GUI Interaction Tools](#gui-interaction-tools)
  - [ClickTool](#clicktool)
  - [TypeTool](#typetool)
  - [SelectionTool](#selectiontool)
  - [ScrollTool](#scrolltool)
  - [WaitTool](#waittool)
  - [PressKeyTool](#presskeytool)
- [Navigation Tools](#navigation-tools)
  - [GoBackTool](#gobacktool)
  - [PageGotoTool](#pagegototool)
- [Analysis Tools](#analysis-tools)
  - [PageParserTool](#pageparsertool)
  - [ImageCheckerTool](#imagecheckertool)
  - [ContentAnalyzerTool](#contentanalyzertool)
  - [MapSearchTool](#mapsearchtool)
  - [GotoHomepageTool](#gotohomepagetool)
- [Web Search Tools](#web-search-tools)
  - [WebSearchTool](#websearchtool)
- [Task Completion Tools](#task-completion-tools)
  - [StopTool](#stoptool)

---

## GUI Interaction Tools

### ClickTool

**Location**: `tools/gui_tools.py:10`

**Description**: Click on an element described by its appearance, text, or location.

**Registration Name**: `click`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `element_id` | string | No | Numeric ID from the SoM (Set-of-Mark) legend. Must be a valid number without any other text. |
| `coords` | string | No | Coordinates in format `<point>x1 y1</point>`. Must contain two valid numbers. |
| `description` | string | Yes | The number label of the item and description of the element to click on. |
| `reasoning` | string | Yes | Reasoning for why this action is necessary. |

**Usage Example**:
```json
{
  "tool": "click",
  "args": {
    "element_id": "42",
    "coords": "<point>320 450</point>",
    "description": "Click on the 'Add to Cart' button",
    "reasoning": "Need to add the selected item to the shopping cart"
  }
}
```

**Notes**:
- The actual action execution happens in the agent's `_process_response` method
- Supports three grounding modes: element ID, pixel coordinates, or grounding model fallback
- Grounding fallback uses UI-Ins-7B model for 2-stage visual grounding

---

### TypeTool

**Location**: `tools/gui_tools.py:61`

**Description**: Type text into an input field described by its appearance or purpose.

**Registration Name**: `type`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to type into the input field. |
| `element_id` | string | No | Numeric ID from the SoM legend. Must be a valid number without other text. |
| `coords` | string | No | Coordinates in format `<point>x1 y1</point>`. |
| `field_description` | string | Yes | The number label and description of the input field. |
| `reasoning` | string | Yes | Reasoning for why this action is necessary. |

**Usage Example**:
```json
{
  "tool": "type",
  "args": {
    "text": "wireless headphones",
    "element_id": "15",
    "coords": "<point>640 120</point>",
    "field_description": "Search bar at the top of the page",
    "reasoning": "Need to search for wireless headphones"
  }
}
```

**Notes**:
- Automatically clicks on the field before typing
- Supports element ID or coordinate-based targeting

---

### SelectionTool

**Location**: `tools/gui_tools.py:116`

**Description**: Select an option from a dropdown menu.

**Registration Name**: `selection`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `element_id` | string | No | Numeric ID from the SoM legend. |
| `description` | string | Yes | Description of the dropdown menu (location, color, etc.). |
| `text` | string | Yes | The option to select from the dropdown menu. |
| `reasoning` | string | Yes | Reasoning for why this action is necessary. |

**Usage Example**:
```json
{
  "tool": "selection",
  "args": {
    "element_id": "28",
    "description": "Dropdown menu for sorting options",
    "text": "Price: Low to High",
    "reasoning": "Sort products by price to find cheapest option"
  }
}
```

---

### ScrollTool

**Location**: `tools/gui_tools.py:168`

**Description**: Scroll the page in specified direction.

**Registration Name**: `scroll`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `direction` | string (enum) | Yes | Direction to scroll: `up`, `down`, `left`, or `right`. |
| `reasoning` | string | Yes | Reasoning for why this action is necessary. |

**Usage Example**:
```json
{
  "tool": "scroll",
  "args": {
    "direction": "down",
    "reasoning": "Need to see more products below the fold"
  }
}
```

---

### WaitTool

**Location**: `tools/gui_tools.py:210`

**Description**: Wait for a specified amount of time (default 2 seconds).

**Registration Name**: `wait`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `reasoning` | string | Yes | Reasoning for why this action is necessary. |

**Usage Example**:
```json
{
  "tool": "wait",
  "args": {
    "reasoning": "Wait for the page to finish loading dynamic content"
  }
}
```

**Notes**:
- Default wait time is 2 seconds
- Useful for waiting for page loads, animations, or AJAX requests

---

### PressKeyTool

**Location**: `tools/gui_tools.py:246`

**Description**: Press a specific key.

**Registration Name**: `press_key`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `key` | string (enum) | Yes | The key to press: `enter`, `delete`, or `space`. |
| `reasoning` | string | Yes | Reasoning for why this action is necessary. |

**Usage Example**:
```json
{
  "tool": "press_key",
  "args": {
    "key": "enter",
    "reasoning": "Submit the search query"
  }
}
```

---

## Navigation Tools

### GoBackTool

**Location**: `tools/gui_tools.py:328`

**Description**: Navigate back to the previous page using browser history.

**Registration Name**: `go_back`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `reasoning` | string | Yes | Reasoning for why going back is necessary. |

**Usage Example**:
```json
{
  "tool": "go_back",
  "args": {
    "reasoning": "Return to search results to try a different product"
  }
}
```

---

### PageGotoTool

**Location**: `tools/gui_tools.py:358`

**Description**: Navigate to specific predefined pages based on user intent.

**Registration Name**: `goto_url`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `page_name` | string | Yes | The page name to navigate to (see supported pages below). |
| `reasoning` | string | Yes | Reasoning for why this navigation is necessary. |

**Supported Pages**:
| Page Name | URL | Description |
|-----------|-----|-------------|
| `ticket` | https://www.trip.com/flights/ | Flight ticket booking |
| `car` | https://sg.trip.com/carhire/... | Car rental service |
| `flight` | https://www.momondo.com/ | Flight search |
| `hotel` | https://sg.trip.com/hotels/... | Hotel booking |
| `shopping` | http://ec2-3-146-212-252...7770/ | Shopping website |
| `event` | https://www.eventbrite.com/ | Event search |
| `map` | https://www.google.com/maps | Google Maps |
| `youtube` | https://www.youtube.com/ | YouTube |
| `food` | https://www.timeout.com/ | Food/restaurant guide |
| `travel` | https://www.nomadicmatt.com/ | Travel guide |
| `dollars` | https://www.xe.com/ | Currency exchange |
| `twitter` | https://twitter.com/home | Twitter/X |
| `wiki` | https://www.wikipedia.org/ | Wikipedia |

**Usage Example**:
```json
{
  "tool": "goto_url",
  "args": {
    "page_name": "shopping",
    "reasoning": "Navigate to shopping website to find products"
  }
}
```

**Notes**:
- Page name matching is case-insensitive and supports partial matching
- Example: "car rental" → matches "car"

---

## Analysis Tools

### PageParserTool

**Location**: `tools/analysis_tools.py:20`

**Description**: Get content of the current web page using AsyncWebCrawler.

**Name**: `page_parser`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `reasoning` | string | Yes | Reasoning for why this parsing is necessary. |

**Returns**: Markdown-formatted page content extracted from the current URL.

**Usage Example**:
```python
page_content = await page_parser_tool.call(page=page)
```

**Notes**:
- Asynchronous method using `AsyncWebCrawler`
- Extracts clean, structured content from HTML
- Useful for getting text-based page information

---

### ImageCheckerTool

**Location**: `tools/analysis_tools.py:59`

**Description**: Get captions and analyze the first 5 images on the current page with AI-generated descriptions.

**Name**: `image_checker`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Query or context for image analysis (e.g., "What products are shown?"). |
| `reasoning` | string | Yes | Reasoning for why image analysis is necessary. |

**Returns**: JSON object containing:
- `page_url`: Current page URL
- `total_images_found`: Total number of images on page
- `analyzed_images`: Array of top 5 most relevant images with:
  - `index`: Image number (1-5)
  - `src`: Image URL
  - `alt`: Alt text
  - `title`: Title attribute
  - `width`, `height`: Dimensions
  - `caption`: Extracted caption from nearby text
  - `relevance_score`: CLIP-based relevance score (0-1)
  - `ai_description`: LLM-generated description (if available)

**Usage Example**:
```json
{
  "tool": "image_checker",
  "args": {
    "query": "wireless headphones",
    "reasoning": "Need to verify the product images match the search query"
  }
}
```

**Features**:
- Uses CLIP model (`openai/clip-vit-base-patch32`) for image-text similarity
- Ranks images by relevance to query
- Extracts captions from `<figcaption>` or nearby text
- Generates AI descriptions using multimodal LLM
- Falls back to first 5 images if CLIP fails or <5 images total

---

### ContentAnalyzerTool

**Location**: `tools/analysis_tools.py:475`

**Description**: Comprehensive page analysis including text content, images, and insights. Especially useful for information-intensive pages.

**Name**: `content_analyzer`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Query or context you want to find on the page. |
| `reasoning` | string | Yes | Reasoning for why this content analysis is necessary. |

**Returns**: JSON object containing:
- `page_url`: Current page URL
- `query`: The analysis query
- `page_content`: LLM-summarized page content (query-focused if LLM available)
- `image_analysis`: Image analysis results (same format as ImageCheckerTool)
- `summary`: Comprehensive summary combining text and image insights

**Usage Example**:
```json
{
  "tool": "content_analyzer",
  "args": {
    "query": "What are the top-rated wireless headphones under $100?",
    "reasoning": "Need comprehensive analysis of product page to find best options"
  }
}
```

**Features**:
- Combines `PageParserTool` + `ImageCheckerTool` functionality
- LLM-powered content summarization
- Query-specific insights extraction
- Multimodal analysis (text + images)
- Fallback to basic parsing if LLM unavailable

**Internal Methods**:
- `_parse_page_content()`: AsyncWebCrawler + LLM summarization
- `_analyze_images()`: CLIP-based image ranking + AI descriptions
- `_generate_summary()`: Combines text and image analysis into coherent summary

---

### MapSearchTool

**Location**: `tools/analysis_tools.py:437` (commented out)

**Description**: Navigate to Google Maps to search for geographical information.

**Name**: `map_search`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query for map information (e.g., "Nanning China location"). |
| `reasoning` | string | Yes | Reasoning for why this map search is necessary. |

**Returns**: Google Maps URL with encoded search query.

**Usage Example**:
```json
{
  "tool": "map_search",
  "args": {
    "query": "Nanning China location",
    "reasoning": "Need to find the geographical location of Nanning"
  }
}
```

**Notes**:
- Currently commented out in codebase
- Constructs URL: `https://www.google.com/maps/search/{encoded_query}`

---

### GotoHomepageTool

**Location**: `tools/analysis_tools.py:1010` (commented out)

**Description**: Navigate to the homepage with all available websites.

**Name**: `goto_homepage`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `reasoning` | string | Yes | Reasoning for why this navigation is necessary. |

**Returns**: Homepage URL (`http://localhost:8080/`)

**Notes**:
- Currently commented out in codebase
- Used for WebVoyager/Mind2Web benchmarks with local homepage

---

## Web Search Tools

### WebSearchTool

**Location**: `tools/web_search_tools.py:28`

**Description**: Search the web using multiple search engines (Google, Bing, DuckDuckGo, Yahoo, Baidu) and extract relevant content. Use this when you need to find information that's not available on the current page.

**Registration Name**: `google_web_search`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | The search query to find information about. |
| `reasoning` | string | Yes | Why you need to search for this information. |

**Configuration**:
- `max_queries`: 3 (generates up to 3 search query variations)
- `max_results_per_query`: 3 (fetches top 3 results per query)
- `search_engines`: `["google", "bing"]` (default engines)

**Returns**: Extracted context from relevant web pages with source URLs and screenshots.

**Usage Example**:
```json
{
  "tool": "google_web_search",
  "args": {
    "query": "best wireless headphones 2025 under $100",
    "reasoning": "Need to find current market information for product recommendations"
  }
}
```

**Features**:

1. **LLM-Generated Search Queries**:
   - Generates up to 3 diverse, precise search queries
   - Uses LLM to understand user intent and create comprehensive queries

2. **Multi-Engine Search**:
   - Supports: Google, Bing, DuckDuckGo, Yahoo, Baidu
   - Uses SERPAPI for search results
   - Deduplicates results across engines

3. **Intelligent Content Extraction**:
   - Fetches and parses HTML from search results
   - Removes scripts, styles, navigation elements
   - Extracts clean text content

4. **Screenshot Analysis**:
   - Takes full-page screenshots of search results
   - Splits screenshots into 3 sections (top, middle, bottom)
   - Uses multimodal LLM for visual relevance evaluation

5. **Relevance Filtering**:
   - LLM evaluates if page is useful for user query
   - Considers both text content and visual elements
   - Returns only relevant results

6. **Context Extraction**:
   - LLM extracts query-relevant information from pages
   - Focuses on answering user's specific question
   - Removes noise and irrelevant content

**Internal Methods**:
| Method | Description |
|--------|-------------|
| `_generate_search_queries()` | Generate search query variations using LLM |
| `_perform_search()` | Execute search via SERPAPI across multiple engines |
| `_fetch_page_content()` | Download and parse HTML to extract clean text |
| `_take_screenshot_async()` | Capture full-page screenshot using Playwright |
| `_split_screenshot_into_subfigures()` | Split screenshot into top/middle/bottom sections |
| `_is_page_useful()` | LLM-based relevance evaluation (text + visual) |
| `_extract_relevant_context()` | Extract query-specific information using LLM |

**Output Format**:
```
Source: https://example.com/page1
[Extracted context from page 1]
[Screenshot available: screenshots/example_com_page1.png]

Source: https://example.com/page2
[Extracted context from page 2]
[Screenshot available: screenshots/example_com_page2.png]
```

**Requirements**:
- SERPAPI API key (set in `SERPAPI_API_KEY`)
- Playwright for screenshot capture
- LLM instance for query generation and content analysis
- CLIP model not used (unlike ImageChecker)

---

## Task Completion Tools

### StopTool

**Location**: `tools/gui_tools.py:288`

**Description**: Stop the agent and provide an answer to the task.

**Registration Name**: `stop`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `answer` | string | Yes | Final answer to the task. |
| `reasoning` | string | Yes | Reasoning for why the task is complete. |

**Usage Example**:
```json
{
  "tool": "stop",
  "args": {
    "answer": "Sony WH-1000XM4, $249.99",
    "reasoning": "Found the requested wireless headphones with best reviews in the price range"
  }
}
```

**Notes**:
- Terminates the agent's execution loop
- The `answer` field is returned as the final task result
- Used for task completion in all evaluation benchmarks

---

## Tool Integration Architecture

### Base Tool Pattern

All tools inherit from `BaseTool` (Qwen-Agent framework) and implement:

```python
class ExampleTool(BaseTool):
    name = "tool_name"
    description = "Tool description"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.parameters = {
            'type': 'object',
            'properties': {...},
            'required': [...]
        }

    def call(self, args: str, **kwargs) -> str:
        # Tool implementation
        pass
```

### Tool Registration

Tools are registered using the `@register_tool('name')` decorator:

```python
from qwen_agent.tools.base import register_tool

@register_tool('click')
class ClickTool(BaseTool):
    ...
```

### Action Execution Flow

1. **Agent generates action** → LLM outputs structured JSON with tool call
2. **Tool.call()** → Acknowledges function call, returns confirmation
3. **Agent._process_response()** → Parses tool call and creates browser action
4. **Browser environment** → Executes action via Playwright
5. **Observation** → Returns screenshot + HTML for next step

### Grounding Mechanism

For GUI tools (click, type, selection), the system uses a tiered fallback:

**`auto` mode (default)**:
1. ✅ Use provided pixel coordinates from agent
2. ✅ Use element_id to get element center from accessibility tree
3. ⚠️ Fallback: Call UI-Ins-7B grounding model (2-stage grounding)

**Other modes**:
- `prefer`: coords → grounding model → element ID
- `force`: always call grounding model, fallback to coords/ID on failure
- `off`: coords → element ID only, never call grounding model

### Multimodal Capabilities

Several tools leverage multimodal LLMs:

- **ImageCheckerTool**: CLIP for ranking, LLM for descriptions
- **ContentAnalyzerTool**: LLM for text summarization + image descriptions
- **WebSearchTool**: LLM for query generation, relevance evaluation, context extraction

### Configuration

Tools can be configured via `config` parameter in `__init__`:

```python
config = {
    'llm': llm_instance,  # For LLM-powered tools
    'max_queries': 5,     # Tool-specific settings
}
tool = WebSearchTool(config)
```

---

## Common Patterns

### Error Handling

All tools include try-except blocks:

```python
def call(self, args: str, **kwargs) -> str:
    try:
        # Tool logic
        return result
    except Exception as e:
        return f"Error in {self.name} tool: {str(e)}"
```

### Argument Parsing

Standard pattern for parsing JSON arguments:

```python
if isinstance(args, str):
    args = json.loads(args)

param1 = args.get('param1', default_value)
param2 = args.get('param2')
```

### Page Context Retrieval

Analysis tools retrieve page context from kwargs:

```python
page = kwargs.get('page')
if not page:
    trajectory = kwargs.get('trajectory')
    if trajectory and hasattr(trajectory, 'env'):
        page = trajectory.env.page
```

---

## Best Practices

1. **Always provide reasoning**: All tools require reasoning to explain action necessity
2. **Use element_id when available**: More reliable than coordinates
3. **Provide both coords and element_id**: Ensures fallback options
4. **Use ContentAnalyzer for complex pages**: Better than separate PageParser + ImageChecker calls
5. **Scroll before clicking**: If element not visible, scroll first
6. **Wait after navigation**: Use WaitTool after goto_url or clicks that trigger navigation
7. **Use WebSearch strategically**: Only when information not available on current page
8. **Verify with ImageChecker**: For visual confirmation of elements or products

---

## Dependencies

**Required Libraries**:
- `qwen_agent`: Base tool framework
- `playwright`: Browser automation
- `beautifulsoup4`: HTML parsing
- `requests`: HTTP requests
- `aiohttp`, `asyncio`: Async operations
- `transformers`: CLIP model
- `torch`: Deep learning backend
- `PIL`: Image processing
- `crawl4ai`: AsyncWebCrawler for page parsing

**External Services**:
- SERPAPI: Web search results (requires API key)

**Models**:
- `openai/clip-vit-base-patch32`: Image-text similarity (ImageChecker, ContentAnalyzer)
- UI-Ins-7B: Grounding fallback (configured in agent, not loaded in tools)

---

## Future Extensions

To add a new tool:

1. Create class inheriting from `BaseTool`
2. Define `name`, `description`, `parameters`
3. Implement `call()` method
4. Add `@register_tool('name')` decorator
5. Register in `agent/agent.py` tool list

Example:

```python
@register_tool('new_tool')
class NewTool(BaseTool):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = 'new_tool'
        self.description = 'Description of new tool'
        self.parameters = {
            'type': 'object',
            'properties': {
                'param1': {'type': 'string', 'description': '...'},
                'reasoning': {'type': 'string', 'description': '...'}
            },
            'required': ['param1', 'reasoning']
        }

    def call(self, args: str, **kwargs) -> str:
        if isinstance(args, str):
            args = json.loads(args)
        # Implementation
        return result
```

---

## References

- **Tool Implementations**: `CoMEM-Agent-Inference/tools/`
- **Agent Integration**: `CoMEM-Agent-Inference/agent/agent.py`
- **Action Execution**: `CoMEM-Agent-Inference/browser_env/action_parser_ground.py`
- **Qwen-Agent Docs**: https://github.com/QwenLM/Qwen-Agent
