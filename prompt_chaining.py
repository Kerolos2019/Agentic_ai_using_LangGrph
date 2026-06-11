"""
Task: Content Generation pipeline with Quality Control

Input:
- Topic
- Quality Requirements

Steps:
- Generate an initial draft
- Fact check the draft
- Improve the draft based on recommendations from the previous step
- Format for publication
"""
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class ContentState(TypedDict):
    topic: str
    requirements: str
    draft: str
    fact_check_results: str
    improved_content: str
    final_draft: str

llm = ChatOpenAI(model="gpt-4o")


# Define Nodes
def generate_draft(state: ContentState) -> ContentState:
    """Generate intial blog post draft"""

    prompt = f"""
        Write a 200-word blog post about : {state['topic']}

        Requirements: {state['requirements']}

        Focus on creating engaging, informative content
    """

    draft = llm.invoke(prompt).content

    print("=== STEP 1: Draft Generated ===")
    print(draft[:150] +"....\n")

    return {
        "draft": draft
    }

def fact_check(state: ContentState) -> ContentState:
    """Check draft for factual accuracy and consistency"""

    prompt = f"""
    Review the following blog post draft for factual accuracy and consistency:
    
    {state['draft']}

    Identify:
    1. Any factual claims that seem questionable
    2. Internal inconsistencies
    3. Statements that need citations

    Provide a brief report."""

    fact_check_results = llm.invoke(prompt).content

    print("=== STEP 2: Fact Check Complete ===")
    print(fact_check_results[:150] +"....\n")

    return {
        "fact_check_results": fact_check_results
    }

def improve_content(state: ContentState) -> ContentState:
    """Revise content based on fact check feedback"""

    prompt = f"""
    Here is a blog post draft:

    {state['draft']}

    Here is feedback from fact-checking:

    {state['fact_check_results']}

    Revise the blog post to address the feedback while maintaining engaging writing. Keep it around 200 words."""

    improved = llm.invoke(prompt).content

    print("=== STEP 3: Content Improved ===")
    print(improved[:150] +"....\n")

    return {
        "improved_content": improved
    }

def format_output(state: ContentState) -> ContentState:
    """Format content with HTML tags and elements"""

    prompt = f"""
    Format the following blog post for web publication:

    {state['improved_content']}

    Add:
    - An engaging title wrapped in <h1> tags
    - Subheadings where appropriate with <h2> tags
    - Paragraph tags <p>
    - A meta description (1-2 sentences)

    Output the formatted HTML."""

    final = llm.invoke(prompt).content

    print("=== STEP 4: Formatted for Publication ===")
    print(final[:200] +"....\n")

    return {
        "final_draft": final
    }


builder = StateGraph(ContentState)

builder.add_node(generate_draft)
builder.add_node(fact_check)
builder.add_node(improve_content)
builder.add_node(format_output)

# Build the prompt-chaining flow

builder.add_edge(START, "generate_draft")
builder.add_edge("generate_draft", "fact_check")
builder.add_edge("fact_check", "improve_content")
builder.add_edge("improve_content", "format_output")
builder.add_edge("format_output", END)

graph = builder.compile()

result = graph.invoke({
    "topic": "The benefits of morning exercise",
    "requirements": "Target audience: ai engineers"
})

png_data = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)

print("\n" + "="*50)
print("FINAL RESULT")
print("="*50)
print(result["final_draft"])





#html file part:
import webbrowser
import os

html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Content Generation Result</title>
    <style>
        body {{
            font-family: Georgia, serif;
            max-width: 800px;
            margin: 60px auto;
            padding: 0 20px;
            line-height: 1.7;
            color: #333;
            background: #fafafa;
        }}
        .section {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px 28px;
            margin-bottom: 32px;
            background: white;
        }}
        .section h2 {{
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #888;
            margin: 0 0 14px;
        }}
        .section pre {{
            white-space: pre-wrap;
            font-family: Georgia, serif;
            margin: 0;
            font-size: 15px;
        }}
    </style>
</head>
<body>
    <div class="section">
        <h2>Topic</h2>
        <pre>{result["topic"]}</pre>
    </div>
    <div class="section">
        <h2>Requirements</h2>
        <pre>{result["requirements"]}</pre>
    </div>
    <div class="section">
        <h2>Draft</h2>
        <pre>{result["draft"]}</pre>
    </div>
    <div class="section">
        <h2>Fact Check Results</h2>
        <pre>{result["fact_check_results"]}</pre>
    </div>
    <div class="section">
        <h2>Improved Content</h2>
        <pre>{result["improved_content"]}</pre>
    </div>
    <div class="section">
        <h2>Final Draft</h2>
        {result["final_draft"]}
    </div>
</body>
</html>"""

output_path = os.path.abspath("output.html")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

webbrowser.open(f"file://{output_path}")
print(f"\nOpened in browser: {output_path}")