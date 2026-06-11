from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
import json
from dotenv import load_dotenv

load_dotenv()
class ProductReview(BaseModel):
    """Structured product review analysis"""
    product_name: str = Field(description="Name of the product")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    rating: int = Field(description="Rating from 1-5", ge=1, le=5)
    pros: List[str] = Field(description="List of postive aspects")
    cons: List[str] = Field(description="List of negative aspects")
    summary: str = Field(description="Brief summary of the review")


llm = ChatOpenAI(model="gpt-4o")

structured_llm = llm.with_structured_output(ProductReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a product review analyzer. Extract structured information from reviews"),
    ("user", "{review_text}")
])

chain = prompt | structured_llm

review_text = """
I bought this wireless mouse last month and it's been mostly great. 
The battery life is incredible - I've only charged it once in 4 weeks. 
The ergonomic design fits my hand perfectly and the buttons are responsive.
However, the scroll wheel is a bit stiff and makes clicking sounds. 
Also, it's quite expensive compared to similar models. 
Overall, I'd give it 4 out of 5 stars.
"""

result = chain.invoke({
    "review_text": review_text
})

print(json.dumps(result.model_dump(), indent=2))