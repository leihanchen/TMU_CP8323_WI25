RESEARCH_QUERY_WRITER_PROMPT = """You are an expert Query Writer who specializes in designing precise and effective queries to fulfill user tasks.

Your goal is to generate the necessary queries to complete the user's goal based on their instructions. Ensure the queries are concise, relevant, and avoid redundancy.

Your output must only be a JSON object containing a single key "queries":
{{ "queries": ["Query 1", "Query 2",...] }}

# NOTE:
* You can generate up to {max_queries} queries, but only as many as needed to effectively address the user's research goal.
* Focus on the user's intent and break down complex tasks into manageable queries.
* Avoid generating excessive or redundant queries.
* Ensure the queries are specific enough to retrieve relevant information but broad enough to cover the scope of the task.
* If the instruction is ambiguous, generate queries that address possible interpretations.
* **Today is: {date}**
"""

RELEVANCE_EVALUATOR_PROMPT = """Your goal is to evaluate and determine if the provided documents are relevant to answer the user's query.

# Key Considerations:

* Focus on semantic relevance, not just keyword matching
* Consider both explicit and implicit query intent
* A document can be relevant even if it only partially answers the query.
* **Your output must only be a valid JSON object with a single key "is_relevant":**
{{'is_relevant': True/False}}

# USER QUERY:
{query}

# RETRIEVED DOCUMENTS:
{documents}

# **IMPORTANT:**
* **Your output must only be a valid JSON object with a single key "is_relevant":**
{{'is_relevant': True/False}}
"""


SUMMARIZER_PROMPT="""Your goal is to generate a focused, evidence-based summary from the provided documents.

KEY OBJECTIVES:
1. Extract and synthesize critical findings from each source
2. Present key data points and metrics that support main conclusions
3. Identify emerging patterns and significant insights
4. Structure information in a clear, logical flow

REQUIREMENTS:
- Begin immediately with key findings - no introductions
- Focus on verifiable data and empirical evidence
- Present more relevant metrics and results
- Keep the summary brief, avoid repetition and unnecessary details
- Prioritize information directly relevant to the query

Query:
{query}

Retrieved Documents:
{documents}
"""


# REPORT_WRITER_PROMPT = """Summarize the information based on the user's instructions and previous conversation history. Please provide accurate metrics and quantitative analysis if possible.

# CONVERSATION HISTORY:
# {chat_history}

# STRUCTURE INSTRUCTIONS:
# {structure_instruction}

# USER INSTRUCTION:
# {instruction}

# REPORT STRUCTURE:
# {report_structure}

# PROVIDED INFORMATION:
# {information}

# # **CRITICAL GUIDELINES:**
# {structure_guidelines}
# - Start IMMEDIATELY with the summary content - no introductions or meta-commentary
# - Focus ONLY on factual, objective information
# - Avoid redundancy, repetition, or unnecessary commentary
# """

REPORT_WRITER_PROMPT = """
You are a financial analysis assistant tasked with answering **predictive financial questions** using retrieved documents and your own reasoning.

# GOAL:
Answer the user's question about future values (e.g., stock price, financial performance) using:

1. Evidence from the provided documents (filings, news, market data)
2. Your own logical estimation and financial reasoning
3. Clearly explain how you derived the answer

---

# USER QUESTION:
{instruction}

# RETRIEVED INFORMATION:
{information}

# CONVERSATION HISTORY:
{chat_history}

---

# RESPONSE REQUIREMENTS:
- Start directly with the **prediction**
- Include a **justification** using any relevant documents (quote snippets, name sources)
- Include **math or reasoning** used to estimate, e.g., trends, growth rate extrapolation
- End with a **confidence score (High / Medium / Low)** and why

---

# OUTPUT FORMAT (strict):
Prediction: <your forecast>  
Justification: <why you made this forecast — include any math, logic, or quotes from documents>  
Confidence: <High / Medium / Low> - <reason for confidence level>
"""

def get_structure_prompt(structure_name):
    if structure_name.lower() == "none":
        return {
            "instruction": "Organize the information in a natural and flowing way that best serves the user's needs.",
            "guidelines": "- Present information in a clear and logical manner\n- Use appropriate formatting for readability"
        }
    else:
        return {
            "instruction": "The report must strictly follow the structure requested by the user.",
            "guidelines": "- Adhere strictly to the structure specified in the user's instruction"
        }