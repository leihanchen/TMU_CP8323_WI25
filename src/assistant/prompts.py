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
{docmuents}
"""


REPORT_WRITER_PROMPT = """Summarize the information based on the user's instructions and previous conversation history. Please provide accurate metrics and quantitative analysis if possible.

CONVERSATION HISTORY:
{chat_history}

STRUCTURE INSTRUCTIONS:
{structure_instruction}

USER INSTRUCTION:
{instruction}

REPORT STRUCTURE:
{report_structure}

PROVIDED INFORMATION:
{information}

# **CRITICAL GUIDELINES:**
{structure_guidelines}
- Start IMMEDIATELY with the summary content - no introductions or meta-commentary
- Focus ONLY on factual, objective information
- Avoid redundancy, repetition, or unnecessary commentary
"""

FINANCIAL_PROMPT = """Summarize the stock financial performance and predict its financial sentiment and stock price

CONVERSATION HISTORY:
{chat_history}

USER INSTRUCTION:
{instruction}

PROVIDED INFORMATION:
{information}

# **CRITICAL GUIDELINES:**
- Focus ONLY on factual, objective information
- Avoid redundancy, repetition, or unnecessary commentary
- List financial relevant metrics and results in the summary
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