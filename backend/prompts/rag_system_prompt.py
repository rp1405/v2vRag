rag_system_prompt = """You are an intelligent document analysis assistant. Your role is to:
1. Understand and analyze documents thoroughly
2. Provide accurate and relevant answers based on the document content
3. Maintain context awareness throughout the conversation
4. Be concise yet comprehensive in your responses
5. Acknowledge when information is not available in the document
6. Use proper formatting and structure in your responses

Please ensure your responses are:
- Factual and based on the document content
- Well-structured and easy to read
- Professional and helpful
- Clear about any limitations or uncertainties

Below we have the conversation that had been going on till now as well the relevant context for answering the query, please take all these things into account and process the same."""

llama_system_prompt = """You are a helpful document question-answering assistant.
Your job is to read the document content provided below, and answer the user's query based only on the content. If the answer is not present in the document, say: "The document does not contain that information."
Please answer the question in the same language as the user's query.

## You just need to answer the question, no need to add any other text, answer only the question.
"""


class SystemPrompt:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _llama_prompt(
        self,
        prev_context: str,
        context: str,
        query: str,
        user_prompt: str,
        query_language: str,
    ):
        return f"""### User Prompt: {user_prompt}
### Context:
{context}

### Instructions:
{rag_system_prompt}

### Previous Context:
User: "Hey tell me about yourself"
Assistant: "I am a helpful document question-answering assistant."

User: "Can you tell me how to use the document"
Assistant: "You can use the document by asking questions about it."

{prev_context}

You should answer in {query_language} language.
### User: {query}
### Assistant:
"""

    def _openai_prompt(
        self,
        prev_context: str,
        context: str,
        query: str,
        user_prompt: str,
        query_language: str,
    ):
        return f"""### Instructions:
{rag_system_prompt}

###Question:
{query}

### Previous Context:
{prev_context}

### Relevant Documents:
{context}

### User Prompt:
{user_prompt}

Make sure you generate direct assistant answer, basically what the assistant will reply in this conversation.
You should answer in {query_language} language.
Answer: 
"""

    def convert_context_to_fid_style(self, context: list):
        res = ""
        for index, doc in enumerate(context):
            res += f"Document {index}:\n{doc}\n\n"
        return res

    def convert_to_default_style(self, context: list):
        res = ""
        for index, doc in enumerate(context):
            res += f"{doc}\n\n"
        return res

    def get_prompt(
        self,
        query: str,
        prev_context: str,
        context: list,
        user_prompt: str,
        style: str = "fid",
        query_language: str = "english",
    ):
        if style == "fid":
            context_string = self.convert_context_to_fid_style(context)
        else:
            context_string = self.convert_to_default_style(context)

        if self.model_name == "llama":
            return self._llama_prompt(
                prev_context, context_string, query, user_prompt, query_language
            )
        else:
            return self._openai_prompt(
                prev_context, context_string, query, user_prompt, query_language
            )
