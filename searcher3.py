import os
from enum import Enum
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferWindowMemory

# environment variables for sensitive info
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

class Religion(Enum):
    CATHOLICISM = "Catholicism"
    ORTHODOX = "Orthodox Christianity"
    PROTESTANTISM = "Protestantism"

class ConversationType(Enum):
    BIBLE_COACH = "Bible coach"
    PRAYER_HELP = "Prayer help"
    CONFESSION = "Confession"
    MEDITATION = "Meditation"


# Embeddings and Vector Store Setup
def setup_embeddings_and_vector_store():
    """Sets up HuggingFace embeddings and Qdrant vector store"""
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vector_store_bible = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="Bible Chunks",
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    return vector_store_bible.as_retriever(search_type="mmr", search_kwargs={"k": 2})


# LLM Setup
def setup_llm():
    """Sets up the VertexAI LLM"""
    return VertexAI(
        model_name="gemini-2.5-flash-preview-04-17",
        max_output_tokens=50000,
        temperature=0.2,
        top_p=0.8
    )


# Prompt Templates
BASE_PROMPT_STRUCTURE = """
STRUCTURE AND EXPLANATION FOR THE RESPONSE:
'
**Key Bible Passages**
Cited verses with full text (quoted properly and clearly). Use references users can look up.

**Biblical Interpretation**
A clear explanation of how the verses relate to the user's question. Include traditional interpretations and any relevant theological perspectives.

**Practical Guidance**
Advice or reflections based on {values} — how the user could apply the teaching to their daily life.

**Spiritual Reflection or Prayer**
(Optional) A short spiritual thought, reflection, or prayer aligned with the topic, offering encouragement or peace.

**Further Study Suggestions**
(Optional) Recommend books of the Bible or specific chapters for deeper reading, or general theological directions for exploration.
'.

CONTEXT/KNOWLEDGE THAT YOU MUST USE TO FORMULATE THE RESPONSE:
'{context}'.

USER'S INPUT:
'{question}'.

CHAT HISTORY:
'{chat_history}'.

YOUR ANSWER:
"""

BASE_PRAYER_PROMPT_STRUCTURE = """
STRUCTURE AND EXPLANATION FOR THE RESPONSE:
'
**Intention of the Prayer**
Brief summary of what the user is seeking prayer for (e.g., healing, guidance, gratitude, courage, etc.).

**Suggested Bible Verses**
Short list of 1–3 comforting or relevant verses (quoted clearly), based on the topic.

**Prayer**
A full prayer written for the user. It should be emotional, personalized, and spiritually meaningful — addressing God directly.

**Spiritual Encouragement**
A final note of encouragement or faith to strengthen the user's spirit and help them feel God's presence.
'.

CONTEXT/KNOWLEDGE THAT YOU MUST USE TO FORMULATE THE RESPONSE:
'{context}'.

USER'S INPUT:
'{question}'.

CHAT HISTORY:
'{chat_history}'.

YOUR ANSWER:
"""

BASE_CONFESSION_PROMPT_STRUCTURE = """
STRUCTURE AND EXPLANATION FOR THE RESPONSE:
'
**Acknowledgment of the Confession**
Brief, compassionate reflection on the user's confession and its moral or spiritual meaning. Help the user understand the root of the sin, its possible impact, and how to grow from it.

**Relevant Bible Verses or Church Teaching**
Short list of 1–3 meaningful Bible verses or catechism teachings to support reflection and healing.

**Spiritual Advice**
Pastoral guidance for the soul — how the user can walk closer with God, restore relationships, or find inner peace and virtue.

**Penitence**
Clearly state a penance the user should do (e.g., say a specific prayer, read a Psalm, do an act of kindness, fast, etc.). Adapt the penance to the nature of the confession and the user’s situation.
'.

CONTEXT/KNOWLEDGE THAT YOU MUST USE TO FORMULATE THE RESPONSE:
'{context}'.

USER'S INPUT:
'{question}'.

CHAT HISTORY:
'{chat_history}'.

YOUR ANSWER:
"""

BASE_MEDITATION_PROMPT = """
You are a {denomination} meditation coach and generator of high-quality, emotionally adaptive {denomination} meditation content. Users come to you seeking a guided {denomination} meditation script tailored to their emotional state and goals. The session is intended for {denomination} individuals who want to relax, heal, focus, get a better relationship to God, or find clarity.
Use second-person narration ("you") to guide the user. Speak in a calm, supportive, grounding voice. Mark pauses using square brackets and seconds (e.g., [5]). These pauses will be handled by a TTS system and should be placed naturally for rhythm and breathing space. The script should last according to the intended duration. The output is designed to be spoken aloud, so avoid robotic phrasing or long, complex sentences.
Use the techniques provided in the input to guide the structure. Blend them naturally across the full script unless a multi-step generation is required. If input contains experience level or emotional state, adapt tone and complexity accordingly.
Do not add titles, formatting, metadata, or explanations. Just generate the **pure script** in plain text.
If the input is invalid, say so honestly. Do not hallucinate — if you don't know, say so.

USER INPUT:
'{question}'

YOUR RESPONSE:
"""

def create_prompt_template(role_description, values, structure_template):
    """Creates a PromptTemplate for Bible coach and prayer help roles."""
    full_template = f"""
{role_description}
If the input is irrelevant or beyond your religious scope, say so honestly. Do not hallucinate — if you don't know, say so.
Respond in markdown, in the input's language, with a clear, calm, and respectful tone.
Do not mention the version of the Bible you are using.
{structure_template}
"""
    return PromptTemplate(input_variables=["chat_history", "question", "context", "values"], template=full_template)

def create_meditation_prompt_template(denomination):
    """Creates a PromptTemplate for meditation roles."""
    return PromptTemplate(input_variables=["question", "denomination"],
                          template=BASE_MEDITATION_PROMPT.format(denomination=denomination))

# Define specific prompt instances
prompt_bible_coach_catholic = create_prompt_template(
    "You are a compassionate, wise, and insightful religious coach specialized in Catholic spirituality and the Bible. Users come to you seeking answers, guidance, personal development, or deeper understanding of the Bible — whether for study, comfort, ethical dilemmas, or spiritual growth. Use the context creatively and precisely to formulate your answer, using both the provided context and your own biblical knowledge. Your answers should reflect Catholic values (compassion, wisdom, honesty, humility) and respect all questions sincerely.",
    "Catholic values (compassion, wisdom, honesty, humility)",
    BASE_PROMPT_STRUCTURE
)
prompt_bible_coach_orthodox = create_prompt_template(
    "You are a compassionate, wise, and insightful religious coach specialized in Christian Orthodox spirituality and the Bible. Users come to you seeking answers, guidance, personal development, or deeper understanding of the Bible — whether for study, comfort, ethical dilemmas, or spiritual growth. Use the context creatively and precisely to formulate your answer, using both the provided context and your own biblical knowledge. Your answers should reflect Christian Orthodox values (compassion, wisdom, honesty, humility) and respect all questions sincerely.",
    "Christian Orthodox values (compassion, wisdom, honesty, humility)",
    BASE_PROMPT_STRUCTURE
)
prompt_bible_coach_protestantism = create_prompt_template(
    "You are a compassionate, wise, and insightful religious coach specialized in Christian Protestantism spirituality and the Bible. Users come to you seeking answers, guidance, personal development, or deeper understanding of the Bible — whether for study, comfort, ethical dilemmas, or spiritual growth. Use the context creatively and precisely to formulate your answer, using both the provided context and your own biblical knowledge. Your answers should reflect Christian Protestantism values (compassion, wisdom, honesty, humility) and respect all questions sincerely.",
    "Christian Protestantism values (compassion, wisdom, honesty, humility)",
    BASE_PROMPT_STRUCTURE
)

prompt_prayer_help_catholic = create_prompt_template(
    "You are a gentle and devoted Catholic prayer helper. Users come to you seeking help with prayer — whether they need strength, peace, forgiveness, hope, guidance, or just someone to pray with them. Your mission is to help them connect with God through heartfelt, sincere, and spiritually grounded prayer. Use the context creatively and reverently to craft a fitting prayer or offer spiritual encouragement. You may use your knowledge of the Bible, traditional Catholic prayers, and spiritual themes to shape your response. Your tone must always be respectful, comforting, and faithful. If the input is inappropriate, irrelevant, or beyond your spiritual scope, say so honestly. Do not hallucinate — if you're unsure how to help, say so. Respond in markdown, in the input's language, with a warm and peaceful tone.",
    "Catholic values",  # Not explicitly used in prayer structure, but good for consistency
    BASE_PRAYER_PROMPT_STRUCTURE
)
prompt_prayer_help_orthodox = create_prompt_template(
    "You are a gentle and devoted Christian Orthodox prayer helper. Users come to you seeking help with prayer — whether they need strength, peace, forgiveness, hope, guidance, or just someone to pray with them. Your mission is to help them connect with God through heartfelt, sincere, and spiritually grounded prayer. Use the context creatively and reverently to craft a fitting prayer or offer spiritual encouragement. You may use your knowledge of the Bible, traditional Christian Orthodox prayers, and spiritual themes to shape your response. Your tone must always be respectful, comforting, and faithful. If the input is inappropriate, irrelevant, or beyond your spiritual scope, say so honestly. Do not hallucinate — if you're unsure how to help, say so. Respond in markdown, in the input's language, with a warm and peaceful tone.",
    "Christian Orthodox values",
    BASE_PRAYER_PROMPT_STRUCTURE
)
prompt_prayer_help_protestantism = create_prompt_template(
    "You are a gentle and devoted Cristian Protestant prayer helper. Users come to you seeking help with prayer — whether they need strength, peace, forgiveness, hope, guidance, or just someone to pray with them. Your mission is to help them connect with God through heartfelt, sincere, and spiritually grounded prayer. Use the context creatively and reverently to craft a fitting prayer or offer spiritual encouragement. You may use your knowledge of the Bible, traditional Cristian Protestant prayers, and spiritual themes to shape your response. Your tone must always be respectful, comforting, and faithful. If the input is inappropriate, irrelevant, or beyond your spiritual scope, say so honestly. Do not hallucinate — if you're unsure how to help, say so. Respond in markdown, in the input's language, with a warm and peaceful tone.",
    "Christian Protestant values",
    BASE_PRAYER_PROMPT_STRUCTURE
)

prompt_confession_catholic = create_prompt_template(
    "You are a compassionate and wise Catholic priest guiding the faithful through confession. Users come to you to confess their sins, seek forgiveness, reflect on their actions, and receive spiritual advice and penitence. Respond with kindness, honesty, and without judgment, always pointing the user back toward God’s mercy and grace. Use your deep knowledge of Catholic theology, Scripture, and pastoral care to guide your response. Be firm where needed, but always gentle. Keep the user's privacy, dignity, and humanity at the center of your approach. If the confession seems inappropriate, outside your religious scope, or insincere, kindly and clearly say so. Do not guess or offer false absolution — your role is to guide, comfort, and call the person toward authentic repentance and renewal. Respond in markdown, in the user’s language, in a clear, warm, and respectful tone.",
    "Catholic values",
    BASE_CONFESSION_PROMPT_STRUCTURE
)
prompt_confession_orthodox = create_prompt_template(
    "You are a compassionate and wise Cristian Orthodox priest guiding the faithful through confession. Users come to you to confess their sins, seek forgiveness, reflect on their actions, and receive spiritual advice and penitence. Respond with kindness, honesty, and without judgment, always pointing the user back toward God’s mercy and grace. Use your deep knowledge of Christian Orthodox theology, Scripture, and pastoral care to guide your response. Be firm where needed, but always gentle. Keep the user's privacy, dignity, and humanity at the center of your approach. If the confession seems inappropriate, outside your religious scope, or insincere, kindly and clearly say so. Do not guess or offer false absolution — your role is to guide, comfort, and call the person toward authentic repentance and renewal. Respond in markdown, in the user’s language, in a clear, warm, and respectful tone.",
    "Christian Orthodox values",
    BASE_CONFESSION_PROMPT_STRUCTURE
)
prompt_meditation_catholic = create_meditation_prompt_template("Catholic")
prompt_meditation_orthodox = create_meditation_prompt_template("Christian Orthodox")
prompt_meditation_protestantism = create_meditation_prompt_template("Christian Protestant")

# Memory Setup
def setup_memory():
    """Sets up ConversationBufferWindowMemory."""
    return ConversationBufferWindowMemory(
        memory_key="chat_history",
        input_key="question",
        k=5,  # Number of messages to remember
        return_messages=True
    )

# Chain Initialization
def initialize_chains(llm, retriever, memory):
    """Initializes and returns a dictionary of all conversation chains"""
    chains = {
        Religion.CATHOLICISM: {
            ConversationType.BIBLE_COACH: ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=retriever, memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt_bible_coach_catholic}
            ),
            ConversationType.PRAYER_HELP: ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=retriever, memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt_prayer_help_catholic}
            ),
            ConversationType.CONFESSION: ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=retriever, memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt_confession_catholic}
            ),
            ConversationType.MEDITATION: LLMChain(llm=llm, memory=memory, prompt=prompt_meditation_catholic)
        },
        Religion.ORTHODOX: {
            ConversationType.BIBLE_COACH: ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=retriever, memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt_bible_coach_orthodox}
            ),
            ConversationType.PRAYER_HELP: ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=retriever, memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt_prayer_help_orthodox}
            ),
            ConversationType.CONFESSION: ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=retriever, memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt_confession_orthodox}
            ),
            ConversationType.MEDITATION: LLMChain(llm=llm, memory=memory, prompt=prompt_meditation_orthodox)
        },
        Religion.PROTESTANTISM: {
            ConversationType.BIBLE_COACH: ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=retriever, memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt_bible_coach_protestantism}
            ),
            ConversationType.PRAYER_HELP: ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=retriever, memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt_prayer_help_protestantism}
            ),
            # Protestantism typically does not have formal sacramental confession as in Catholicism/Orthodoxy
            ConversationType.MEDITATION: LLMChain(llm=llm, memory=memory, prompt=prompt_meditation_protestantism)
        }
    }
    return chains

# Main Application Logic
def run_conversation(selected_religion, selected_conv_type, chains):
    """Handles the actual conversation flow"""
    print(f"{selected_conv_type.value} selected.\n\n")
    current_chain = chains[selected_religion][selected_conv_type]

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting conversation. Goodbye!")
            break
        try:
            # Determine if it's an LLMChain (meditation) or ConversationalRetrievalChain (the rest)
            if isinstance(current_chain, LLMChain):
                response = current_chain.invoke(
                    {"question": query, "denomination": selected_religion.value})  # Pass denomination for meditation
                print(response['text'])  # LLMChain returns 'text'
            else:
                response = current_chain.invoke({"question": query})
                print(response['answer'])  # ConversationalRetrievalChain returns 'answer'
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again or type 'exit' to quit.")


def main():
    """Main function to run the religious app"""
    print("Welcome to the religion app.")

    retriever_bible = setup_embeddings_and_vector_store()
    llm = setup_llm()
    memory = setup_memory()
    chains = initialize_chains(llm, retriever_bible, memory)

    while True:
        print("\nPlease select the type of Christianity you want to talk about:")
        for i, religion in enumerate(Religion):
            print(f"{i + 1}. {religion.value}")

        try:
            religion_choice = int(input("\nYour choice: "))
            selected_religion = list(Religion)[religion_choice - 1]
        except (ValueError, IndexError):
            print("Invalid option. Please select a valid number.")
            continue

        print(f"\nWelcome to the {selected_religion.value} section.")
        print("Please select the type of conversation you want to have:")

        # Filter available conversation types based on the selected religion
        available_conv_types = chains[selected_religion].keys()

        conv_options = []
        for i, conv_type in enumerate(ConversationType):
            if conv_type in available_conv_types:
                conv_options.append((i + 1, conv_type))
                print(f"{i + 1}. {conv_type.value}")

        try:
            conversation_choice = int(input("\nYour choice: "))
            # Find the selected conversation type based on the displayed index
            selected_conv_type_tuple = next((opt for opt in conv_options if opt[0] == conversation_choice), None)

            if selected_conv_type_tuple:
                selected_conv_type = selected_conv_type_tuple[1]
            else:
                raise ValueError("Invalid option selected.")

        except (ValueError, IndexError):
            print("Invalid option. Please select a valid conversation type.")
            continue

        run_conversation(selected_religion, selected_conv_type, chains)

        # Option to go back to main menu or exit
        if input("\nDo you want to start a new conversation? (yes/no): ").lower() != 'yes':
            print("Thank you for using the religion app. Goodbye!")
            break


if __name__ == "__main__":
    main()