#
# Synthetic Data Generation Pipeline for Social Sciences & Humanities (V6)
#
# This script uses the Qwen/Qwen3-32B model to generate nuanced, multi-turn
# conversations. This version uses a detailed dictionary of academic subfields
# instead of broad topics for more granular and specific data generation.
#
# Hardware Requirement: Runs effectively on an NVIDIA A100 80GB GPU.
#
# How it works:
# 1.  Configuration: Defines personas with specific task lists, a detailed dictionary
#     of academic subfields, and generation parameters.
# 2.  Strict System Prompting: A detailed system prompt instructs the model on its
#     role, the specific subfield, the overarching topic, the task, and the desired
#     conversational style.
# 3.  Dynamic Turn-by-Turn Simulation: The script iterates through all combinations
#     of subfield, persona, and task, prompting the model to generate one response at
#     a time.
# 4.  Dynamic Thinking Mode: Enables thinking for the SHARE AI persona for
#     reasoned responses and disables it for the human persona.
# 5.  Data Structuring & Saving: Saves the complete, dynamically generated
#     conversations to a JSONL file.
#

import os # Import os first to set environment variables
import json
import torch
from tqdm import tqdm # For a nice progress bar
import itertools # To help with pairing topics and personas

# --- CRITICAL: Set custom cache directory BEFORE importing transformers ---
# The transformers library reads these environment variables upon its initial import.
# Setting them first ensures that all components, including the 'hub' downloader,
# use the correct directory from the start.

CACHE_DIR = "/run/surfdrive_data/"

# Set all relevant Hugging Face environment variables
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

# Now, create the directory to ensure it exists before the library tries to use it.
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"Hugging Face cache directory is configured to: {CACHE_DIR}")

# --- Now that the environment is set, import transformers ---
# This import must come AFTER the os.environ calls above.
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


# --- 1. Configuration ---

MODEL_ID = "Qwen/Qwen3-32B"
OUTPUT_FILE = "synthetic_humanities_chats_dynamic_v6.jsonl" # New output file

NUM_TURNS_PER_CONVERSATION = 4 # Results in an 8-message conversation
MAX_NEW_TOKENS = 2048

# --- 2. Define Personas, Tasks, and SSH Subfields ---

# The central AI persona
SHARE_PERSONA = {
    "key": "share_ai",
    "name": "SHARE",
    "description": "the social humanities language model for research and education, a helpful AI trained on a large dataset of social sciences and humanities academic texts."
}

# The human personas who will interact with SHARE, now with specific tasks
HUMAN_PERSONAS = {
    "student": {
        "key": "student",
        "name": "Alex, the Student",
        "description": "an inquisitive undergraduate student.",
        "tasks": [
            "Brainstorming for research topic ideas",
            "Getting feedback regarding an assignment text",
            "Understanding a topic or academic paragraph of text"
        ]
    },
    "teacher": {
        "key": "teacher",
        "name": "Dr. Davies, the Lecturer",
        "description": "an experienced university lecturer looking for resources for their courses.",
        "tasks": [
            "Getting examples to use in class",
            "Brainstorming exam questions",
            "Understanding a topic in depth"
        ]
    },
    "researcher": {
        "key": "researcher",
        "name": "Dr. Chen, the Researcher",
        "description": "a university researcher exploring interdisciplinary connections and looking for source materials.",
        "tasks": [
            "Brainstorming for a grant proposal",
            "Getting constructive criticism regarding a research idea",
            "Finding methods or resources to implement a research project"
        ]
    },
    "administrator": {
        "key": "administrator",
        "name": "Dean Evans, the Administrator",
        "description": "a university administrator focused on curriculum development and strategy.",
        "tasks": [
            "Understanding key challenges to universities",
            "Creating communication materials for a new programme",
            "Getting feedback on a short curriculum structure"
        ]
    }
}

# The list of topics to be discussed, now with subfields
SSH_SUBFIELDS = {
    "Archaeology": [
        "Bioarchaeology: The study of human remains from archaeological sites to understand diet, health, and lifestyle in the past.",
        "Zooarchaeology: The analysis of animal remains from archaeological contexts to reconstruct past environments and human-animal interactions.",
        "Paleoethnobotany: The study of plant remains from archaeological sites to understand past diets, agriculture, and land use.",
        "Historical Archaeology: The study of past societies that left behind written records, combining archaeological evidence with textual sources.",
        "Maritime Archaeology: The investigation of human interaction with the sea, lakes, and rivers through the study of submerged sites, shipwrecks, and coastal landscapes."
    ],
    "Area Studies": [
        "African Studies: An interdisciplinary field focusing on the diverse cultures, histories, politics, and economies of the African continent.",
        "Asian Studies: The study of the histories, cultures, languages, and political systems of Asian countries and their diasporas.",
        "European Studies: An interdisciplinary approach to understanding the history, cultures, politics, and societies of Europe.",
        "Latin American Studies: The integrated study of the history, cultures, languages, and political economies of Latin America and the Caribbean.",
        "Middle Eastern Studies: An interdisciplinary field that examines the languages, cultures, religions, history, and politics of the Middle East and North Africa."
    ],
    "Art and Architecture": [
        "Art History: The study of art objects in their historical and stylistic contexts, tracing the development of visual arts across time and cultures.",
        "Architectural History: The study of the history of buildings and the built environment, including their design, construction, and cultural significance.",
        "Museum Studies (Museology): The study of museums and their role in society, including collecting, preserving, interpreting, and exhibiting cultural heritage.",
        "Historic Preservation: The professional practice of protecting and conserving buildings, objects, landscapes, and other artifacts of historical significance.",
        "Aesthetics: A branch of philosophy dealing with the nature of beauty, art, and taste, and with the creation and appreciation of beauty."
    ],
    "Business Administration": [
        "Finance: The study and management of money, investments, and other financial instruments.",
        "Marketing: The study of creating, communicating, delivering, and exchanging offerings that have value for customers, clients, partners, and society at large.",
        "Human Resource Management: The strategic approach to the effective management of people in an organization to help their business gain a competitive advantage.",
        "Operations Management: The administration of business practices to create the highest level of efficiency possible within an organization.",
        "International Business: The study of business activities that cross national borders."
    ],
    "Communication Sciences": [
        "Interpersonal Communication: The study of how individuals exchange messages and create meaning in face-to-face interactions.",
        "Organizational Communication: The study of communication processes within and among organizations.",
        "Mass Communication: The study of how individuals and entities relay information through mass media to large segments of the population at the same time.",
        "Health Communication: The study and use of communication strategies to inform and influence individual and community decisions that enhance health.",
        "Political Communication: The study of the relationship between communication and politics, including the role of media in political processes."
    ],
    "Computers and the Humanities (Digital Humanities)": [
        "Digital Archiving and Preservation: The creation and maintenance of digital collections of humanities materials.",
        "Computational Linguistics: The use of computer science to study and process human language.",
        "Digital History: The use of digital tools and methods to research, analyze, and present historical information.",
        "Electronic Literature: The study of literary works that are created and read on digital devices.",
        "Geospatial Humanities: The application of geographic information systems (GIS) and other spatial technologies to humanities research."
    ],
    "Cultural Anthropology": [
        "Ethnography: The systematic study of people and cultures through participant observation and other qualitative methods.",
        "Medical Anthropology: The study of how health and illness are shaped, experienced, and understood in light of social, cultural, and biological factors.",
        "Visual Anthropology: The study and production of ethnographic photography, film, and new media as tools for anthropological research and representation.",
        "Economic Anthropology: The study of economic life from an anthropological perspective, examining how people produce, exchange, and consume goods and services.",
        "Psychological Anthropology: The study of the relationship between culture and the individual, focusing on how cultural beliefs and practices shape psychological processes."
    ],
    "Demography": [
        "Fertility and Fecundity: The study of birth rates, family planning, and the biological and social factors influencing reproduction.",
        "Mortality and Morbidity: The study of death rates, life expectancy, and the causes and distribution of diseases in populations.",
        "Migration and Immigration: The study of the movement of people across geographical borders, both internal and international.",
        "Population Aging: The study of the social, economic, and health consequences of aging populations.",
        "Social Demography: The study of the social determinants and consequences of population processes and structures."
    ],
    "Development Studies": [
        "Development Economics: A branch of economics that focuses on improving the fiscal, economic, and social conditions in developing countries.",
        "Gender and Development: The study of the gendered dimensions of development processes and the impact of development policies on gender relations.",
        "Sustainable Development: The study of development that meets the needs of the present without compromising the ability of future generations to meet their own needs.",
        "Post-Colonial Studies: An academic field that analyzes, explains, and responds to the cultural legacy of colonialism and imperialism.",
        "Humanitarian Action: The study and practice of providing aid to people in need, typically in response to natural disasters or armed conflict."
    ],
    "Economics": [
        "Microeconomics: The study of the economic behavior of individuals, households, and firms in decision-making and allocation of resources.",
        "Macroeconomics: The study of the economy as a whole, including phenomena such as inflation, unemployment, and economic growth.",
        "Econometrics: The application of statistical methods to economic data to give empirical content to economic relationships.",
        "Behavioral Economics: A method of economic analysis that applies psychological insights into human behavior to explain economic decision-making.",
        "International Economics: The study of economic interactions between countries, including trade, finance, and investment."
    ],
    "Environmental Science": [
        "Ecology: The scientific study of the interactions between organisms and their environment.",
        "Environmental Chemistry: The study of the chemical processes that occur in the environment, including the sources, reactions, transport, effects, and fates of chemical species.",
        "Atmospheric Sciences: The study of the Earth's atmosphere and its various inner-working physical processes.",
        "Geosciences: The study of the Earth, its oceans, and atmosphere, and their interactions.",
        "Environmental Policy and Management: The study of how to develop and implement policies and manage resources to protect the environment."
    ],
    "Gender Studies": [
        "Feminist Theory: The study of the social, economic, and political equality of the sexes, and the development of theories to understand gender inequality.",
        "Masculinity Studies: An interdisciplinary field that focuses on the social and cultural construction of masculinities.",
        "Queer Theory: A field of critical theory that emerged in the early 1990s out of the fields of queer studies and women's studies.",
        "Transgender Studies: An interdisciplinary field that examines the experiences, histories, and cultural representations of transgender people.",
        "Gender and Sexuality Studies: The interdisciplinary study of gender and sexuality as social and cultural constructs."
    ],
    "Geography / Planning": [
        "Human Geography: The study of the spatial organization of human activity and of people's relationships with their environments.",
        "Physical Geography: The study of the patterns and processes of the natural environment, including the atmosphere, hydrosphere, biosphere, and geosphere.",
        "Geographic Information Science (GIScience): The scientific discipline that studies data structures and computational techniques to capture, represent, process, and analyze geographic information.",
        "Urban and Regional Planning: A technical and political process concerned with the development and design of land use and the built environment.",
        "Environmental Geography: The study of the spatial aspects of interactions between humans and the natural world."
    ],
    "History": [
        "Social History: The study of the history of ordinary people and their everyday lives.",
        "Political History: The study of political events, ideas, movements, leaders, and institutions.",
        "Cultural History: The study of the cultural practices, beliefs, and values of past societies.",
        "Economic History: The study of how economies and economic phenomena have evolved over time.",
        "Military History: The study of armed conflict in the history of humanity, and its impact on societies, their cultures, economies and politics."
    ],
    "History of Science": [
        "History of Medicine: The study of the historical development of medical knowledge, practices, and institutions.",
        "History of Technology: The study of the development of technology and its impact on society and culture.",
        "History of Natural Philosophy: The study of the historical development of what is now known as science, particularly from ancient Greece to the early modern period.",
        "Science and Society: The study of the social, cultural, and political contexts in which science and technology develop.",
        "Philosophy of Science: The branch of philosophy that examines the foundations, methods, and implications of science."
    ],
    "Language and Literature": [
        "Comparative Literature: The study of literature and cultural expression across linguistic, national, and disciplinary boundaries.",
        "Literary Theory and Criticism: The study of the principles and methods of interpreting and evaluating literary texts.",
        "Philology: The study of language in oral and written historical sources; it is a combination of literary criticism, history, and linguistics.",
        "Discourse Analysis: The study of language in use, examining how it functions in social contexts.",
        "Postcolonial Literature: The study of literature produced by peoples from formerly colonized countries."
    ],
    "Law": [
        "Constitutional Law: The body of law which defines the role, powers, and structure of different entities within a state, namely, the executive, the parliament or legislature, and the judiciary.",
        "Criminal Law: The body of law that relates to crime. It proscribes conduct perceived as threatening, harmful, or otherwise endangering to the property, health, safety, and moral welfare of people.",
        "International Law: The set of rules, norms, and standards generally accepted in relations between nations.",
        "Corporate Law: The body of law governing the rights, relations, and conduct of persons, companies, organizations and businesses.",
        "Family Law: An area of the law that deals with family matters and domestic relations."
    ],
    "Linguistics": [
        "Phonetics and Phonology: The study of the sounds of human speech and the systematic organization of sounds in languages.",
        "Syntax: The study of the rules that govern the structure of sentences in a given language.",
        "Semantics and Pragmatics: The study of meaning in language, both in terms of literal meaning (semantics) and how context affects meaning (pragmatics).",
        "Sociolinguistics: The study of the relationship between language and society.",
        "Psycholinguistics: The study of the psychological and neurobiological factors that enable humans to acquire, use, and understand language."
    ],
    "Music, Theatre, Performing Arts and Media": [
        "Musicology: The scholarly study of music, encompassing historical musicology, ethnomusicology, and music theory.",
        "Theatre Studies: The study of theatrical performance in its cultural, historical, and theoretical contexts.",
        "Performance Studies: An interdisciplinary field that studies performance and uses performance as a lens to study the world.",
        "Dance Studies: The critical and historical study of dance as a cultural practice and art form.",
        "Media Studies: The field of study that deals with the content, history, and effects of various media; in particular, the mass media."
    ],
    "Pedagogics": [
        "General Pedagogy: The study of the fundamental principles and theories of education and teaching.",
        "Social Pedagogy: An approach to care and education that emphasizes social learning and the well-being of the whole person.",
        "Special Education: The education of students with special needs in a way that addresses their individual differences and needs.",
        "Andragogy (Adult Education): The theory and practice of teaching adult learners.",
        "Higher Education Pedagogy: The study of teaching and learning in universities and other institutions of higher education."
    ],
    "Philosophy": [
        "Epistemology: The branch of philosophy concerned with the theory of knowledge.",
        "Metaphysics: The branch of philosophy that examines the fundamental nature of reality, including the relationship between mind and matter, between substance and attribute, and between potentiality and actuality.",
        "Ethics (Moral Philosophy): The branch of philosophy that involves systematizing, defending, and recommending concepts of right and wrong conduct.",
        "Logic: The study of reasoning and argumentation.",
        "Political Philosophy: The philosophical study of government, addressing questions about the nature, scope, and legitimacy of public agents and institutions and the relationships between them."
    ],
    "Psychology": [
        "Clinical Psychology: The branch of psychology concerned with the assessment and treatment of mental illness, abnormal behavior, and psychiatric problems.",
        "Cognitive Psychology: The scientific study of mental processes such as attention, language use, memory, perception, problem solving, creativity, and thinking.",
        "Developmental Psychology: The scientific study of how and why human beings change over the course of their life.",
        "Social Psychology: The scientific study of how people's thoughts, feelings, and behaviors are influenced by the actual, imagined, or implied presence of others.",
        "Neuropsychology: The study of the relationship between brain function and behavior."
    ],
    "Public Administration and Political Science": [
        "Comparative Politics: The study of the domestic politics, political institutions, and conflicts of countries.",
        "International Relations: The study of the interactions of states and other actors in the international system.",
        "Political Theory (Political Philosophy): The study of fundamental questions about the state, government, politics, liberty, justice and the enforcement of a legal code by authority.",
        "Public Policy: The study of how governmental policies are made and implemented.",
        "Public Management: The study of the implementation of government policy and the management of public sector organizations."
    ],
    "Religious Studies and Theology": [
        "Comparative Religion: The systematic comparison of the doctrines and practices of the world's religions.",
        "Sociology of Religion: The study of the beliefs, practices, and organizational forms of religion using the tools and methods of the discipline of sociology.",
        "Philosophy of Religion: The philosophical examination of the central themes and concepts involved in religious traditions.",
        "Biblical Studies: The academic application of a set of diverse disciplines to the study of the Bible (the Tanakh and the New Testament).",
        "Systematic Theology: A discipline of Christian theology that formulates an orderly, rational, and coherent account of the Christian faith and beliefs."
    ],
    "Science of Teaching (Education)": [
        "Curriculum and Instruction: The study of the design, development, and implementation of educational curricula and teaching methods.",
        "Educational Psychology: The study of how humans learn in educational settings, the effectiveness of educational interventions, the psychology of teaching, and the social psychology of schools as organizations.",
        "Educational Leadership and Administration: The study and practice of leading, managing, and administering educational institutions.",
        "Higher Education: The study of all aspects of post-secondary education, including its history, philosophy, governance, and role in society.",
        "Instructional Technology: The theory and practice of design, development, utilization, management, and evaluation of processes and resources for learning."
    ],
    "Sociology": [
        "Social Stratification: The study of the hierarchical arrangement of individuals into social classes, castes, and divisions in a society.",
        "Criminology: The scientific study of the nature, extent, management, causes, control, consequences, and prevention of criminal behavior, both on the individual and social levels.",
        "Urban Sociology: The sociological study of life and human interaction in metropolitan areas.",
        "Medical Sociology: The sociological analysis of medical organizations and institutions; the production of knowledge and selection of methods, the actions and interactions of healthcare professionals, and the social or cultural effects of medical practice.",
        "Political Sociology: The study of power and the intersection of society and politics."
    ]
}


def load_model_and_tokenizer(model_id: str):
    """
    Loads the specified model and tokenizer with optimizations for a large GPU.
    """
    print(f"Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def generate_conversation(model, tokenizer, topic, subfield, human_persona_details, task, num_turns):
    """
    Simulates a dynamic, multi-turn conversation driven by a specific task and subfield.
    """
    conversation_history = []
    participants = [human_persona_details, SHARE_PERSONA]

    # --- REVISED PROMPTING LOGIC ---

    # 1. Create a stricter system prompt that includes the subfield for more context.
    system_prompt = (
        f"You are a conversation simulator. Your task is to generate a single response at a time for one of two personas. The context for this conversation is **higher education** (e.g., university level).\n\n"
        f"**The General Topic:** {topic}\n"
        f"**The Specific Subfield for Discussion:** {subfield}\n"
        f"**The Task:** The human persona will initiate the conversation to accomplish the following task: '{task}'.\n\n"
        f"**Persona 1 (Human):**\n"
        f"- Name: {human_persona_details['name']}\n"
        f"- Role: {human_persona_details['description']}\n\n"
        f"**Persona 2 (AI Assistant):**\n"
        f"- Name: {SHARE_PERSONA['name']}\n"
        f"- Role: {SHARE_PERSONA['description']}\n\n"
        f"**IMPORTANT INSTRUCTIONS:**\n"
        f"1. You will be prompted to generate a response for one persona at a time. Generate ONLY that single response.\n"
        f"2. Do NOT add prefixes like 'Turn 1:' or '{human_persona_details['name']}:'. Just write the message content directly.\n"
        f"3. When generating responses for the **Human persona**, make them sound natural and direct. Humans usually get straight to the point. **Avoid having them summarize or rephrase the AI's previous response.** Instead, they should ask a direct follow-up question, state a point of confusion, or move the conversation forward."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # 2. Main loop for generating each turn of the conversation dynamically.
    for turn_index in range(num_turns * 2):

        current_persona = participants[turn_index % 2]
        is_share_turn = current_persona['key'] == 'share_ai'

        # Set generation parameters and thinking mode based on the current speaker.
        if is_share_turn:
            enable_thinking = True
            gen_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
            role_for_model = "assistant"
        else:
            enable_thinking = False
            gen_params = {"temperature": 0.75, "top_p": 0.85, "top_k": 30}
            role_for_model = "user"

        # 3. Add a temporary, explicit instruction for the current turn, now including the task.
        if turn_index == 0:
            instruction_content = f"Begin the conversation by generating an opening request from **{current_persona['name']}** about the subfield of **{subfield.split(':')[0]}** that aligns with their task: '{task}'."
        else:
            instruction_content = f"Now, generate the response for **{current_persona['name']}**."

        temp_messages = messages + [{"role": "user", "content": instruction_content}]

        # Apply the chat template with the temporary, guided message list.
        input_ids = tokenizer.apply_chat_template(
            temp_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=enable_thinking
        ).to(model.device)

        # Generate the response for the current persona.
        outputs = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            **gen_params
        )

        response_ids = outputs[0, input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        # Clean up the response to remove potential unwanted prefixes.
        response_text = response_text.removeprefix(f"{current_persona['name']}:").strip()

        # Add the clean response to our final history for saving.
        conversation_history.append({"role": current_persona['name'], "content": response_text})

        # Add the *actual* generated response to the permanent message history for context.
        messages.append({"role": role_for_model, "content": response_text})

    return {
        "topic": topic,
        "subfield": subfield,
        "persona": human_persona_details['name'],
        "task": task,
        "conversation": conversation_history
    }

def main():
    """
    Main function to run the data generation pipeline.
    """
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)

    print("\nStarting subfield-driven synthetic data generation...")

    # Create all combinations of topics, subfields, personas, and tasks.
    conversation_plan = []
    for topic, subfields in SSH_SUBFIELDS.items():
        for subfield in subfields:
            for p_key, p_details in HUMAN_PERSONAS.items():
                for task in p_details['tasks']:
                    conversation_plan.append((topic, subfield, p_key, task))

    print(f"Total conversations to generate: {len(conversation_plan)}")

    # Use a file handle to write JSON Lines, which is memory-efficient
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Use tqdm for a progress bar
        for topic, subfield, p_key, task in tqdm(conversation_plan, desc="Generating Conversations"):

            human_details = HUMAN_PERSONAS[p_key]

            # Generate one full conversation dynamically
            generated_data = generate_conversation(
                model=model,
                tokenizer=tokenizer,
                topic=topic,
                subfield=subfield,
                human_persona_details=human_details,
                task=task,
                num_turns=NUM_TURNS_PER_CONVERSATION
            )

            # Write the resulting JSON object as a single line in the file
            f.write(json.dumps(generated_data) + '\n')

    print(f"\nData generation complete. Output saved to '{OUTPUT_FILE}'.")
    print(f"Generated {len(conversation_plan)} conversations.")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
