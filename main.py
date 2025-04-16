import os
import json
import uuid
import requests
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

# --- Implémentation de la classe GeminiLLM compatible LangChain ---
class GeminiLLM(LLM):
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 500  # Augmentation du nombre de tokens pour générer une réponse complète
    api_key: str  # Clé API

    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 500, **kwargs):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("La clé GEMINI_API_KEY n'est pas définie dans l'environnement.")
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=key,
            **kwargs
        )

    def _call(self, prompt: str, stop=None) -> str:
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens
            }
        }
        headers = {
            "Content-Type": "application/json",
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        return response_json['candidates'][0]['content']['parts'][0]['text']

    @property
    def _identifying_params(self) -> dict:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    @property
    def _llm_type(self) -> str:
        return "gemini"

# Instanciation de l'LLM Gemini
gemini_llm = GeminiLLM(model_name="gemini-model-v1", temperature=0.7, max_tokens=500)

# Configuration du prompt template pour la génération de questions de quiz.
template = (
    "Génère une question à choix multiples pour un quiz dans le domaine '{domain}' "
    "avec un niveau de difficulté '{level}'. Fournis une question, quatre options de réponse, "
    "la bonne réponse, une explication détaillée sur pourquoi cette option est correcte et pourquoi "
    "les autres options ne le sont pas, ainsi qu'un lien vers une ressource permettant d'approfondir le sujet. "
    "Retourne le tout au format JSON suivant :\n"
    "{{\n"
    '  "question": "<texte de la question>",\n'
    '  "options": ["option1", "option2", "option3", "option4"],\n'
    '  "correct_answer": "<option correcte>",\n'
    '  "explanation": "<explication détaillée>",\n'
    '  "reference": "<lien de référence>"\n'
    "}}\n"
    "Assure-toi que le JSON est complet et se termine par '}}'."
)
prompt_template = PromptTemplate(input_variables=["domain", "level"], template=template)

# Chaînage du prompt et de l'appel au LLM pour les questions de quiz
llm_chain = prompt_template | gemini_llm

# Stockage en mémoire des sessions de quiz
quiz_sessions = {}

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

app = FastAPI(
    title="Quiz AI API avec LangChain & Gemini",
    description="Génère des questions de quiz dynamiques et permet à l'utilisateur d'y répondre, ainsi qu'un chatbot conversationnel.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fonction de réessai pour garantir que le JSON généré est complet
def generate_complete_response(prompt_data: dict, retries: int = 3) -> str:
    for attempt in range(retries):
        result_text = llm_chain.invoke(prompt_data)
        # Nettoyer la chaîne en supprimant les balises Markdown si présentes
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        if result_text.endswith("}"):
            return result_text
        # Vous pouvez ajouter du logging ici pour déboguer en cas de réponse incomplète
    # Retourner la dernière réponse, même incomplète, pour déclencher l'erreur suivante
    return result_text

# Modèles des requêtes et réponses pour le quiz
class QuizRequest(BaseModel):
    domain: str
    level: str

class QuestionOut(BaseModel):
    quiz_id: str
    question: str
    options: Dict[str, str]  # Options sous forme de dictionnaire : A, B, C, D
    explanation: str  # Explication détaillée
    reference: str  # Lien vers la ressource associée

class AnswerRequest(BaseModel):
    quiz_id: str
    answer: str

class AnswerFeedback(BaseModel):
    correct: bool
    message: str
    explanation: str

# Modèles des requêtes et réponses pour le chatbot
class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None  # Optionnel, permet éventuellement de gérer des sessions

class ChatResponse(BaseModel):
    chat_id: str
    reply: str

@app.post("/generate_question", response_model=QuestionOut)
async def generate_question(quiz_request: QuizRequest):
    try:
        # Utilisation de la fonction de réessai pour obtenir un JSON complet
        result_text = generate_complete_response({"domain": quiz_request.domain, "level": quiz_request.level})

        # Extraction du JSON basé sur la première "{" et la dernière "}"
        start = result_text.find("{")
        end = result_text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"La réponse n'est pas au format JSON attendu : {result_text}")

        json_text = result_text[start:end + 1]

        try:
            result_json = json.loads(json_text)
        except json.JSONDecodeError as je:
            raise ValueError(f"La réponse n'est pas au format JSON attendu : {json_text}") from je

        required_keys = {"question", "options", "correct_answer", "explanation", "reference"}
        if not required_keys.issubset(result_json.keys()):
            raise ValueError(f"Le JSON retourné n'a pas toutes les clés requises : {result_json}")

        # Génération d'un quiz_id unique et stockage des informations en session
        quiz_id = str(uuid.uuid4())
        quiz_sessions[quiz_id] = {
            "correct_answer": result_json["correct_answer"],
            "explanation": result_json["explanation"],
            "reference": result_json["reference"]
        }

        # Construction du dictionnaire des options
        options_list = result_json["options"]
        if not isinstance(options_list, list) or len(options_list) < 4:
            raise ValueError("Le champ options n'est pas une liste contenant au moins 4 éléments.")
        options_dict = {
            "A": options_list[0],
            "B": options_list[1],
            "C": options_list[2],
            "D": options_list[3]
        }

        output = {
            "quiz_id": quiz_id,
            "question": result_json["question"],
            "options": options_dict,
            "explanation": result_json["explanation"],
            "reference": result_json["reference"]
        }
        return QuestionOut(**output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de la question : {str(e)}")

@app.post("/submit_answer", response_model=AnswerFeedback)
async def submit_answer(answer_request: AnswerRequest):
    try:
        quiz_id = answer_request.quiz_id
        user_answer = answer_request.answer.strip()
        if quiz_id not in quiz_sessions:
            raise HTTPException(status_code=404, detail="Quiz non trouvé ou expiré.")

        # Récupération des informations stockées
        quiz_data = quiz_sessions.pop(quiz_id)
        correct_answer = quiz_data["correct_answer"]
        explanation = quiz_data.get("explanation", "")
        reference = quiz_data.get("reference", "")

        # Intégrer le lien dans l'explication
        detailed_explanation = f"{explanation}\n\nPour en savoir plus, consultez : {reference}"

        if user_answer.lower() == correct_answer.lower():
            feedback = AnswerFeedback(
                correct=True,
                message="Bravo, c'est la bonne réponse !",
                explanation=detailed_explanation
            )
        else:
            feedback = AnswerFeedback(
                correct=False,
                message=f"Dommage, la bonne réponse était : {correct_answer}",
                explanation=detailed_explanation
            )
        return feedback
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la soumission de la réponse : {str(e)}")

# Nouveau endpoint pour le chatbot
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Si aucun chat_id n'est fourni, on en génère un nouveau
        chat_id = request.chat_id or str(uuid.uuid4())
        # Construction d'un prompt pour le chatbot
        prompt = (
            "Vous êtes un chatbot conversationnel intelligent. "
            f"L'utilisateur a dit : \"{request.message}\" "
            "Répondez de manière claire, utile et conversationnelle."
        )
        # Appel direct au LLM pour générer la réponse
        reply = gemini_llm._call(prompt)
        return ChatResponse(chat_id=chat_id, reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de la réponse du chatbot : {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)
