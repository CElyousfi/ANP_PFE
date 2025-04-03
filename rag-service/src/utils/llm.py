# rag-service/src/utils/llm.py
import os
import logging
import time
import json
from typing import Dict, Any, List, Optional
import re

logger = logging.getLogger(__name__)

# Check if Groq is available
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq package not installed. Using mock LLM.")

class LLMService:
    """Service for interacting with Language Models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LLM service with configuration."""
        self.provider = config.get("provider", "mock")
        self.model = config.get("model", "llama-3.1-8b-instant")
        self.temperature = config.get("temperature", 0.5)
        self.max_tokens = config.get("max_tokens", 4096)
        self.api_key = config.get("api_key") or os.getenv("GROQ_API_KEY")
        
        self.client = None
        self.last_request_time = 0
        
        # Initialize real client if possible
        if self.provider == "groq" and GROQ_AVAILABLE and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info(f"Initialized Groq client with model {self.model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
                self.provider = "mock"
        else:
            logger.info("Using mock LLM for development/testing")
            self.provider = "mock"
    
    def generate_text(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate text using the LLM."""
        start_time = time.time()
        
        # Use default values if not provided
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            if self.provider == "groq" and self.client:
                # Generate with Groq
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                
                # Add response format if provided
                if response_format:
                    kwargs["response_format"] = response_format
                
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                self.last_request_time = time.time() - start_time
                return content
            else:
                # Generate more sophisticated mock responses based on the context
                self.last_request_time = 0.1
                return self._generate_improved_mock_response(messages, response_format)
                
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            self.last_request_time = time.time() - start_time
            
            # Return error message as fallback
            if response_format and response_format.get("type") == "json_object":
                return json.dumps({"error": str(e)})
            return f"Error generating response: {str(e)}"
    
    def _generate_improved_mock_response(self, messages, response_format=None):
        """Generate a mock response that actually uses the context provided."""
        # Extract the user query and context from messages
        user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        
        # Check if this is a JSON response request
        if response_format and response_format.get("type") == "json_object":
            if "context adequacy" in user_message.lower():
                return json.dumps({
                    "is_adequate": True,
                    "confidence": 0.85,
                    "reason": "Context appears to contain relevant information"
                })
            
            if "evaluate" in user_message.lower():
                return json.dumps({
                    "answer_relevance": 0.8,
                    "groundedness": 0.9
                })
            
            return json.dumps({
                "response": "This is a mock response",
                "confidence": 0.7
            })
        
        # Check if it's a greeting
        greeting_patterns = [
            r"\b(?:hi|hello|hey|bonjour|salut|hola|good morning|good afternoon|good evening)\b",
            r"\bhow are you\b",
            r"\bnice to meet you\b",
            r"\bthanks?\b|\bthank you\b|\bmerci\b"
        ]
        
        is_greeting = any(re.search(pattern, user_message.lower()) for pattern in greeting_patterns)
        
        if is_greeting:
            # Detect language for greeting response
            if "english" in system_message.lower():
                return "Hello! I'm the ANP Assistant for the National Ports Agency of Morocco. How can I help you today with port-related information?"
            elif "arabic" in system_message.lower():
                return "مرحبًا! أنا مساعد ANP لوكالة الموانئ الوطنية بالمغرب. كيف يمكنني مساعدتك اليوم بمعلومات متعلقة بالموانئ؟"
            elif "spanish" in system_message.lower():
                return "¡Hola! Soy el Asistente ANP de la Agencia Nacional de Puertos de Marruecos. ¿Cómo puedo ayudarle hoy con información relacionada con puertos?"
            else:
                return "Bonjour ! Je suis l'Assistant ANP de l'Agence Nationale des Ports du Maroc. Comment puis-je vous aider aujourd'hui avec des informations portuaires ?"
        
        # Extract context from user message if available
        context_match = re.search(r"base de connaissances:\s*---------------------\s*(.*?)\s*---------------------", user_message, re.DOTALL)
        context = context_match.group(1) if context_match else ""
        
        # Extract query from user message if available
        query_match = re.search(r"Question de l'utilisateur:\s*\"(.*?)\"", user_message)
        query = query_match.group(1) if query_match else ""
        if not query:
            query_match = re.search(r"query:\s*\"(.*?)\"", user_message, re.IGNORECASE)
            query = query_match.group(1) if query_match else ""
        
        # If we have context, try to extract relevant info, otherwise return a basic response
        if context:
            # Look for context relevant to specific topics in the query
            if "article 3" in query.lower() or "environmental" in query.lower() or "environnement" in query.lower():
                if "Article 3: Environmental Policies" in context:
                    return """D'après le règlement portuaire, l'Article 3 concerne les Politiques Environnementales. 

Le port suit des directives environnementales strictes pour minimiser la pollution :
- Toute élimination des déchets doit suivre les protocoles établis
- Les plans de gestion des eaux de ballast doivent être soumis et approuvés
- Des mesures de prévention des déversements d'hydrocarbures doivent être en place pour tous les navires
- Les émissions atmosphériques doivent respecter les normes nationales et internationales
- Les niveaux de bruit doivent être maintenus au minimum, en particulier pendant les opérations nocturnes

Ces mesures environnementales sont essentielles pour assurer des opérations portuaires durables et conformes aux réglementations."""
                    
            if "safety" in query.lower() or "sécurité" in query.lower() or "gear" in query.lower() or "équipement" in query.lower():
                if "Safety Regulations" in context or "safety gear" in context.lower():
                    return """Selon les règlements portuaires, voici les exigences en matière d'équipement de sécurité :

1. Tout le personnel doit porter un équipement de sécurité approprié dans les zones désignées, notamment :
   - Casques de protection
   - Gilets haute visibilité
   - Chaussures de sécurité

2. Les procédures d'urgence doivent être clairement affichées sur tous les navires et dans toutes les installations portuaires.

3. Des exercices de sécurité sont régulièrement menés chaque mois pour assurer la préparation aux situations d'urgence.

4. Les navires transportant des matières dangereuses doivent afficher les signaux d'avertissement appropriés et en informer les autorités portuaires à l'avance.

5. La limite de vitesse dans les eaux portuaires est de 5 nœuds, sauf indication contraire.

Ces mesures visent à garantir la sécurité de toutes les personnes travaillant dans la zone portuaire."""
                    
            if "pilotage" in query.lower() or "fees" in query.lower() or "tariffs" in query.lower() or "tarifs" in query.lower() or "frais" in query.lower():
                if "Pilotage Fees" in context or "tariff" in context.lower():
                    return """Concernant les frais de pilotage pour les navires, voici les tarifs selon les régulations portuaires :

Le pilotage est obligatoire pour tous les navires de plus de 100 TJB (Tonnage de Jauge Brute).

Les tarifs sont structurés comme suit :
| Taille du navire (TJB) | Tarif (MAD) |
|------------------------|-------------|
| 0 - 500                | 750         |
| 501 - 1 000            | 1 250       |
| 1 001 - 5 000          | 2 000       |
| 5 001 - 10 000         | 3 000       |
| 10 001 - 20 000        | 4 250       |
| > 20 000               | 5 500       |

Ces frais sont calculés en fonction du tonnage du navire et sont essentiels pour garantir une navigation sécurisée dans les zones portuaires."""
                    
            if "container" in query.lower() or "containers" in query.lower() or "conteneurs" in query.lower() or "storage" in query.lower() or "stockage" in query.lower():
                if "Container" in context or "storage" in context.lower():
                    return """Concernant les frais de stockage pour les conteneurs dans les ports, selon les informations disponibles :

Les conteneurs bénéficient d'une période de franchise de 7 jours.

Après cette période, les frais de stockage s'appliquent selon le barème suivant :
- Première période (jours 8-15) : 150 MAD par conteneur et par jour
- Seconde période (après le jour 15) : 300 MAD par conteneur et par jour

Par ailleurs, pour la manutention des conteneurs, les tarifs sont les suivants :
| Service                   | Conteneur 20' (MAD) | Conteneur 40' (MAD) |
|---------------------------|---------------------|---------------------|
| Déchargement/Chargement   | 750                 | 1 125               |
| Déplacement à bord        | 375                 | 560                 |
| Entrée/sortie             | 250                 | 375                 |
| Connexion reefer (par jour)| 300                | 450                 |
| Supplément IMDG           | +50%                | +50%                |

Ces tarifs sont établis pour garantir une gestion efficace de l'espace portuaire tout en offrant un service de qualité aux opérateurs."""
                    
            if "crane" in query.lower() or "cranes" in query.lower() or "grue" in query.lower() or "grues" in query.lower() or "technical" in query.lower() or "technique" in query.lower():
                if "Crane Specifications" in context:
                    return """Concernant les spécifications techniques des grues portuaires, voici les informations disponibles :

1. Grues Ship-to-Shore (STS) :
   - Capacité de levage : 65 tonnes sous spreader
   - Portée : 65 mètres
   - Hauteur de levage : 42 mètres au-dessus du rail
   - Vitesse de levage : 90 mètres/min avec charge
   - Vitesse du chariot : 180 mètres/min
   - Vitesse de déplacement : 45 mètres/min
   - Alimentation électrique : 11kV/60Hz

2. Grues sur pneumatiques (RTG) :
   - Capacité de levage : 40 tonnes sous spreader
   - Hauteur de levage : 18,5 mètres (6+1 conteneurs)
   - Envergure : 23,6 mètres
   - Vitesse de levage : 30 mètres/min avec charge
   - Vitesse du chariot : 70 mètres/min
   - Vitesse de déplacement : 130 mètres/min
   - Énergie : Diesel-électrique ou électrique avec enrouleur de câble

Ces équipements sont essentiels pour les opérations de chargement et déchargement des navires dans nos ports."""
                    
            if "maintenance" in query.lower() or "entretien" in query.lower() or "equipment" in query.lower() or "équipement" in query.lower():
                if "Maintenance Requirements" in context:
                    return """Concernant la fréquence d'entretien des équipements portuaires, voici les exigences réglementaires :

Tout équipement portuaire doit être entretenu selon le calendrier suivant :
- Inspections quotidiennes : Avant chaque poste de travail
- Entretien hebdomadaire : Lubrification et ajustements mineurs
- Entretien mensuel : Vérification complète des systèmes
- Entretien trimestriel : Révision majeure des systèmes
- Certification annuelle : Vérification de la sécurité et de la capacité

Les registres d'entretien doivent être conservés pendant au moins 5 ans et être disponibles pour inspection à tout moment.

Ce calendrier d'entretien rigoureux garantit la sécurité et l'efficacité des opérations portuaires, tout en prolongeant la durée de vie des équipements."""
                    
            if "registration" in query.lower() or "register" in query.lower() or "vessel" in query.lower() or "ships" in query.lower() or "navire" in query.lower() or "bateau" in query.lower() or "enregistrement" in query.lower():
                if "vessels must register" in context.lower():
                    return """Selon les règlements portuaires, tous les navires doivent s'enregistrer auprès des autorités portuaires au moins 24 heures avant leur arrivée.

Cette procédure d'enregistrement nécessite la présentation des documents suivants :
- Préavis d'arrivée du navire (72 heures)
- Documentation de dédouanement
- Manifeste de cargaison
- Déclaration de marchandises dangereuses (le cas échéant)
- Liste d'équipage et liste des passagers (le cas échéant)
- Déclaration maritime de santé
- Certificat international de sûreté du navire

Cette exigence permet aux autorités portuaires de planifier efficacement les opérations, d'assurer la sécurité et de préparer les services nécessaires pour l'accueil du navire."""
                    
            if "documents" in query.lower() or "documentation" in query.lower() or "document" in query.lower():
                if "Required Documentation" in context or "document" in context.lower():
                    return """Selon les procédures administratives portuaires, les documents requis pour les navires comprennent :

1. Préavis d'arrivée du navire (72 heures avant l'arrivée)
2. Documentation de dédouanement
3. Manifeste de cargaison
4. Déclaration de marchandises dangereuses (le cas échéant)
5. Liste d'équipage et liste des passagers (le cas échéant)
6. Déclaration maritime de santé
7. Certificat international de sûreté du navire

Ces documents sont essentiels pour assurer le respect des réglementations nationales et internationales, garantir la sécurité et faciliter les opérations portuaires efficaces. Tous les documents doivent être soumis dans les délais prescrits pour éviter des retards dans le traitement et l'accostage du navire."""
            
            # Detect language for response
            if "english" in system_message.lower() or re.search(r"respond in\s+english", system_message.lower()):
                language = "english"
            elif "arabic" in system_message.lower():
                language = "arabic"
            elif "spanish" in system_message.lower():
                language = "spanish"
            else:
                language = "french"
            
            # Provide a more relevant response based on document context
            if "port regulations" in context.lower() or "règlement" in context.lower():
                if language == "english":
                    return """Based on the port regulations documentation, here are the key points:

1. All vessels must register with port authorities at least 24 hours before arrival at any Moroccan port.
2. Safety protocols must be strictly followed during loading and unloading operations.
3. Environmental compliance is mandatory for all vessels, including waste disposal and emissions standards.
4. Speed limit within port waters is 5 knots unless otherwise specified.
5. Emergency procedures must be clearly displayed on all vessels and port facilities.

These regulations are enforced to ensure safe and efficient port operations. For more specific information about particular aspects of port regulations, please provide details about the area of interest."""
                elif language == "arabic":
                    return """بناءً على وثائق لوائح الميناء، إليك النقاط الرئيسية:

1. يجب على جميع السفن التسجيل لدى سلطات الميناء قبل 24 ساعة على الأقل من وصولها إلى أي ميناء مغربي.
2. يجب اتباع بروتوكولات السلامة بدقة أثناء عمليات التحميل والتفريغ.
3. الامتثال البيئي إلزامي لجميع السفن، بما في ذلك التخلص من النفايات ومعايير الانبعاثات.
4. حد السرعة داخل مياه الميناء هو 5 عقد ما لم يُذكر خلاف ذلك.
5. يجب عرض إجراءات الطوارئ بوضوح على جميع السفن ومرافق الميناء.

يتم تطبيق هذه اللوائح لضمان عمليات ميناء آمنة وفعالة. لمزيد من المعلومات المحددة حول جوانب معينة من لوائح الميناء، يرجى تقديم تفاصيل حول مجال الاهتمام."""
                elif language == "spanish":
                    return """Según la documentación de las regulaciones portuarias, aquí están los puntos clave:

1. Todos los buques deben registrarse con las autoridades portuarias al menos 24 horas antes de su llegada a cualquier puerto marroquí.
2. Los protocolos de seguridad deben seguirse estrictamente durante las operaciones de carga y descarga.
3. El cumplimiento ambiental es obligatorio para todos los buques, incluida la eliminación de residuos y los estándares de emisiones.
4. El límite de velocidad dentro de las aguas portuarias es de 5 nudos a menos que se especifique lo contrario.
5. Los procedimientos de emergencia deben mostrarse claramente en todos los buques e instalaciones portuarias.

Estas regulaciones se aplican para garantizar operaciones portuarias seguras y eficientes. Para obtener información más específica sobre aspectos particulares de las regulaciones portuarias, proporcione detalles sobre el área de interés."""
                else:  # French
                    return """D'après la documentation sur les règlements portuaires, voici les points essentiels :

1. Tous les navires doivent s'enregistrer auprès des autorités portuaires au moins 24 heures avant leur arrivée dans tout port marocain.
2. Les protocoles de sécurité doivent être strictement suivis pendant les opérations de chargement et de déchargement.
3. La conformité environnementale est obligatoire pour tous les navires, y compris l'élimination des déchets et les normes d'émissions.
4. La limite de vitesse dans les eaux portuaires est de 5 nœuds sauf indication contraire.
5. Les procédures d'urgence doivent être clairement affichées sur tous les navires et installations portuaires.

Ces règlements sont appliqués pour garantir des opérations portuaires sûres et efficaces. Pour des informations plus spécifiques sur des aspects particuliers des règlements portuaires, veuillez fournir des détails sur le domaine d'intérêt."""

        # If all else fails, provide a generic response in the appropriate language
        if re.search(r"english", system_message.lower()):
            return """Based on the available information, here is my response:

Port regulations require all vessels to register with port authorities 24 hours prior to arrival. Safety protocols must be strictly followed during all loading and unloading operations, and environmental compliance is mandatory for all vessels operating in port waters.

These regulations ensure the safe and efficient operation of all port facilities while protecting the marine environment. For specific regulatory details regarding your particular situation, please provide more details about your requirements."""
        elif re.search(r"arabic", system_message.lower()):
            return """بناءً على المعلومات المتاحة، إليك ردي:

تتطلب لوائح الميناء من جميع السفن التسجيل لدى سلطات الميناء قبل 24 ساعة من الوصول. يجب اتباع بروتوكولات السلامة بدقة خلال جميع عمليات التحميل والتفريغ، والامتثال البيئي إلزامي لجميع السفن العاملة في مياه الميناء.

تضمن هذه اللوائح التشغيل الآمن والفعال لجميع مرافق الميناء مع حماية البيئة البحرية. للحصول على تفاصيل تنظيمية محددة بخصوص حالتك الخاصة، يرجى تقديم المزيد من التفاصيل حول متطلباتك."""
        elif re.search(r"spanish", system_message.lower()):
            return """Según la información disponible, aquí está mi respuesta:

Las regulaciones portuarias requieren que todas las embarcaciones se registren con las autoridades portuarias 24 horas antes de su llegada. Los protocolos de seguridad deben seguirse estrictamente durante todas las operaciones de carga y descarga, y el cumplimiento ambiental es obligatorio para todas las embarcaciones que operan en aguas portuarias.

Estas regulaciones garantizan el funcionamiento seguro y eficiente de todas las instalaciones portuarias mientras protegen el medio ambiente marino. Para detalles regulatorios específicos sobre su situación particular, proporcione más detalles sobre sus requisitos."""
        else:  # Default to French
            return """Selon les informations disponibles, voici ma réponse :

Les règlements portuaires exigent que tous les navires s'inscrivent auprès des autorités portuaires 24 heures avant leur arrivée. Les protocoles de sécurité doivent être strictement suivis pendant toutes les opérations de chargement et de déchargement, et la conformité environnementale est obligatoire pour tous les navires opérant dans les eaux portuaires.

Ces règlements assurent le fonctionnement sûr et efficace de toutes les installations portuaires tout en protégeant l'environnement marin. Pour des détails réglementaires spécifiques concernant votre situation particulière, veuillez fournir plus de détails sur vos besoins."""
    
    def health_check(self) -> bool:
        """Check if the LLM service is healthy."""
        if self.provider == "mock":
            return True
            
        try:
            if self.client:
                # Try to generate a short response as a health check
                test_messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"}
                ]
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=test_messages,
                    temperature=0.1,
                    max_tokens=5
                )
                return bool(response.choices[0].message.content)
            return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


def get_llm_service(config: Dict[str, Any]) -> LLMService:
    """Get an LLM service based on configuration."""
    return LLMService(config)