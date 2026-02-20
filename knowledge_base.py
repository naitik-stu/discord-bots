import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import os

class KnowledgeBase:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.questions = []
        self.answers = []
        self.question_embeddings = None
        self.index = None
        
    def load_training_data(self, file_path: str):
        """Load training data from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse Q&A pairs (format: Q: question\nA: answer)
            qa_pairs = []
            sections = content.split('Q:')
            
            for section in sections[1:]:  # Skip first empty section
                if 'A:' in section:
                    question = section.split('A:')[0].strip()
                    answer = section.split('A:')[1].strip()
                    if question and answer:
                        qa_pairs.append((question, answer))
            
            self.questions = [qa[0] for qa in qa_pairs]
            self.answers = [qa[1] for qa in qa_pairs]
            
            print(f"Loaded {len(self.questions)} Q&A pairs")
            return True
            
        except FileNotFoundError:
            print(f"Training data file {file_path} not found")
            return False
        except Exception as e:
            print(f"Error loading training data: {e}")
            return False
    
    def build_index(self):
        """Build FAISS index for fast similarity search"""
        if not self.questions:
            print("No questions loaded")
            return False
            
        print("Building embeddings...")
        self.question_embeddings = self.model.encode(
            self.questions, 
            convert_to_tensor=False,
            show_progress_bar=True
        )
        
        # Create FAISS index
        dimension = self.question_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.question_embeddings).astype('float32'))
        
        print(f"Index built with {len(self.questions)} questions")
        return True
    
    def save_index(self, file_path: str):
        """Save FAISS index to disk"""
        if self.index is not None:
            faiss.write_index(self.index, file_path)
            print(f"Index saved to {file_path}")
    
    def load_index(self, file_path: str):
        """Load FAISS index from disk"""
        if os.path.exists(file_path):
            self.index = faiss.read_index(file_path)
            print(f"Index loaded from {file_path}")
            return True
        return False
    
    def find_best_answer(self, query: str, threshold: float = 0.7) -> Tuple[str, float]:
        """Find the best answer for a given query"""
        if self.index is None or not self.questions:
            return "I don't have information about that topic. Please contact a server admin.", 0.0
        
        # Preprocess query to improve matching
        processed_query = self.preprocess_query(query)
        
        # Encode query
        query_embedding = self.model.encode([processed_query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search for similar questions
        distances, indices = self.index.search(query_embedding, k=5)  # Search top 5 instead of 3
        
        # Convert L2 distance to similarity score
        best_distance = distances[0][0]
        similarity = 1 / (1 + best_distance)  # Convert to similarity score
        
        # Use a lower threshold for better coverage
        adjusted_threshold = max(threshold - 0.1, 0.5)  # Don't go below 0.5
        
        if similarity >= adjusted_threshold and indices[0][0] < len(self.answers):
            return self.answers[indices[0][0]], similarity
        
        return "I don't have a good answer for that question. Please contact a server admin for help.", similarity
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess query to improve semantic matching"""
        # Convert to lowercase
        query = query.lower()
        
        # Common question variations to normalize
        variations = {
            # Timezone variations
            'timezone of the server': 'server timezone',
            'server timezone': 'server timezone',
            'what timezone': 'server timezone',
            'time zone': 'timezone',
            'server time': 'server timezone',
            
            # Rules variations
            'rules of the server': 'server rules',
            'server rules': 'server rules',
            'what are the rules': 'server rules',
            'guidelines': 'server rules',
            
            # Getting started variations
            'how to start': 'get started',
            'getting started': 'get started',
            'new here': 'get started',
            'beginner': 'get started',
            
            # Role variations
            'how to get roles': 'get roles',
            'assign roles': 'get roles',
            'role assignment': 'get roles',
            
            # Moderator variations
            'who are mods': 'moderators',
            'admin': 'moderators',
            'staff': 'moderators',
            
            # Report variations
            'report issue': 'report problem',
            'report someone': 'report problem',
            'complaint': 'report problem',
            
            # Voice channel variations
            'voice channels': 'voice channels',
            'vc': 'voice channels',
            'voice chat': 'voice channels',
            
            # Invite variations
            'invite friends': 'invite friends',
            'add friends': 'invite friends',
            'share server': 'invite friends',
            
            # Music variations
            'music bot': 'music bot',
            'play music': 'music bot',
            'songs': 'music bot',
            
            # Events variations
            'server events': 'events',
            'activities': 'events',
            'what events': 'events',
            
            # Suggestion variations
            'suggest feature': 'suggest new features',
            'ideas': 'suggest new features',
            'feedback': 'suggest new features',
            
            # Level variations
            'rank up': 'level up',
            'ranking': 'level up',
            'experience': 'level up',
            
            # Channel variations
            'create channel': 'create my own channel',
            'new channel': 'create my own channel',
            
            # Technical support variations
            'tech help': 'technical issues',
            'support': 'technical issues',
            'problem': 'technical issues',
        }
        
        # Apply variations
        for variation, standard in variations.items():
            if variation in query:
                query = query.replace(variation, standard)
        
        # Remove common question words that don't add semantic value
        filler_words = ['what is', 'what are', 'how do i', 'can i', 'could you', 'please', 'thank you', 'thanks']
        for filler in filler_words:
            query = query.replace(filler, '')
        
        return query.strip()
    
    def add_qa_pair(self, question: str, answer: str):
        """Add a new Q&A pair to the knowledge base"""
        self.questions.append(question)
        self.answers.append(answer)
        
        # Rebuild index with new data
        self.build_index()
