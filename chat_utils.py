import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Constants
PAST_CONVO_DIR = "./PastConvo"
os.makedirs(PAST_CONVO_DIR, exist_ok=True)

def save_conversation(messages: List[Dict[str, str]], session_id: Optional[str] = None) -> str:
    """
    Save conversation to a JSON file in the PastConvo directory.
    Returns the session_id used to save the file.
    """
    if not messages:
        return ""
        
    session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    file_path = os.path.join(PAST_CONVO_DIR, f"{session_id}.json")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'session_id': session_id,
                'created_at': datetime.now().isoformat(),
                'messages': messages
            }, f, indent=2, ensure_ascii=False)
        return session_id
    except Exception as e:
        print(f"Error saving conversation: {e}")
        return ""

def load_conversation(session_id: str) -> Optional[Dict[str, Any]]:
    """Load a specific conversation by session_id."""
    file_path = os.path.join(PAST_CONVO_DIR, f"{session_id}.json")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def list_conversations(limit: int = 10) -> List[Dict[str, str]]:
    """List available conversations, most recent first."""
    if not os.path.exists(PAST_CONVO_DIR):
        return []
        
    conversations = []
    for filename in sorted(os.listdir(PAST_CONVO_DIR), reverse=True):
        if not filename.endswith('.json'):
            continue
            
        file_path = os.path.join(PAST_CONVO_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conversations.append({
                    'session_id': data.get('session_id', filename),
                    'created_at': data.get('created_at', ''),
                    'message_count': len(data.get('messages', [])),
                    'preview': data.get('messages', [{}])[0].get('content', '')[:100] + '...' if data.get('messages') else ''
                })
                
            if len(conversations) >= limit:
                break
                
        except (json.JSONDecodeError, KeyError):
            continue
            
    return conversations

def delete_conversation(session_id: str) -> bool:
    """Delete a specific conversation by session_id."""
    file_path = os.path.join(PAST_CONVO_DIR, f"{session_id}.json")
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
    except Exception as e:
        print(f"Error deleting conversation {session_id}: {e}")
    return False
