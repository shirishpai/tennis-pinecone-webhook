from http.server import BaseHTTPRequestHandler
import json
import os
from pinecone import Pinecone

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Healthy!')
            return
        
        self.send_response(404)
        self.end_headers()
    
    def do_POST(self):
        if self.path == '/store-vector':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                # Initialize Pinecone
                pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
                index = pc.Index("tennis-knowledge-base")
                
                # Process embedding
                embedding = data.get('embedding', [])
                if isinstance(embedding, str):
                    embedding = [float(x.strip()) for x in embedding.split(',')]
                
                # Prepare metadata
                metadata = {
                    'content_id': data.get('content_id', ''),
                    'short_summary': data.get('short_summary', ''),
                    'key_takeaways': data.get('key_takeaways', ''),
                    'detailed_analysis': data.get('detailed_analysis', ''),
                    'tennis_topics': data.get('tennis_topics', ''),
                    'coaching_style': data.get('coaching_style', ''),
                    'skill_level': data.get('skill_level', ''),
                    'player_references': data.get('player_references', ''),
                    'common_problems': data.get('common_problems', ''),
                    'key_tags': data.get('key_tags', ''),
                    'equipment_required': data.get('equipment_required', ''),
                    'time_investment': data.get('time_investment', ''),
                    'solutions_provided': data.get('solutions_provided', ''),
                    'user_keywords': data.get('user_keywords', ''),
                    'immediate_actionable': data.get('immediate_actionable', ''),
                    'video_title': data.get('video_title', ''),
                    'full_transcript': data.get('full_transcript', ''),
                    'content_text': data.get('content_text', ''),
                    'youtube_url': data.get('youtube_url', '')
                }
                
                # Store vector
                vector_id = f"tennis-{data.get('content_id', 'unknown')}"
                index.upsert(vectors=[(vector_id, embedding, metadata)])
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response = {'status': 'success', 'vector_id': vector_id}
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = {'error': str(e)}
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_response(404)
            self.end_headers()
