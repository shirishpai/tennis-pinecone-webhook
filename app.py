import os
import json
from flask import Flask, request, jsonify
from pinecone import Pinecone
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index = pc.Index('tennis-knowledge-base')

@app.route('/store-vector', methods=['POST'])
def store_vector():
    try:
        data = request.json
        logger.info(f"Received data: {json.dumps(data, indent=2)}")
        
        content_id = data.get('content_id')
        embedding = data.get('embedding')
        
        metadata = {
            'content_id': data.get('content_id'),
            'content_text': data.get('content_text'),
            'youtube_url': data.get('youtube_url'),
            'short_summary': data.get('short_summary'),
            'key_takeaways': data.get('key_takeaways'),
            'detailed_analysis': data.get('detailed_analysis'),
            'tennis_topics': data.get('tennis_topics'),
            'skill_level': data.get('skill_level'),
            'coaching_style': data.get('coaching_style'),
            'player_references': data.get('player_references'),
            'common_problems': data.get('common_problems'),
            'key_tags': data.get('key_tags'),
            'full_transcript': data.get('full_transcript'),
            'equipment_required': data.get('equipment_required'),
            'time_investment': data.get('time_investment'),
            'solutions_provided': data.get('solutions_provided'),
            'user_keywords': data.get('user_keywords'),
            'immediate_actionable': data.get('immediate_actionable'),
            'created_time': data.get('created_time'),
            'video_title': data.get('video_title')
        }
        
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        if not content_id:
            return jsonify({'error': 'content_id is required'}), 400
        
        if not embedding:
            return jsonify({'error': 'embedding is required'}), 400
            
        if not isinstance(embedding, list):
            return jsonify({'error': 'embedding must be an array'}), 400
        
        vector_data = {
            'id': content_id,
            'values': embedding,
            'metadata': metadata
        }
        
        logger.info(f"Storing vector with ID: {content_id}")
        logger.info(f"Embedding dimensions: {len(embedding)}")
        logger.info(f"Metadata fields: {list(metadata.keys())}")
        
        index.upsert(vectors=[vector_data])
        
        return jsonify({
            'status': 'success',
            'message': f'Vector stored successfully with ID: {content_id}',
            'metadata_fields': len(metadata),
            'embedding_dimensions': len(embedding)
        })
        
    except Exception as e:
        logger.error(f"Error storing vector: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Tennis Knowledge Base Pinecone Integration'
    })

if __name__ == '__main__':
    if not os.environ.get('PINECONE_API_KEY'):
        logger.error("PINECONE_API_KEY environment variable is required")
        exit(1)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
