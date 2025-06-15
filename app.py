from flask import Flask, request, jsonify
import os
import json
from pinecone import Pinecone
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/health', methods=['GET'])
def health():
    return 'Healthy!', 200

@app.route('/store-vector', methods=['POST'])
def store_vector():
    try:
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Raw data: {request.get_data(as_text=True)}")
        
        data = request.get_json(force=True)
        
        except Exception as json_error:
            logger.error(f"JSON parsing error: {json_error}")
            return jsonify({'error': f'Invalid JSON: 400 Bad Request: The browser (or proxy) sent a request that this server could not understand.'}), 400
        
        if not data:
            return jsonify({'error': 'No data received'}), 400
            
        logger.info(f"Parsed data: {json.dumps(data, indent=2)}")
        
        # Initialize Pinecone
        api_key = os.environ.get('PINECONE_API_KEY')
        if not api_key:
            logger.error("Pinecone API key not found")
            return jsonify({'error': 'Pinecone API key not configured'}), 500
            
        pc = Pinecone(api_key=api_key)
        index = pc.Index("tennis-knowledge-base")
        
        # Process embedding with enhanced array handling
        embedding = data.get('embedding', [])
        if isinstance(embedding, str):
            # Handle comma-separated string
            try:
                embedding = [float(x.strip()) for x in embedding.split(',') if x.strip()]
            except ValueError:
                logger.error("Invalid embedding string format")
                return jsonify({'error': 'Invalid embedding format'}), 400
        elif isinstance(embedding, list):
            # Handle direct array - ensure all elements are floats
            try:
                embedding = [float(x) for x in embedding]
            except (ValueError, TypeError):
                logger.error("Invalid embedding array values")
                return jsonify({'error': 'Invalid embedding array values'}), 400
        else:
            logger.error(f"Embedding type not supported: {type(embedding)}")
            return jsonify({'error': 'Embedding must be array or string'}), 400

        # Validate embedding length
        if len(embedding) != 3072:
            logger.error(f"Invalid embedding length: {len(embedding)}")
            return jsonify({'error': f'Invalid embedding length: {len(embedding)} (expected 3072)'}), 400
        
        logger.info(f"Processed embedding: {len(embedding)} dimensions")
        
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
        
        # Create vector ID
        content_id = data.get('content_id', 'unknown')
        vector_id = f"tennis-{content_id}"
        
        logger.info(f"Storing vector with ID: {vector_id}")
        
        # Store vector in Pinecone
        index.upsert(vectors=[(vector_id, embedding, metadata)])
        
        logger.info("Vector stored successfully in Pinecone")
        
        return jsonify({
            'status': 'success',
            'vector_id': vector_id,
            'embedding_length': len(embedding),
            'metadata_fields': len(metadata)
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
