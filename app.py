from flask import Flask, request, jsonify
import os
import json
from pinecone import Pinecone
import logging
import re

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
        raw_data = request.get_data(as_text=True)
        logger.info(f"Raw data length: {len(raw_data)}")
        
        # Try to fix common JSON issues
        try:
            # First attempt: direct parsing
            data = json.loads(raw_data)
            logger.info("Successfully parsed JSON directly")
        except json.JSONDecodeError as e:
            logger.info(f"Direct JSON parsing failed: {e}")
            try:
                # Second attempt: fix common issues
                fixed_data = raw_data.replace('\\"', '"').replace("'", "\\'")
                # Remove any problematic characters
                fixed_data = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', fixed_data)
                data = json.loads(fixed_data)
                logger.info("Successfully parsed JSON after fixing")
            except:
                logger.error("JSON parsing completely failed")
                return jsonify({'error': 'JSON parsing failed'}), 400
            
        logger.info(f"Data keys: {list(data.keys())}")
        
        # Initialize Pinecone
        api_key = os.environ.get('PINECONE_API_KEY')
        if not api_key:
            return jsonify({'error': 'Pinecone API key not configured'}), 500
            
        pc = Pinecone(api_key=api_key)
        index = pc.Index("tennis-knowledge-base")
        
        # Process embedding
        embedding = data.get('embedding', [])
        if isinstance(embedding, list):
            try:
                embedding = [float(x) for x in embedding]
            except:
                return jsonify({'error': 'Invalid embedding values'}), 400
        else:
            return jsonify({'error': 'Embedding must be a list'}), 400

        if len(embedding) != 3072:
            return jsonify({'error': f'Invalid embedding length: {len(embedding)}'}), 400
        
        logger.info(f"Processing {len(embedding)} dimension embedding")
        
        # Simple metadata
        metadata = {
            'content_id': str(data.get('content_id', 'unknown')),
            'summary': str(data.get('short_summary', ''))[:200]  # Truncate to avoid issues
        }
        
        # Create vector ID
        content_id = data.get('content_id', 'unknown')
        vector_id = f"tennis-{content_id}"
        
        logger.info(f"Storing vector: {vector_id}")
        
        # Store vector in Pinecone
        index.upsert(vectors=[(vector_id, embedding, metadata)])
        
        logger.info("SUCCESS: Vector stored in Pinecone!")
        
        return jsonify({
            'status': 'success',
            'vector_id': vector_id,
            'embedding_length': len(embedding)
        }), 200
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
