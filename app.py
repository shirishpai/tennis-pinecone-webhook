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
        raw_data = request.get_data(as_text=True)
        logger.info(f"Raw data length: {len(raw_data)}")
        logger.info(f"Raw data first 1000 chars: {raw_data[:1000]}")
        
        # Force JSON parsing only
        try:
            data = json.loads(raw_data)
            logger.info("Successfully parsed as JSON")
            logger.info(f"Data keys: {list(data.keys())}")
        except Exception as e:
            logger.error(f"JSON parsing failed with error: {str(e)}")
            logger.error(f"Raw data causing error: {raw_data}")
            return jsonify({'error': f'JSON parsing failed: {str(e)}'}), 400
            
        # Initialize Pinecone
        api_key = os.environ.get('PINECONE_API_KEY')
        if not api_key:
            logger.error("Pinecone API key not found")
            return jsonify({'error': 'Pinecone API key not configured'}), 500
            
        pc = Pinecone(api_key=api_key)
        index = pc.Index("tennis-knowledge-base")
        
        # Process embedding
        embedding = data.get('embedding', [])
        logger.info(f"Embedding type: {type(embedding)}")
        logger.info(f"Embedding length before processing: {len(embedding) if isinstance(embedding, list) else 'not a list'}")
        
        if isinstance(embedding, list):
            try:
                embedding = [float(x) for x in embedding]
                logger.info(f"Converted embedding to floats, length: {len(embedding)}")
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting embedding to floats: {e}")
                return jsonify({'error': 'Invalid embedding values'}), 400
        else:
            logger.error(f"Embedding is not a list, it's: {type(embedding)}")
            return jsonify({'error': f'Embedding must be a list, got {type(embedding)}'}), 400

        # Validate embedding length
        if len(embedding) != 3072:
            logger.error(f"Invalid embedding length: {len(embedding)} (expected 3072)")
            return jsonify({'error': f'Invalid embedding length: {len(embedding)} (expected 3072)'}), 400
        
        logger.info(f"Processed embedding: {len(embedding)} dimensions")
        
        # Simple metadata
        metadata = {
            'content_id': str(data.get('content_id', 'unknown')),
            'short_summary': str(data.get('short_summary', ''))[:300]
        }
        
        # Create vector ID
        content_id = data.get('content_id', 'unknown')
        vector_id = f"tennis-{content_id}"
        
        logger.info(f"About to store vector with ID: {vector_id}")
        
        # Store vector in Pinecone
        index.upsert(vectors=[(vector_id, embedding, metadata)])
        
        logger.info("SUCCESS: Vector stored in Pinecone!")
        
        return jsonify({
            'status': 'success',
            'vector_id': vector_id,
            'embedding_length': len(embedding)
        }), 200
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
