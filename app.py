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

        try:
            data = json.loads(raw_data)
            logger.info("Successfully parsed as JSON")
            logger.info(f"Data keys: {list(data.keys())}")
        except Exception as e:
            logger.error(f"JSON parsing failed with error: {str(e)}")
            logger.error(f"Raw data causing error: {raw_data}")
            return jsonify({'error': f'JSON parsing failed: {str(e)}'}), 400

        # Validate vector input
        vectors = data.get('vectors', [])
        if not vectors or not isinstance(vectors, list):
            logger.error("Missing or invalid 'vectors' field")
            return jsonify({'error': "Missing or invalid 'vectors' field"}), 400

        vector = vectors[0]
        embedding = vector.get('values', [])
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

        if len(embedding) != 3072:
            logger.error(f"Invalid embedding length: {len(embedding)} (expected 3072)")
            return jsonify({'error': f'Invalid embedding length: {len(embedding)} (expected 3072)'}), 400

        metadata = vector.get('metadata', {})
        vector_id = vector.get('id', 'tennis-unknown')

        logger.info(f"About to store vector with ID: {vector_id}")
        logger.info(f"Metadata: {metadata}")

        # Initialize Pinecone
        api_key = os.environ.get('PINECONE_API_KEY')
        if not api_key:
            logger.error("Pinecone API key not found")
            return jsonify({'error': 'Pinecone API key not configured'}), 500

        pc = Pinecone(api_key=api_key)
        index = pc.Index("tennis-knowledge-base")

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
