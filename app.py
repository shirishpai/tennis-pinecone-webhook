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
        logger.info(f"Raw data preview: {raw_data[:200]}...")
        
        # Try multiple parsing approaches
        data = None
        
        # Approach 1: Standard JSON
        try:
            data = request.get_json(force=True)
            logger.info("Successfully parsed as JSON")
        except Exception as e:
            logger.info(f"JSON parsing failed: {e}")
            
        # Approach 2: Form data
        if data is None:
            try:
                data = request.form.to_dict()
                logger.info("Successfully parsed as form data")
                # Convert embedding string to array if needed
                if 'embedding' in data and isinstance(data['embedding'], str):
                    data['embedding'] = [float(x.strip()) for x in data['embedding'].split(',') if x.strip()]
            except Exception as e:
                logger.info(f"Form parsing failed: {e}")
                
        # Approach 3: Manual JSON parse
        if data is None:
            try:
                data = json.loads(raw_data)
                logger.info("Successfully parsed raw JSON")
            except Exception as e:
                logger.info(f"Raw JSON parsing failed: {e}")
                
        if data is None:
            logger.error("All parsing methods failed")
            return jsonify({'error': 'Could not parse request data'}), 400
            
        logger.info(f"Successfully parsed data with keys: {list(data.keys())}")
        
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
        
        # Helper function to safely get and truncate fields
        def safe_truncate(value, max_length=500):
            if value is None:
                return ''
            str_value = str(value)
            if len(str_value) > max_length:
                return str_value[:max_length] + "..."
            return str_value
        
        # Minimal essential metadata to ensure success
        metadata = {
            'content_id': safe_truncate(data.get('content_id', ''), 100),
            'short_summary': safe_truncate(data.get('short_summary', ''), 300),
            'tennis_topics': safe_truncate(data.get('tennis_topics', ''), 200),
            'skill_level': safe_truncate(data.get('skill_level', ''), 50),
            'coaching_style': safe_truncate(data.get('coaching_style', ''), 100)
        }
        
        # Create vector ID
        content_id = data.get('content_id', 'unknown')
        vector_id = f"tennis-{content_id}"
        
        logger.info(f"Storing vector with ID: {vector_id}")
        logger.info(f"Metadata: {metadata}")
        
        # Store vector in Pinecone
        index.upsert(vectors=[(vector_id, embedding, metadata)])
        
        logger.info("Vector stored successfully in Pinecone!")
        
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
