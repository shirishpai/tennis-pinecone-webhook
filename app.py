from flask import Flask, request, jsonify
import os
import json
import requests
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
        
        # Try to fix common JSON issues
        try:
            data = json.loads(raw_data)
            logger.info("Successfully parsed JSON directly")
        except json.JSONDecodeError as e:
            logger.info(f"Direct JSON parsing failed: {e}")
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
            'summary': str(data.get('short_summary', ''))[:200]
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

@app.route('/process-airtable', methods=['GET', 'POST'])
def process_airtable():
    """
    Complete Airtable → OpenAI → Pinecone processing
    """
    try:
        logger.info("Starting Airtable processing...")
        
        # Airtable configuration
        AIRTABLE_BASE_ID = "appEa8P6iWB6YTqyE"  # Your Tennis Coaching Database
        AIRTABLE_TABLE = "Tennis Knowledge Database"
        AIRTABLE_API_KEY = os.environ.get('AIRTABLE_API_KEY')
        OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
        PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
        
        if not all([AIRTABLE_API_KEY, OPENAI_API_KEY, PINECONE_API_KEY]):
            return jsonify({'error': 'Missing API keys. Please set AIRTABLE_API_KEY, OPENAI_API_KEY, and PINECONE_API_KEY environment variables.'}), 500
        
        # Get records from Airtable
        airtable_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE}"
        headers = {
            'Authorization': f'Bearer {AIRTABLE_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Fetch records (limit to 5 for testing)
        params = {'maxRecords': 5}
        response = requests.get(airtable_url, headers=headers, params=params)
        
        if response.status_code != 200:
            logger.error(f"Airtable API error: {response.text}")
            return jsonify({'error': f'Airtable API error: {response.status_code}'}), 500
        
        records = response.json().get('records', [])
        logger.info(f"Retrieved {len(records)} records from Airtable")
        
        if not records:
            return jsonify({'message': 'No records found in Airtable'}), 200
        
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("tennis-knowledge-base")
        
        processed_count = 0
        
        for record in records:
            try:
                record_id = record['id']
                fields = record.get('fields', {})
                
                # Get content for embedding (combine available text fields)
                content_parts = []
                for field in ['Video Title', 'Full Transcript', 'Short Summary']:
                    if field in fields and fields[field]:
                        content_parts.append(str(fields[field]))
                
                if not content_parts:
                    logger.warning(f"No content found for record {record_id}")
                    continue
                
                content_text = " ".join(content_parts)
                logger.info(f"Processing record {record_id}: {content_text[:100]}...")
                
                # Generate embedding with OpenAI
                openai_url = "https://api.openai.com/v1/embeddings"
                openai_headers = {
                    'Authorization': f'Bearer {OPENAI_API_KEY}',
                    'Content-Type': 'application/json'
                }
                openai_data = {
                    'input': content_text,
                    'model': 'text-embedding-3-large'
                }
                
                embedding_response = requests.post(openai_url, headers=openai_headers, json=openai_data)
                
                if embedding_response.status_code != 200:
                    logger.error(f"OpenAI API error for record {record_id}: {embedding_response.text}")
                    continue
                
                embedding_data = embedding_response.json()
                embedding = embedding_data['data'][0]['embedding']
                
                logger.info(f"Generated {len(embedding)} dimension embedding for record {record_id}")
                
                # Prepare metadata
                metadata = {
                    'content_id': record_id,
                    'video_title': str(fields.get('Video Title', ''))[:200],
                    'short_summary': str(fields.get('Short Summary', ''))[:300],
                    'youtube_url': str(fields.get('YouTube URL', ''))
                }
                
                # Store in Pinecone
                vector_id = f"tennis-{record_id}"
                index.upsert(vectors=[(vector_id, embedding, metadata)])
                
                logger.info(f"SUCCESS: Stored vector {vector_id} in Pinecone")
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing record {record_id}: {str(e)}")
                continue
        
        return jsonify({
            'status': 'success',
            'message': f'Processed {processed_count} records successfully',
            'total_records': len(records),
            'processed_records': processed_count
        }), 200
        
    except Exception as e:
        logger.error(f"Error in process_airtable: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
