import os
import json
from flask import Flask, request, jsonify
from pinecone import Pinecone
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(**name**)

app = Flask(**name**)

pc = Pinecone(api_key=os.environ.get(‘PINECONE_API_KEY’))
index = pc.Index(‘tennis-knowledge-base’)

@app.route(’/store-vector’, methods=[‘POST’])
def store_vector():
try:
logger.info(f”Content-Type: {request.content_type}”)
logger.info(f”Raw data: {request.get_data(as_text=True)}”)

```
    try:
        data = request.get_json(force=True)
    except Exception as json_error:
        logger.error(f"JSON parsing error: {json_error}")
        return jsonify({'error': f'Invalid JSON: {json_error}'}), 400
    
    if not data:
        return jsonify({'error': 'No data received'}), 400
        
    logger.info(f"Parsed data: {json.dumps(data, indent=2)}")
    
    content_id = data.get('content_id')
    embedding = data.get('embedding')
    
    # Handle Make.com's comma-separated embedding format
    if embedding:
        if isinstance(embedding, str):
            # Parse comma-separated string like "0.1,0.2,0.3" into array
            try:
                embedding = [float(x.strip()) for x in embedding.split(',')]
                logger.info(f"Parsed embedding string into {len(embedding)} values")
            except Exception as e:
                logger.error(f"Could not parse embedding string: {e}")
                return jsonify({'error': f'Could not parse embedding: {e}'}), 400
        elif not isinstance(embedding, list):
            logger.error(f"Embedding type: {type(embedding)}, value: {embedding}")
            return jsonify({'error': f'Embedding must be a list or string, got {type(embedding)}'}), 400
    
    metadata = {}
    for field in ['content_id', 'content_text', 'youtube_url', 'short_summary', 
                 'key_takeaways', 'detailed_analysis', 'tennis_topics', 'skill_level',
                 'coaching_style', 'player_references', 'common_problems', 'key_tags',
                 'full_transcript', 'equipment_required', 'time_investment', 
                 'solutions_provided', 'user_keywords', 'immediate_actionable',
                 'created_time', 'video_title']:
        value = data.get(field)
        if value is not None:
            metadata[field] = str(value)
    
    if not content_id:
        return jsonify({'error': 'content_id is required'}), 400
    
    if not embedding:
        return jsonify({'error': 'embedding is required'}), 400
        
    if not isinstance(embedding, list):
        return jsonify({'error': 'embedding must be an array'}), 400
    
    vector_data = {
        'id': str(content_id),
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
    logger.error(f"Error type: {type(e)}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    return jsonify({
        'status': 'error',
        'message': str(e)
    }), 500
```

@app.route(’/health’, methods=[‘GET’])
def health_check():
return jsonify({
‘status’: ‘healthy’,
‘service’: ‘Tennis Knowledge Base Pinecone Integration’
})

if **name** == ‘**main**’:
if not os.environ.get(‘PINECONE_API_KEY’):
logger.error(“PINECONE_API_KEY environment variable is required”)
exit(1)

```
app.run(host='0.0.0.0', port=5000, debug=True)
```
