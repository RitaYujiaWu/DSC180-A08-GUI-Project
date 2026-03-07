import os
from datetime import datetime
from typing import List, Tuple


def generate_html_content(current_state_image: str, similar_states: List[Tuple], individual_evaluations: List[Tuple]) -> str:
    """Generate HTML content for visualizing similar states and evaluations."""
    
    # Prepare data safely
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_img_data = current_state_image.split(',')[1] if current_state_image.startswith('data:image') else current_state_image
    num_similar = len(similar_states)
    
    # Start HTML document - use string concatenation instead of format to avoid CSS conflicts
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Similar States Analysis</title>
<style>
    body {{
        font-family: Arial, sans-serif;
        margin: 20px;
        background-color: #f5f5f5;
    }}
    .container {{
        max-width: 1200px;
        margin: 0 auto;
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }}
    .header {{
        text-align: center;
        margin-bottom: 30px;
        padding-bottom: 20px;
        border-bottom: 2px solid #e0e0e0;
    }}
    .current-state {{
        margin-bottom: 30px;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 8px;
        border-left: 4px solid #007bff;
    }}
    .similar-state {{
        margin-bottom: 30px;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 8px;
        background-color: #fafafa;
    }}
    .state-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid #eee;
    }}
    .similarity-score {{
        background-color: #007bff;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }}
    .state-images {{
        display: flex;
        gap: 20px;
        margin-bottom: 15px;
        flex-wrap: wrap;
    }}
    .state-image {{
        flex: 1;
        min-width: 200px;
        text-align: center;
    }}
    .state-image img {{
        max-width: 100%;
        height: auto;
        border: 2px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    .state-label {{
        margin-top: 10px;
        font-weight: bold;
        color: #333;
    }}
    .action-section {{
        margin: 15px 0;
        padding: 15px;
        background-color: #e3f2fd;
        border-radius: 6px;
        border-left: 4px solid #2196f3;
    }}
    .action-text {{
        font-family: monospace;
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 4px;
        white-space: pre-wrap;
        word-break: break-word;
    }}
    .evaluation-section {{
        margin: 15px 0;
        padding: 15px;
        background-color: #f3e5f5;
        border-radius: 6px;
        border-left: 4px solid #9c27b0;
    }}
    .evaluation-content {{
        white-space: pre-wrap;
        word-break: break-word;
    }}
    .metadata {{
        font-size: 0.9em;
        color: #666;
        margin-top: 10px;
    }}
    .success-indicator {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        margin-left: 10px;
    }}
    .success-true {{
        background-color: #d4edda;
        color: #155724;
    }}
    .success-false {{
        background-color: #f8d7da;
        color: #721c24;
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🔍 Similar States Analysis</h1>
        <p>Generated on: {timestamp}</p>
    </div>
    
    <div class="current-state">
        <h2>📸 Current State</h2>
        <div class="state-images">
            <div class="state-image">
                <img src="data:image/png;base64,{current_img_data}" alt="Current State">
                <div class="state-label">Current State</div>
            </div>
        </div>
    </div>
    
    <h2>🎯 Similar States Found: {num_similar}</h2>"""
    
    # Add each similar state
    for i, ((sas, similarity), (evaluation, eval_similarity)) in enumerate(zip(similar_states, individual_evaluations), 1):
        # Extract success status from evaluation
        success_status = "Unknown"
        success_class = ""
        if isinstance(evaluation, dict):
            success = evaluation.get('success', 'unknown')
            if success == True:
                success_status = "SUCCESS"
                success_class = "success-true"
            elif success == False:
                success_status = "FAILURE"
                success_class = "success-false"
        
        # Extract evaluation content
        eval_content = ""
        if isinstance(evaluation, dict):
            eval_content = evaluation.get('evaluation', str(evaluation))
        else:
            eval_content = str(evaluation)
        
        # Clean up action text and escape HTML special characters
        action_text = sas.action.replace('\n', ' ').strip()
        action_text = action_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
        
        # Escape evaluation content
        eval_content = eval_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
        
        # Get image data safely
        state1_img_data = sas.state1_image.split(',')[1] if sas.state1_image.startswith('data:image') else sas.state1_image
        state2_img_data = sas.state2_image.split(',')[1] if sas.state2_image.startswith('data:image') else sas.state2_image
        
        # Get trajectory filename safely
        trajectory_name = os.path.basename(sas.trajectory_path) if hasattr(sas, 'trajectory_path') else 'Unknown'
        
        html += f"""
    <div class="similar-state">
        <div class="state-header">
            <h3>Similar State #{i}</h3>
            <div>
                <span class="similarity-score">Similarity: {similarity:.3f}</span>
                <span class="success-indicator {success_class}">{success_status}</span>
            </div>
        </div>
        
        <div class="state-images">
            <div class="state-image">
                <img src="data:image/png;base64,{state1_img_data}" alt="State 1">
                <div class="state-label">State 1 (Before)</div>
            </div>
            <div class="state-image">
                <img src="data:image/png;base64,{state2_img_data}" alt="State 2">
                <div class="state-label">State 2 (After)</div>
            </div>
        </div>
        
        <div class="action-section">
            <h4>🎬 Action Taken:</h4>
            <div class="action-text">{action_text}</div>
        </div>
        
        <div class="evaluation-section">
            <h4>🤖 LLM Evaluation:</h4>
            <div class="evaluation-content">{eval_content}</div>
        </div>
        
        <div class="metadata">
            <strong>Trajectory:</strong> {trajectory_name}<br>
            <strong>Round Index:</strong> {sas.round_index}<br>
            <strong>Evaluation Similarity:</strong> {eval_similarity:.3f}
        </div>
    </div>
"""
    
    # Close HTML
    html += """
</div>
</body>
</html>
"""
    
    return html