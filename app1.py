import requests
import time
import json
import sys

def send_to_n8n_webhook(fire_data, webhook_url):
    """
    Sends a JSON payload to the n8n webhook.
    
    Args:
        fire_data (dict): A dictionary containing fire detection data.
        webhook_url (str): The URL of the n8n webhook.
    """
    try:
        # Construct the payload to match the n8n webhook's expected structure
        payload = {
            "body": {
                "label": fire_data['label'],
                "confidence": fire_data['confidence']
            }
        }
        
        # Send the POST request to n8n webhook
        response = requests.post(
            webhook_url,
            json=payload,
            headers={
                'Content-Type': 'application/json'
            },
            timeout=10
        )
        
        # Check for successful response status
        response.raise_for_status()
        
        print(f"✅ Success! Webhook sent. Status Code: {response.status_code}")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Webhook request failed. Details: {e}")
        return False

if __name__ == "__main__":
    # Read the JSON output from the ML model's standard output
    try:
        ml_output_json = sys.stdin.read()
        model_output = json.loads(ml_output_json)
    except (IOError, json.JSONDecodeError) as e:
        print(f"❌ Error: Could not read or parse JSON from terminal. Details: {e}")
        sys.exit(1)

    # --- Main Logic to send the trigger ---
    
    # Check if the model detected a fire
    if model_output.get('label') == 'fire':
        print("🔥 Wildfire detected! Sending trigger to n8n...")
        
        # Replace with your actual n8n webhook URL
        webhook_url = "https://varshh07.app.n8n.cloud/webhook/fire-alert"
        
        success = send_to_n8n_webhook(model_output, webhook_url)
        
        if success:
            print("Message sent successfully.")
        else:
            print("Failed to send message.")
    else:
        print("No fire detected. No action needed.")



