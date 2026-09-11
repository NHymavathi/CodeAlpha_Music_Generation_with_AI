import os
import time
import json
from app import app
import config

def test_end_to_end():
    client = app.test_client()
    
    print("=== STEP 1: Get Dataset Info ===")
    res = client.get('/api/dataset-info')
    data = res.get_json()
    print("Dataset Info Response:", data)
    assert data['success'] is True
    assert data['num_files'] >= 4

    print("\n=== STEP 2: Preprocess Dataset ===")
    res = client.post('/api/preprocess', json={"sequence_length": 15})
    data = res.get_json()
    print("Preprocess Response:", data)
    assert data['success'] is True
    assert data['num_samples'] > 0

    print("\n=== STEP 3: Run Model Training ===")
    res = client.post('/api/train', json={
        "sequence_length": 15,
        "embedding_dim": 32,
        "lstm_units": 64,
        "epochs": 3,
        "batch_size": 16
    })
    data = res.get_json()
    print("Train Trigger Response:", data)
    assert data['success'] is True

    # Poll train status until complete
    print("Polling training progress...")
    for _ in range(30):
        time.sleep(1)
        res = client.get('/api/train-status')
        st_data = res.get_json()
        status = st_data.get('status', {})
        print(f"Status: Epoch {status.get('current_epoch')}/{status.get('total_epochs')}, Message: {status.get('message')}")
        if not status.get('is_training', False) and status.get('current_epoch', 0) >= 3:
            print("Training finished!")
            break

    assert os.path.exists(config.MODEL_SAVE_PATH), "Model file should exist after training!"

    print("\n=== STEP 4: Generate Music ===")
    res = client.post('/api/generate', json={
        "num_notes": 40,
        "temperature": 1.0
    })
    data = res.get_json()
    print("Generation Response:", data)
    assert data['success'] is True
    assert 'filename' in data

    generated_path = os.path.join(config.GENERATED_DATA_DIR, data['filename'])
    assert os.path.exists(generated_path), f"Generated file {generated_path} does not exist!"
    assert os.path.getsize(generated_path) > 0, "Generated MIDI file size is 0 bytes!"

    print(f"\n[SUCCESS] END-TO-END WORKFLOW VERIFIED SUCCESSFULLY! Generated file: {data['filename']} ({os.path.getsize(generated_path)} bytes)")

if __name__ == '__main__':
    test_end_to_end()
