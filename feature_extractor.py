def extract_features(landmarks):
    # Simple example: flatten (x,y) coordinates
    features = []
    for x,y in landmarks:
        features.append(x)
        features.append(y)
    return features
