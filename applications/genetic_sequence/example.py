"""Example usage of Genetic Sequence Detector."""

from genetic_sequence import GeneticSequenceDetector


def main():
    # Initialize detector
    detector = GeneticSequenceDetector(min_length=10)
    
    # Example image path
    image_path = "path/to/document_with_sequences.png"
    
    # Extract sequences
    sequences = detector.extract_sequences(image_path)
    
    # Display results
    print(f"Found {len(sequences)} genetic sequences:\n")
    for i, seq in enumerate(sequences, 1):
        print(f"Sequence {i}:")
        print(f"  Type: {seq['sequence_type']}")
        print(f"  Text: {seq['text']}")
        print(f"  Confidence: {seq['confidence']:.2f}")
        print(f"  BBox: {seq['bbox']}")
        print()


if __name__ == "__main__":
    main()
