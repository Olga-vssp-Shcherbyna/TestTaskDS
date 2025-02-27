from MnistClassifier import MnistClassifier


def run():
    algorithm = input("Enter wanted algorithm (available options - rf, nn, cnn) or 'exit' to quit: ").lower()

    if algorithm.lower() == 'exit':
        print("Exiting the program.")
        return  # Ends program if user wants us to

    try:
        print("Processing...")
        model = MnistClassifier(algorithm)
        model.train()
        print("Model accuracy: ", model.accuracy())
    except ValueError as e:
        print(e)  # Outputs "Algorithm is not found", if algorithm name is wrong
    except Exception as e:
        print("An error occurred: ", e)  # For other errors

    run()


if __name__ == '__main__':
    run()
