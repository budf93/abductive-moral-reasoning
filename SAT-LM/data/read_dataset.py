import json
import argparse
import sys

def main():
    # 1. Setup the Argument Parser
    parser = argparse.ArgumentParser(description="Read JSON data with filters")
    
    # Add arguments
    parser.add_argument('--file', type=str, default='hard_question.json', help='Path to the JSON file')
    parser.add_argument('--first', type=int, help='Read only the first N items')
    parser.add_argument('--index', type=int, help='Read data at a specific index')

    args = parser.parse_args()

    # 2. Load the data
    try:
        with open(args.file, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File {args.file} not found.")
        sys.exit(1)

    # 3. Apply Filters
    # If the user wants a specific index
    if args.index is not None:
        try:
            result = data[args.index]
            print(f"--- Data at index {args.index} ---")
            print(json.dumps(result, indent=4))
        except IndexError:
            print(f"Error: Index {args.index} is out of range.")

    # If the user wants the first N items
    elif args.first is not None:
        result = data[:args.first]
        print(f"--- First {args.first} items ---")
        print(json.dumps(result, indent=4))

    # Default: Print everything
    else:
        print(json.dumps(data, indent=4))

if __name__ == "__main__":
    main()